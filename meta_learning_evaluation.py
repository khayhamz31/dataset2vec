#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_target_data(target_path="final/targets.csv"):
    """Load target labels and compute majority class baseline."""
    target = pd.read_csv(target_path, index_col=0)
    
    # Compute majority class baseline
    majority_class = target["best_algorithm"].value_counts().idxmax()
    majority_count = target["best_algorithm"].value_counts().max()
    total = len(target)
    majority_accuracy = majority_count / total
    
    print(f"Majority Class: {majority_class}")
    print(f"Majority Class Accuracy (baseline): {majority_accuracy:.2%}")
    
    return target, majority_accuracy


def load_metafeatures():
    """Load and combine different types of metafeatures."""
    d2v = pd.read_csv("qualities/d2v/test_combined_metafeatures.csv", index_col=0)
    traditional = pd.read_csv("qualities/traditional/225_qualities_processed.csv", index_col=0)
    hybrid = pd.concat([traditional, d2v], axis=1, join='inner')
    y = pd.read_csv("final/targets.csv", index_col=0)
    
    return d2v, traditional, hybrid, y


def save_datasets_with_targets(d2v, traditional, hybrid, y, output_dir="final"):
    """Add target column to datasets and save them."""
    ensure_dir(output_dir)
    
    target_column = y["best_algorithm_encoded"]
    
    d2v_with_y = d2v.join(target_column, how="inner")
    traditional_with_y = traditional.join(target_column, how="inner")
    hybrid_with_y = hybrid.join(target_column, how="inner")
    
    d2v_with_y.to_csv(os.path.join(output_dir, "d2v_with_target.csv"))
    traditional_with_y.to_csv(os.path.join(output_dir, "traditional_with_target.csv"))
    hybrid_with_y.to_csv(os.path.join(output_dir, "hybrid_with_target.csv"))
    
    print("All files saved to:", output_dir)
    return d2v_with_y, traditional_with_y, hybrid_with_y


def load_labeled_datasets(output_dir="final"):
    """Load labeled meta-features datasets."""
    d2v = pd.read_csv(os.path.join(output_dir, "d2v_with_target.csv"), index_col=0)
    traditional = pd.read_csv(os.path.join(output_dir, "traditional_with_target.csv"), index_col=0)
    hybrid = pd.read_csv(os.path.join(output_dir, "hybrid_with_target.csv"), index_col=0)
    
    print("Meta-feature datasets loaded.")
    print("Shapes:")
    print("  - d2v:", d2v.shape)
    print("  - traditional:", traditional.shape)
    print("  - hybrid:", hybrid.shape)
    
    return d2v, traditional, hybrid


def evaluate_dataset(df, seed):
    """Evaluate a dataset using Leave-One-Out cross-validation with Random Forest."""
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    loo = LeaveOneOut()
    clf = RandomForestClassifier(random_state=seed)
    
    correct = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        correct += int(y_pred[0] == y_test[0])
    
    return correct / len(X)


def run_evaluation(datasets, n_repeats=10, seed=175, save_dir=None):
    """Run evaluation on multiple datasets with multiple repetitions."""
    results = {}
    if save_dir:
        ensure_dir(save_dir)
    
    for name in sorted(datasets):
        print(f"\n Evaluating {name} meta-model ...")
        scores = []
        for rep in range(n_repeats):
            current_seed = seed + rep
            acc = evaluate_dataset(datasets[name], current_seed)
            scores.append(acc)
        results[name] = scores
        print(f"  Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        if save_dir:
            df = pd.DataFrame({
                "repetition": list(range(1, n_repeats + 1)),
                "accuracy": scores
            })
            df.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False)
    
    return results


def plot_results(results, majority_accuracy, figsize=(8, 5), save_path=None):
    """Plot evaluation results as boxplots."""
    plt.figure(figsize=figsize)
    plt.boxplot([results["d2v"], results["traditional"], results["hybrid"]],
                labels=["Dataset2Vec", "Traditional", "Hybrid"])
    
    plt.axhline(majority_accuracy, color="green", linestyle="--", alpha=0.5,
                label=f"Majority ({majority_accuracy:.2f})")
    
    plt.title("Meta-Learning Algorithm Selection Accuracy")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def run_complete_evaluation(n_repeats=10, seed=175, plot_figsize=(8, 5), save_plot=None):
    """Run the complete meta-learning evaluation pipeline."""
    target, majority_accuracy = load_target_data()
    d2v, traditional, hybrid, y = load_metafeatures()
    save_datasets_with_targets(d2v, traditional, hybrid, y)
    d2v_labeled, traditional_labeled, hybrid_labeled = load_labeled_datasets()
    
    datasets = {
        "d2v": d2v_labeled,
        "traditional": traditional_labeled,
        "hybrid": hybrid_labeled
    }
    
    save_dir = f"models_{seed}"
    results = run_evaluation(datasets, n_repeats=n_repeats, seed=seed, save_dir=save_dir)
    
    plot_results(results, majority_accuracy, figsize=plot_figsize, save_path=save_plot)
    
    print(f"\nResults saved to: {save_dir}/")
    return results, majority_accuracy


if __name__ == "__main__":
    results, majority_accuracy = run_complete_evaluation()
