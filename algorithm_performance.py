import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define classifiers
classifiers = {
    "SVC": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# Define scoring metrics
scoring_metrics = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, average='macro'),
    'roc_auc': 'roc_auc_ovr',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro')
}

def evaluate_algorithms_on_dataset(X, y, dataset_name):
    """Evaluate all algorithms on a single dataset using cross-validation"""
    results = {"dataset": dataset_name}
    n_classes = len(np.unique(y))

    # Use appropriate CV strategy
    cv = StratifiedKFold(n_splits=min(5, min(pd.Series(y).value_counts())),
                         shuffle=True, random_state=42)

    print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes)")

    for clf_name, clf in classifiers.items():
        try:
            scores = cross_validate(clf, X, y, cv=cv, scoring=scoring_metrics, n_jobs=-1)

            for metric in scoring_metrics.keys():
                key = f"{clf_name}_{metric}"
                results[key] = np.mean(scores[f'test_{metric}'])

            print(f"  {clf_name:15}: " + ', '.join(
                [f"{metric}={results[f'{clf_name}_{metric}']:.4f}" for metric in scoring_metrics.keys()]
            ))

        except Exception as e:
            print(f"  {clf_name:15}: FAILED ({str(e)[:30]}...)")
            for metric in scoring_metrics.keys():
                results[f"{clf_name}_{metric}"] = None

    return results

def create_dataset2vec_files(results_df, output_dir="algorithm_selection_data", selection_metric="f1"):
    """Create Dataset2Vec-compatible files"""
    os.makedirs(output_dir, exist_ok=True)

    # Create performance files for each dataset
    for _, row in results_df.iterrows():
        dataset_name = row['dataset']
        performance_vector = []

        for clf_name in classifiers.keys():
            score = row.get(f"{clf_name}_{selection_metric}", 0.0)
            score = score if pd.notna(score) else 0.0
            performance_vector.append(score)

        output_file = os.path.join(output_dir, f"{dataset_name}.txt")
        with open(output_file, 'w') as f:
            f.write(' '.join(f"{score:.6f}" for score in performance_vector))

    # Create algorithm selection labels
    algorithm_labels = []
    for _, row in results_df.iterrows():
        scores = {
            clf: row[f"{clf}_{selection_metric}"]
            for clf in classifiers.keys()
            if pd.notna(row[f"{clf}_{selection_metric}"])
        }

        if scores:
            best_algorithm = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_algorithm]

            label_row = {"dataset": row['dataset'], "best_algorithm": best_algorithm, "best_score": best_score}
            for clf_name in classifiers.keys():
                label_row[f"is_{clf_name}"] = 1 if clf_name == best_algorithm else 0

            algorithm_labels.append(label_row)

    labels_df = pd.DataFrame(algorithm_labels)
    labels_df.to_csv(os.path.join(output_dir, "algorithm_selection_labels.csv"), index=False)

    print(f"\n✅ Created Dataset2Vec files in '{output_dir}/':")
    print(f"   📁 {len(results_df)} dataset performance files (.txt)")
    print(f"   📁 algorithm_selection_labels.csv")

    return output_dir

def print_summary(results_df):
    """Print summary statistics for each classifier and metric"""
    print("\n" + "=" * 60)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 60)

    for clf_name in classifiers.keys():
        for metric in scoring_metrics.keys():
            col = f"{clf_name}_{metric}"
            if col in results_df.columns:
                valid_scores = results_df[col].dropna()
                if len(valid_scores) > 0:
                    mean_val = valid_scores.mean()
                    std_val = valid_scores.std()
                    print(f"{col:25}: {mean_val:.4f} ± {std_val:.4f} (on {len(valid_scores)} datasets)")

    print("\n" + "-" * 60)
    print("ALGORITHM SELECTION RESULTS")
    print("-" * 60)

    algorithm_wins = {}
    for _, row in results_df.iterrows():
        scores = {
            clf: row.get(f"{clf}_f1")
            for clf in classifiers.keys()
            if pd.notna(row.get(f"{clf}_f1"))
        }

        if scores:
            best_alg = max(scores.keys(), key=lambda k: scores[k])
            algorithm_wins[best_alg] = algorithm_wins.get(best_alg, 0) + 1

    total_datasets = len(results_df)
    for clf_name in classifiers.keys():
        wins = algorithm_wins.get(clf_name, 0)
        percentage = (wins / total_datasets) * 100
        print(f"{clf_name:15}: {wins:3d} wins ({percentage:5.1f}%)")

def main():
    data_root = "datasets"
    results = []

    print("🚀 Evaluating algorithm performance on preprocessed datasets...\n")

    dataset_dirs = [d for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))]

    print(f"Found {len(dataset_dirs)} datasets to process\n")

    for dataset_name in tqdm(dataset_dirs, desc="Processing datasets"):
        dataset_path = os.path.join(data_root, dataset_name)

        preferred_x_file = f"{dataset_name}_py.dat"
        x_path = os.path.join(dataset_path, preferred_x_file) if preferred_x_file in os.listdir(dataset_path) else None

        y_path = None
        for file in os.listdir(dataset_path):
            if file == "labels_py.dat":
                y_path = os.path.join(dataset_path, file)
                break

        if not x_path:
            for file in os.listdir(dataset_path):
                if file.endswith("_py.dat") and "label" not in file:
                    x_path = os.path.join(dataset_path, file)
                    break

        if not x_path or not y_path:
            print(f"⚠️ Missing files in '{dataset_name}', skipping.")
            continue

        try:
            X = pd.read_csv(x_path, header=None)
            y = pd.read_csv(y_path, header=None).iloc[:, 0]

            if len(set(y)) <= 1 or X.shape[0] < 10:
                print(f"⏭️ Skipping '{dataset_name}' (invalid data)")
                continue

            result = evaluate_algorithms_on_dataset(X, y, dataset_name)
            results.append(result)

        except Exception as e:
            print(f"❌ Error processing '{dataset_name}': {e}")
            continue

    if not results:
        print("❌ No datasets were successfully processed!")
        return

    df_results = pd.DataFrame(results)
    df_results.to_csv("algorithm_performance_results.csv", index=False)
    print(f"\n📁 Results saved to 'algorithm_performance_results.csv'")

    create_dataset2vec_files(df_results, selection_metric="f1")
    print_summary(df_results)

    print(f"\n🎉 Successfully processed {len(df_results)} datasets!")

if __name__ == "__main__":
    main()
