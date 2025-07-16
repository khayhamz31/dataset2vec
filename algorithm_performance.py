import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_classifiers(seed: int):
    """Return a fresh dictionary of classifiers with the given seed"""
    return {
        "SVC": SVC(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed)
    }


def evaluate_algorithms_on_dataset(X, y, dataset_name, classifiers, seed):
    results = {"dataset": dataset_name}
    n_classes = len(np.unique(y))
    
    cv = StratifiedKFold(
        n_splits=min(5, min(pd.Series(y).value_counts())),
        shuffle=True,
        random_state=seed
    )
    
    print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes)")
    
    for clf_name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            results[clf_name] = mean_score
            print(f"  {clf_name:15}: {mean_score:.4f} ± {std_score:.4f}")
        except Exception as e:
            print(f"  {clf_name:15}: FAILED ({str(e)[:30]}...)")
            results[clf_name] = None
    
    return results


def create_dataset2vec_files(results_df, classifiers, output_dir="algorithm_selection_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in results_df.iterrows():
        dataset_name = row['dataset']
        performance_vector = [row.get(clf, 0.0) if pd.notna(row.get(clf)) else 0.0 for clf in classifiers.keys()]
        
        with open(os.path.join(output_dir, f"{dataset_name}.txt"), 'w') as f:
            f.write(' '.join(f"{score:.6f}" for score in performance_vector))
    
    algorithm_labels = []
    for _, row in results_df.iterrows():
        scores = {clf: row[clf] for clf in classifiers if pd.notna(row[clf])}
        if scores:
            best_algorithm = max(scores, key=scores.get)
            best_score = scores[best_algorithm]
            label_row = {
                "dataset": row['dataset'],
                "best_algorithm": best_algorithm,
                "best_score": best_score,
                **{f"is_{clf}": int(clf == best_algorithm) for clf in classifiers}
            }
            algorithm_labels.append(label_row)
    
    labels_df = pd.DataFrame(algorithm_labels)
    labels_df.to_csv(os.path.join(output_dir, "algorithm_selection_labels.csv"), index=False)
    
    print(f"\n✅ Created Dataset2Vec files in '{output_dir}/':")
    print(f"   📁 {len(results_df)} dataset performance files (.txt)")
    print(f"   📁 algorithm_selection_labels.csv")
    
    return output_dir


def print_summary(results_df, classifiers):
    print("\n" + "=" * 60)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for clf_name in classifiers.keys():
        valid_scores = results_df[clf_name].dropna()
        if len(valid_scores) > 0:
            print(f"{clf_name:15}: {valid_scores.mean():.4f} ± {valid_scores.std():.4f} (on {len(valid_scores)} datasets)")
    
    print("\n" + "-" * 60)
    print("ALGORITHM SELECTION RESULTS")
    print("-" * 60)
    
    wins = {clf: 0 for clf in classifiers}
    for _, row in results_df.iterrows():
        scores = {clf: row[clf] for clf in classifiers if pd.notna(row[clf])}
        if scores:
            winner = max(scores, key=scores.get)
            wins[winner] += 1
    
    total = len(results_df)
    for clf, count in wins.items():
        print(f"{clf:15}: {count:3d} wins ({(count/total)*100:5.1f}%)")


def main(seed: int, data_dir: str):  # ✅ added data_dir param
    set_seed(seed)
    classifiers = get_classifiers(seed)
    
    results = []

    print(f"🚀 Evaluating algorithm performance (seed={seed})...\n")
    
    dataset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(dataset_dirs)} datasets to process\n")
    
    for dataset_name in tqdm(dataset_dirs, desc="Processing datasets"):
        dataset_path = os.path.join(data_dir, dataset_name)  # ✅ uses passed-in data_dir

        preferred_x_file = f"{dataset_name}_py.dat"
        x_path = os.path.join(dataset_path, preferred_x_file) if preferred_x_file in os.listdir(dataset_path) else None

        y_path = next((os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f == "labels_py.dat"), None)

        if not x_path:
            x_path = next((os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                           if f.endswith("_py.dat") and "label" not in f), None)

        if not x_path or not y_path:
            print(f"⚠️ Missing files in '{dataset_name}', skipping.")
            continue

        try:
            X = pd.read_csv(x_path, header=None)
            y = pd.read_csv(y_path, header=None).iloc[:, 0]

            if len(set(y)) <= 1 or X.shape[0] < 10:
                print(f"⏭️ Skipping '{dataset_name}' (invalid data)")
                continue

            result = evaluate_algorithms_on_dataset(X, y, dataset_name, classifiers, seed)
            results.append(result)

        except Exception as e:
            print(f"❌ Error processing '{dataset_name}': {e}")

    if not results:
        print("❌ No datasets were successfully processed!")
        return

    df_results = pd.DataFrame(results)
    data_folder_name = os.path.basename(os.path.normpath(data_dir))  # reuse or define if not done yet
    out_file = f"{data_folder_name}_seed_{seed}.csv"
    df_results.to_csv(out_file, index=False)
    print(f"\n📁 Results saved to '{out_file}'")

    data_folder_name = os.path.basename(os.path.normpath(data_dir))
    output_dir = f"{data_folder_name}_seed_{seed}"

    create_dataset2vec_files(df_results, classifiers, output_dir=output_dir)
    print_summary(df_results, classifiers)

    print(f"\n🎉 Successfully processed {len(df_results)} datasets with seed {seed}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run algorithm evaluation with a custom random seed and dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset folders")  # ✅ new arg
    args = parser.parse_args()

    main(seed=args.seed, data_dir=args.data_dir)  # ✅ pass in data_dir