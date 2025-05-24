import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define classifiers
classifiers = {
    "SVC": SVC(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
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
            # Cross-validation accuracy
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            
            results[clf_name] = mean_score
            print(f"  {clf_name:15}: {mean_score:.4f} ± {std_score:.4f}")
            
        except Exception as e:
            print(f"  {clf_name:15}: FAILED ({str(e)[:30]}...)")
            results[clf_name] = None
    
    return results

def create_dataset2vec_files(results_df, output_dir="algorithm_selection_data"):
    """Create Dataset2Vec compatible files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance files for each dataset
    for _, row in results_df.iterrows():
        dataset_name = row['dataset']
        
        # Extract performance scores (handle None values)
        performance_vector = []
        for clf_name in classifiers.keys():
            score = row[clf_name] if pd.notna(row[clf_name]) else 0.0
            performance_vector.append(score)
        
        # Save as space-separated text file (Dataset2Vec format)
        output_file = os.path.join(output_dir, f"{dataset_name}.txt")
        with open(output_file, 'w') as f:
            f.write(' '.join(f"{score:.6f}" for score in performance_vector))
    
    # Create algorithm selection labels (which algorithm is best for each dataset)
    algorithm_labels = []
    for _, row in results_df.iterrows():
        # Find best performing algorithm
        scores = {clf: row[clf] for clf in classifiers.keys() if pd.notna(row[clf])}
        
        if scores:
            best_algorithm = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_algorithm]
            
            # Create one-hot encoding
            label_row = {"dataset": row['dataset'], "best_algorithm": best_algorithm, "best_score": best_score}
            for clf_name in classifiers.keys():
                label_row[f"is_{clf_name}"] = 1 if clf_name == best_algorithm else 0
            
            algorithm_labels.append(label_row)
    
    # Save algorithm selection labels
    labels_df = pd.DataFrame(algorithm_labels)
    labels_df.to_csv(os.path.join(output_dir, "algorithm_selection_labels.csv"), index=False)
    
    print(f"\n✅ Created Dataset2Vec files in '{output_dir}/':")
    print(f"   📁 {len(results_df)} dataset performance files (.txt)")
    print(f"   📁 algorithm_selection_labels.csv")
    
    return output_dir

def print_summary(results_df):
    """Print summary statistics"""
    
    print("\n" + "="*60)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("="*60)
    
    # Overall algorithm performance
    for clf_name in classifiers.keys():
        if clf_name in results_df.columns:
            valid_scores = results_df[clf_name].dropna()
            if len(valid_scores) > 0:
                mean_acc = valid_scores.mean()
                std_acc = valid_scores.std()
                print(f"{clf_name:15}: {mean_acc:.4f} ± {std_acc:.4f} (on {len(valid_scores)} datasets)")
    
    print("\n" + "-"*60)
    print("ALGORITHM SELECTION RESULTS")
    print("-"*60)
    
    # Count wins per algorithm
    algorithm_wins = {}
    for _, row in results_df.iterrows():
        scores = {clf: row[clf] for clf in classifiers.keys() if pd.notna(row[clf])}
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
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))]
    
    print(f"Found {len(dataset_dirs)} datasets to process\n")
    
    for dataset_name in tqdm(dataset_dirs, desc="Processing datasets"):
        dataset_path = os.path.join(data_root, dataset_name)
        
        # Load features and labels (your existing logic)
        preferred_x_file = f"{dataset_name}_py.dat"
        x_path = os.path.join(dataset_path, preferred_x_file) if preferred_x_file in os.listdir(dataset_path) else None
        
        y_path = None
        for file in os.listdir(dataset_path):
            if file == "labels_py.dat":
                y_path = os.path.join(dataset_path, file)
                break
        
        # Fallback for feature file
        if not x_path:
            for file in os.listdir(dataset_path):
                if file.endswith("_py.dat") and "label" not in file:
                    x_path = os.path.join(dataset_path, file)
                    break
        
        if not x_path or not y_path:
            print(f"⚠️ Missing files in '{dataset_name}', skipping.")
            continue
        
        try:
            # Load data (assuming it's already clean)
            X = pd.read_csv(x_path, header=None)
            y = pd.read_csv(y_path, header=None).iloc[:, 0]
            
            # Basic validation
            if len(set(y)) <= 1 or X.shape[0] < 10:
                print(f"⏭️ Skipping '{dataset_name}' (invalid data)")
                continue
            
            # Evaluate algorithms
            result = evaluate_algorithms_on_dataset(X, y, dataset_name)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing '{dataset_name}': {e}")
            continue
    
    if not results:
        print("❌ No datasets were successfully processed!")
        return
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Save main results
    df_results.to_csv("algorithm_performance_results.csv", index=False)
    print(f"\n📁 Results saved to 'algorithm_performance_results.csv'")
    
    # Create Dataset2Vec compatible files
    create_dataset2vec_files(df_results)
    
    # Print summary
    print_summary(df_results)
    
    print(f"\n🎉 Successfully processed {len(df_results)} datasets!")

if __name__ == "__main__":
    main()