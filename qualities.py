import openml
import os
import pandas as pd
from sklearn.impute import KNNImputer

def extract_suite_qualities(suite_num, output_dir="qualities/traditional", impute=True):
    """
    Extract and process dataset qualities/meta-features for all datasets in an OpenML benchmark suite.
    
    Args:
        suite_num (int): OpenML suite ID
        output_dir (str): Directory to save the CSV files
        impute (bool): Whether to perform missing data imputation
    
    Returns:
        pd.DataFrame: Processed DataFrame with dataset qualities
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure target folder exists

    # Get benchmark suite
    benchmark_suite = openml.study.get_suite(suite_num)
    print(f'Benchmark Suite: {benchmark_suite.name}')
    
    # Get unique dataset IDs from tasks
    dataset_ids = {openml.tasks.get_task(task_id).dataset_id for task_id in benchmark_suite.tasks}
    quality_list = openml.datasets.list_qualities()
    
    # Extract qualities for each dataset
    rows = []
    for dataset_id in sorted(dataset_ids):
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_qualities=True)
            qualities = dataset.qualities
            
            row = {"dataset_id": dataset_id}
            for q in quality_list:
                row[q] = qualities.get(q, None)
            rows.append(row)
            
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}: {e}")
    
    # Create DataFrame and save raw version
    df = pd.DataFrame(rows)
    raw_output_path = os.path.join(output_dir, f"{suite_num}_qualities_raw.csv")
    df.to_csv(raw_output_path, index=False)
    
    # Report initial missing data
    features_only = df.drop(columns=["dataset_id"])
    initial_missing = features_only.isnull().sum().sum()
    total_cells = features_only.size
    print(f"Initial missing data: {initial_missing}/{total_cells} ({initial_missing/total_cells*100:.1f}%)")
    
    if impute:
        df_processed = analyze_and_impute_missing_data(df, suite_num, output_dir)
        return df_processed
    else:
        print(f'Raw qualities saved to {raw_output_path}')
        return df

def analyze_and_impute_missing_data(df, suite_num, output_dir):
    """
    Analyze missing data patterns and perform imputation.
    
    Args:
        df (pd.DataFrame): Raw qualities DataFrame
        suite_num (int): Suite number for file naming
        output_dir (str): Output directory
    
    Returns:
        pd.DataFrame: Imputed DataFrame
    """
    features_only = df.drop(columns=["dataset_id"])
    initial_missing = features_only.isnull().sum().sum()
    initial_total = features_only.size
    
    print(f"\n--- Missing Data Processing ---")
    print(f"Initial state: {initial_missing}/{initial_total} missing ({initial_missing/initial_total*100:.1f}%)")
    
    # Remove completely empty columns
    completely_empty_cols = features_only.columns[features_only.isnull().all()].tolist()
    features_cleaned = features_only.drop(columns=completely_empty_cols)
    
    after_cleaning_missing = features_cleaned.isnull().sum().sum()
    after_cleaning_total = features_cleaned.size
    
    print(f"After removing {len(completely_empty_cols)} empty columns: {after_cleaning_missing}/{after_cleaning_total} missing ({after_cleaning_missing/after_cleaning_total*100:.1f}%)")
    print(f"Shape change: {features_only.shape} → {features_cleaned.shape}")
    
    # Calculate missing data statistics
    missing_percentage = (features_cleaned.isnull().sum() / len(features_cleaned)) * 100
    
    # Categorize features by missing data severity
    reliable_features = (missing_percentage < 10).sum()
    problematic_features = (missing_percentage > 50).sum()
    print(f"Feature quality: {reliable_features} reliable (<10% missing), {problematic_features} problematic (>50% missing)")
    
    # Perform KNN imputation
    print(f"Applying KNN imputation (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    features_imputed = imputer.fit_transform(features_cleaned)
    
    # Create final DataFrame
    df_imputed = pd.DataFrame(features_imputed, columns=features_cleaned.columns)
    df_imputed.insert(0, "dataset_id", df["dataset_id"])
    
    final_missing = df_imputed.drop(columns=["dataset_id"]).isnull().sum().sum()
    print(f"After imputation: {final_missing} missing values remaining")
    
    # Save processed version
    processed_output_path = os.path.join(output_dir, f"{suite_num}_qualities_processed.csv")
    df_imputed.to_csv(processed_output_path, index=False)
    
    print(f"Final shape: {df_imputed.shape}")
    print(f"Processed qualities saved to {processed_output_path}")
    
    return df_imputed

def get_missing_data_summary(df):
    """
    Generate detailed missing data analysis for a qualities DataFrame.
    
    Args:
        df (pd.DataFrame): Qualities DataFrame
    
    Returns:
        dict: Summary statistics about missing data
    """
    features_only = df.drop(columns=["dataset_id"])
    missing_percentage = (features_only.isnull().sum() / len(features_only)) * 100
    missing_per_dataset = features_only.isnull().sum(axis=1)
    
    return {
        'total_features': len(features_only.columns),
        'total_missing_cells': features_only.isnull().sum().sum(),
        'missing_percentage_overall': features_only.isnull().sum().sum() / features_only.size * 100,
        'features_by_missing_range': {
            '0%': (missing_percentage == 0).sum(),
            '1-25%': ((missing_percentage > 0) & (missing_percentage <= 25)).sum(),
            '26-50%': ((missing_percentage > 25) & (missing_percentage <= 50)).sum(),
            '51-75%': ((missing_percentage > 50) & (missing_percentage <= 75)).sum(),
            '76-99%': ((missing_percentage > 75) & (missing_percentage < 100)).sum(),
            '100%': (missing_percentage == 100).sum()
        },
        'missing_per_dataset_stats': {
            'min': missing_per_dataset.min(),
            'median': missing_per_dataset.median(),
            'max': missing_per_dataset.max()
        }
    }
