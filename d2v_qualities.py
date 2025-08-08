import os
import pandas as pd
import subprocess
from tqdm import tqdm

def extract_metafeatures_from_datasets(data_root="test_datasets", extraction_script="extract_meta_features.py"):
    extracted_count = 0
    skipped_count = 0

    dataset_names = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]

    for dataset_name in tqdm(dataset_names, desc="Extracting meta-features"):
        dataset_path = os.path.join(data_root, dataset_name)
        predictors_file = os.path.join(dataset_path, f"{dataset_name}_py.dat")
        labels_file = os.path.join(dataset_path, "labels_py.dat")

        if os.path.exists(predictors_file) and os.path.exists(labels_file):
            try:
                subprocess.run(
                    ["python", extraction_script, "--file", dataset_name],
                    capture_output=True, text=True, check=True
                )
                extracted_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Error extracting {dataset_name}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1

    return {"extracted": extracted_count, "skipped": skipped_count}

def combine_metafeature_files(extracted_folder="extracted", output_file="qualities/d2v/test_combined_metafeatures.csv"):
    print("=== Combining meta-features ===")

    if not os.path.exists(extracted_folder):
        print(f"Error: '{extracted_folder}' folder not found. No meta-features to combine.")
        return None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dfs = []

    for file in os.listdir(extracted_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(extracted_folder, file)
            try:
                df = pd.read_csv(file_path, index_col=0)
                df.index = [os.path.splitext(file)[0]]
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if dfs:
        combined_df = pd.concat(dfs)
        combined_df.to_csv(output_file)
        print(f"Combined {len(dfs)} files into '{output_file}'")
        print(f"Final meta-features shape: {combined_df.shape}")
        return combined_df
    else:
        print("No valid meta-feature files found to combine.")
        return None

def extract_and_combine_metafeatures(data_root="test_datasets", extracted_folder="extracted", output_file="qualities/d2v/test_combined_metafeatures.csv"):
    extract_metafeatures_from_datasets(data_root)
    combined_df = combine_metafeature_files(extracted_folder, output_file)
    return combined_df

def extract_metafeatures_for_suite(suite_id, data_root="test_datasets"):
    output_file = f"qualities/d2v/suite_{suite_id}_metafeatures.csv"
    extracted_folder = "extracted"

    print(f"Extracting meta-features for suite {suite_id}...")
    combined_df = extract_and_combine_metafeatures(data_root, extracted_folder, output_file)

    if combined_df is not None:
        print(f"Suite {suite_id} meta-features saved to '{output_file}'")

    return combined_df

# Example usage
if __name__ == "__main__":
    combined_df = extract_and_combine_metafeatures()
    # Or: extract_metafeatures_for_suite(99)
