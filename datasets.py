import os
import numpy as np
import pandas as pd
import openml
import json
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from tqdm import tqdm

def preprocess_features(X, missing_threshold=0.7):
    """
    Handle missing values in features with safe thresholds.
    """
    missing_pct = X.isnull().sum() / len(X)
    features_to_keep = missing_pct <= missing_threshold
    X_cleaned = X.loc[:, features_to_keep]

    sample_missing_pct = X_cleaned.isnull().sum(axis=1) / len(X_cleaned.columns)
    samples_to_keep = sample_missing_pct <= 0.5
    X_final = X_cleaned.loc[samples_to_keep].copy()

    categorical_features = X_final.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_final.select_dtypes(exclude=['object', 'category']).columns.tolist()

    if X_final.isnull().sum().sum() > 0:
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_final[categorical_features] = cat_imputer.fit_transform(X_final[categorical_features])
        if numerical_features:
            n_neighbors = min(5, len(X_final) - 1)
            if n_neighbors > 0:
                num_imputer = KNNImputer(n_neighbors=n_neighbors)
                X_final[numerical_features] = num_imputer.fit_transform(X_final[numerical_features])
            else:
                num_imputer = SimpleImputer(strategy='median')
                X_final[numerical_features] = num_imputer.fit_transform(X_final[numerical_features])

    return X_final, samples_to_keep, categorical_features, numerical_features

def encode_and_scale_features(X, categorical_features, numerical_features):
    """
    Apply one-hot encoding to categorical features and scaling to numerical features.
    """
    transformers = []
    if numerical_features:
        transformers.append(('num', MinMaxScaler(), numerical_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features))

    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        X_processed = preprocessor.fit_transform(X)
        return X_processed
    else:
        return X.values

def download_and_process_dataset(dataset_id, dataset_name, target, output_dir="test_datasets"):
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=target)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        X_processed, valid_samples, categorical_features, numerical_features = preprocess_features(X)
        y_processed = y.loc[valid_samples]

        if len(X_processed) < 10:
            print(f"Dataset {dataset_id}: Too few samples after preprocessing ({len(X_processed)})")
            return False, None

        X_final = encode_and_scale_features(X_processed, categorical_features, numerical_features)
        y_encoded = LabelEncoder().fit_transform(y_processed)

        # Save using dataset ID
        folder_path = os.path.join(output_dir, str(dataset_id))
        os.makedirs(folder_path, exist_ok=True)

        np.savetxt(os.path.join(folder_path, f"{dataset_id}_py.dat"), X_final, fmt="%.6f", delimiter=",")
        np.savetxt(os.path.join(folder_path, "labels_py.dat"), y_encoded, fmt="%d")

        preprocessing_info = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "original_shape_rows": X.shape[0],
            "original_shape_cols": X.shape[1],
            "final_shape_rows": X_final.shape[0],
            "final_shape_cols": X_final.shape[1],
            "samples_removed": len(X) - len(X_processed),
            "features_removed": X.shape[1] - X_processed.shape[1],
            "categorical_features": len(categorical_features),
            "numerical_features": len(numerical_features),
            "classes": len(np.unique(y_encoded))
        }

        return True, preprocessing_info

    except Exception as e:
        print(f"Failed to download dataset {dataset_id} ({dataset_name}): {e}")
        return False, None

def download_benchmark_suite(suite_id, output_dir="test_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)

    preprocessing_log = []
    id_name_map = {}

    try:
        suite = openml.study.get_suite(suite_id)
        print(f"Found {len(suite.data)} datasets in suite")

        successful = 0
        failed = 0

        for dataset_id in tqdm(suite.data, desc="Downloading datasets"):
            try:
                dataset = openml.datasets.get_dataset(dataset_id)
                dataset_name = dataset.name.replace(" ", "_").replace("/", "_")
                target = dataset.default_target_attribute

                id_name_map[str(dataset_id)] = dataset_name  # store as string for JSON

                success, preprocessing_info = download_and_process_dataset(
                    dataset_id, dataset_name, target, output_dir
                )

                if success:
                    successful += 1
                    preprocessing_log.append(preprocessing_info)
                else:
                    failed += 1
            except Exception as e:
                print(f"Failed to process dataset {dataset_id}: {e}")
                failed += 1

        # Save preprocessing log
        if preprocessing_log:
            log_df = pd.DataFrame(preprocessing_log)
            log_path = f"log/suite_{suite_id}_preprocessing_log.csv"
            log_df.to_csv(log_path, index=False)
            print(f"Preprocessing log saved to {log_path}")

        # Save id-name mapping
        map_path = os.path.join(output_dir, "id_name_mapping.json")
        with open(map_path, "w") as f:
            json.dump(id_name_map, f, indent=2)
        print(f"Saved ID-name mapping to {map_path}")

        return {"successful": successful, "failed": failed, "total": len(suite.data)}

    except Exception as e:
        print(f"Failed to get benchmark suite {suite_id}: {e}")
        return {"successful": 0, "failed": 0, "total": 0}

def download_datasets_from_df(df, output_dir="test_datasets"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)

    preprocessing_log = []
    id_name_map = {}
    successful = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading datasets"):
        dataset_id = row['dataset_id']
        dataset_name = row['dataset_name']
        target = row['target']

        id_name_map[str(dataset_id)] = dataset_name

        success, preprocessing_info = download_and_process_dataset(dataset_id, dataset_name, target, output_dir)
        if success:
            successful += 1
            preprocessing_log.append(preprocessing_info)
        else:
            failed += 1

    # Save log
    if preprocessing_log:
        log_df = pd.DataFrame(preprocessing_log)
        log_path = "log/custom_datasets_preprocessing_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"Preprocessing log saved to {log_path}")

    # Save id-name map
    map_path = os.path.join(output_dir, "id_name_mapping.json")
    with open(map_path, "w") as f:
        json.dump(id_name_map, f, indent=2)
    print(f"Saved ID-name mapping to {map_path}")

    return {"successful": successful, "failed": failed, "total": len(df)}
