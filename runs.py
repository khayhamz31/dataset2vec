#!/usr/bin/env python
# coding: utf-8

import os
import openml
import pandas as pd
import warnings
from tqdm import tqdm
from openml import config
import numpy as np
from collections import defaultdict
import random
import json
from sklearn.preprocessing import LabelEncoder


# === Helper: Read CSV with DtypeWarning logging ===
def read_csv_with_warning_log(path, **kwargs):
    """
    Wraps pd.read_csv to log DtypeWarnings with file path, column index, and name.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = pd.read_csv(path, **kwargs)
        for warning in w:
            if "DtypeWarning" in str(warning.message):
                msg = str(warning.message)
                print(f"DtypeWarning in file: {path}")
                print(f"   -> {msg}")
                try:
                    col_idx = int(msg.split("(")[1].split(")")[0])
                    if 0 <= col_idx < len(df.columns):
                        print(f"   Column index {col_idx} likely corresponds to: '{df.columns[col_idx]}'")
                except Exception:
                    pass
    return df


def setup_dirs(accuracies_dir):
    os.makedirs("runs/raw", exist_ok=True)
    os.makedirs("runs/filtered", exist_ok=True)
    os.makedirs("runs/algorithm_splits", exist_ok=True)
    os.makedirs(accuracies_dir, exist_ok=True)
    os.makedirs("runs/statistics", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("final", exist_ok=True)


def load_flow_map(path="flows/filtered_flow_algorithm_mapping_v2.json"):
    with open(path, "r") as f:
        flow_map = json.load(f)
    return flow_map, set(map(int, flow_map.keys()))


def download_runs(task_ids, valid_flow_ids):
    for task_id in tqdm(task_ids, desc="Downloading runs for tasks"):
        try:
            task = openml.tasks.get_task(task_id)
            dataset_id = task.dataset_id

            runs_df = openml.runs.list_runs(task=[task_id], output_format='dataframe')
            if runs_df.empty:
                continue

            runs_df["task_id"] = task_id
            runs_df["dataset_id"] = dataset_id
            runs_df.set_index("dataset_id", inplace=True)

            output_path = os.path.join("runs/raw", f"dataset_{dataset_id}_runs.csv")
            if os.path.exists(output_path):
                existing = read_csv_with_warning_log(output_path, index_col="dataset_id")
                combined = pd.concat([existing, runs_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.to_csv(output_path)
            else:
                runs_df.to_csv(output_path)

        except Exception as e:
            print(f"[download_runs] Error fetching runs for task {task_id}: {e}")


def filter_runs(valid_flow_ids):
    log_rows = []
    for filename in tqdm(os.listdir("runs/raw"), desc="Filtering runs"):
        if not (filename.startswith("dataset_") and filename.endswith("_runs.csv")):
            continue

        dataset_id = filename.split("_")[1]
        input_path = os.path.join("runs/raw", filename)
        output_path = os.path.join("runs/filtered", filename)

        try:
            runs_df = read_csv_with_warning_log(input_path, index_col="dataset_id")

            if 'flow_id' not in runs_df.columns:
                log_rows.append({'dataset_id': dataset_id, 'status': "No 'flow_id' column"})
                continue

            filtered_df = runs_df[runs_df['flow_id'].isin(valid_flow_ids)]
            filtered_df.to_csv(output_path)
            log_rows.append({
                'dataset_id': dataset_id,
                'before_count': len(runs_df),
                'after_count': len(filtered_df),
                'status': "Success"
            })
        except Exception as e:
            print(f"[filter_runs] Error on Dataset {dataset_id}: {e}")
            log_rows.append({'dataset_id': dataset_id, 'status': f"Error: {str(e)}"})

    pd.DataFrame(log_rows).to_csv("log/filtering.csv", index=False)


def split_and_sample(flow_map, seed, accuracies_dir, sample_size=50, batch_size=50):
    rnd = random.Random(seed)
    flow_to_algorithm = {int(fid): entry["algorithm_type"] for fid, entry in flow_map.items()}

    for filename in tqdm(os.listdir("runs/filtered"), desc="Splitting by algorithm"):
        if not (filename.startswith("dataset_") and filename.endswith("_runs.csv")):
            continue

        dataset_id = filename.split("_")[1]
        input_path = os.path.join("runs/filtered", filename)
        try:
            runs_df = read_csv_with_warning_log(input_path)

            if 'run_id' not in runs_df.columns or 'flow_id' not in runs_df.columns:
                continue
            runs_df.set_index("run_id", inplace=True)

            dataset_output_dir = os.path.join("runs/algorithm_splits", f"dataset_{dataset_id}")
            acc_output_dir = os.path.join(accuracies_dir, f"dataset_{dataset_id}")
            os.makedirs(dataset_output_dir, exist_ok=True)
            os.makedirs(acc_output_dir, exist_ok=True)

            alg_groups = defaultdict(list)
            for run_id, row in runs_df.iterrows():
                flow_id = row["flow_id"]
                algo = flow_to_algorithm.get(flow_id)
                if algo:
                    alg_groups[algo].append(run_id)

            for algo, run_ids in alg_groups.items():
                filename_out = f"{algo.lower().replace(' ', '_')}_runs.csv"
                runs_df.loc[run_ids].to_csv(os.path.join(dataset_output_dir, filename_out))

                k = min(sample_size, len(run_ids))
                if k == 0:
                    continue
                sampled = rnd.sample(run_ids, k)

                sampled_df = runs_df.loc[sampled].copy()
                accuracy_map = {}

                def chunks(lst, n):
                    for i in range(0, len(lst), n):
                        yield lst[i:i + n]

                for batch in tqdm(list(chunks(sampled, batch_size)),
                                  desc=f"Fetching accuracies ({algo}, ds {dataset_id})",
                                  leave=False):
                    try:
                        evals = openml.evaluations.list_evaluations(
                            function='predictive_accuracy',
                            runs=[int(rid) for rid in batch],
                            output_format='dataframe'
                        )
                        if not evals.empty:
                            accuracy_map.update(dict(zip(evals['run_id'], evals['value'])))
                    except Exception as e:
                        print(f"[split_and_sample] Accuracy fetch error (ds {dataset_id}, {algo}): {e}")

                sampled_df['predictive_accuracy'] = sampled_df.index.map(lambda x: accuracy_map.get(int(x)))
                sampled_df.to_csv(os.path.join(acc_output_dir, f"{algo.lower().replace(' ', '_')}_accuracies.csv"))

        except Exception as e:
            print(f"[split_and_sample] Error processing {filename}: {e}")


def generate_statistics(accuracies_dir):
    stats_dir = "runs/statistics"
    os.makedirs(stats_dir, exist_ok=True)

    for dataset_folder in os.listdir(accuracies_dir):
        if not dataset_folder.startswith("dataset_"):
            continue
        dataset_id = dataset_folder.split("_")[1]
        dataset_dir = os.path.join(accuracies_dir, dataset_folder)
        stats_path = os.path.join(stats_dir, f"dataset_{dataset_id}_statistics.csv")
        rows = []
        for file in os.listdir(dataset_dir):
            if not file.endswith("_accuracies.csv"):
                continue
            df = read_csv_with_warning_log(os.path.join(dataset_dir, file))
            if 'predictive_accuracy' not in df.columns or df['predictive_accuracy'].dropna().empty:
                continue
            accs = df['predictive_accuracy'].dropna()
            algo = file.replace("_accuracies.csv", "").replace("_", " ").title()
            rows.append({
                'algorithm': algo,
                'count': len(accs),
                'mean': accs.mean(),
                'std': accs.std(),
                'min': accs.min(),
                '25%': accs.quantile(0.25),
                '50% (median)': accs.median(),
                '75%': accs.quantile(0.75),
                'max': accs.max(),
                'top10_median': accs.sort_values(ascending=False).head(10).median()
            })
        if rows:
            pd.DataFrame(rows).set_index("algorithm").to_csv(stats_path)


def build_targets(accuracies_dir, final_path="final/targets.csv"):
    merged = []
    for dataset_folder in os.listdir(accuracies_dir):
        if not dataset_folder.startswith("dataset_"):
            continue
        dataset_id = dataset_folder.split("_")[1]
        dataset_dir = os.path.join(accuracies_dir, dataset_folder)
        row = {'dataset_id': dataset_id}
        medians = {}
        for file in os.listdir(dataset_dir):
            if not file.endswith("_accuracies.csv"):
                continue
            df = read_csv_with_warning_log(os.path.join(dataset_dir, file))
            if 'predictive_accuracy' not in df.columns or df['predictive_accuracy'].dropna().empty:
                continue
            accs = df['predictive_accuracy'].dropna()
            algo = file.replace("_accuracies.csv", "").replace("_", " ").title()
            medians[algo] = accs.sort_values(ascending=False).head(10).median()
        if medians:
            row.update(medians)
            row['best_algorithm'] = max(medians.items(), key=lambda x: x[1])[0]
            merged.append(row)

    df = pd.DataFrame(merged).set_index("dataset_id")
    le = LabelEncoder()
    df["best_algorithm_encoded"] = le.fit_transform(df["best_algorithm"])
    df.to_csv(final_path)

    print(f"Targets saved: {final_path}")
    print("Label encoding:")
    for algo, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {code} -> {algo}")


def run_extract_runs_for_suite(suite_id, seed, api_key):
    print(f"Pipeline start | suite={suite_id}, seed={seed}")
    config.apikey = api_key
    random.seed(seed)
    accuracies_dir = f"runs/accuracies_{seed}"

    setup_dirs(accuracies_dir=accuracies_dir)

    flow_map, valid_flow_ids = load_flow_map()

    task_ids = openml.study.get_suite(suite_id).tasks

    download_runs(task_ids, valid_flow_ids)

    filter_runs(valid_flow_ids)

    split_and_sample(flow_map, seed=seed, accuracies_dir=accuracies_dir)

    generate_statistics(accuracies_dir=accuracies_dir)

    build_targets(accuracies_dir=accuracies_dir, final_path="final/targets.csv")

    print("Complete")
