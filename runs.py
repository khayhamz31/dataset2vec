#!/usr/bin/env python
# coding: utf-8

import os
import openml
import pandas as pd
from tqdm import tqdm
from openml import config
import numpy as np
from collections import defaultdict
import random
import json
from sklearn.preprocessing import LabelEncoder


def setup_dirs(accuracies_dir):
    os.makedirs("runs/raw", exist_ok=True)
    os.makedirs("runs/filtered", exist_ok=True)
    os.makedirs("runs/algorithm_splits", exist_ok=True)
    os.makedirs(accuracies_dir, exist_ok=True)          # seed-specific accuracies
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
                print(f"⚠️ No runs found for task {task_id}. Skipping.")
                continue

            runs_df["task_id"] = task_id
            runs_df["dataset_id"] = dataset_id
            runs_df.set_index("dataset_id", inplace=True)

            output_path = os.path.join("runs/raw", f"dataset_{dataset_id}_runs.csv")
            if os.path.exists(output_path):
                existing = pd.read_csv(output_path, index_col="dataset_id")
                combined = pd.concat([existing, runs_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.to_csv(output_path)
            else:
                runs_df.to_csv(output_path)

        except Exception as e:
            print(f"Error fetching runs for task {task_id}: {e}")


def filter_runs(valid_flow_ids):
    log_rows = []
    for filename in tqdm(os.listdir("runs/raw"), desc="Filtering runs"):
        if filename.startswith("dataset_") and filename.endswith("_runs.csv"):
            dataset_id = filename.split("_")[1]
            input_path = os.path.join("runs/raw", filename)
            output_path = os.path.join("runs/filtered", filename)

            try:
                runs_df = pd.read_csv(input_path, index_col="dataset_id")
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
                log_rows.append({'dataset_id': dataset_id, 'status': f"Error: {str(e)}"})
    pd.DataFrame(log_rows).to_csv("log/filtering.csv", index=False)


def split_and_sample(flow_map, seed, accuracies_dir, sample_size=50, batch_size=50):
    """
    Split runs by algorithm and sample runs reproducibly using `seed`.
    Accuracies are saved under the seed-specific accuracies_dir (e.g., runs/accuracies_175).
    """
    rnd = random.Random(seed)
    flow_to_algorithm = {int(fid): entry["algorithm_type"] for fid, entry in flow_map.items()}

    for filename in tqdm(os.listdir("runs/filtered"), desc="Splitting by algorithm"):
        if filename.startswith("dataset_") and filename.endswith("_runs.csv"):
            dataset_id = filename.split("_")[1]
            input_path = os.path.join("runs/filtered", filename)
            try:
                runs_df = pd.read_csv(input_path)
                if 'run_id' not in runs_df.columns or 'flow_id' not in runs_df.columns:
                    continue
                runs_df.set_index("run_id", inplace=True)

                dataset_output_dir = os.path.join("runs/algorithm_splits", f"dataset_{dataset_id}")
                acc_output_dir = os.path.join(accuracies_dir, f"dataset_{dataset_id}")
                os.makedirs(dataset_output_dir, exist_ok=True)
                os.makedirs(acc_output_dir, exist_ok=True)

                # group run_ids by algorithm type (from flow map)
                alg_groups = defaultdict(list)
                for run_id, row in runs_df.iterrows():
                    flow_id = row["flow_id"]
                    algo = flow_to_algorithm.get(flow_id)
                    if algo:
                        alg_groups[algo].append(run_id)

                for algo, run_ids in alg_groups.items():
                    # Save all runs of this algo (no sampling) for reference
                    filename_out = f"{algo.lower().replace(' ', '_')}_runs.csv"
                    output_path = os.path.join(dataset_output_dir, filename_out)
                    runs_df.loc[run_ids].to_csv(output_path)

                    # Reproducible sampling
                    k = min(sample_size, len(run_ids))
                    if k == 0:
                        continue
                    sampled = rnd.sample(run_ids, k)

                    sampled_df = runs_df.loc[sampled].copy()
                    accuracy_map = {}

                    def chunks(lst, n):
                        for i in range(0, len(lst), n):
                            yield lst[i:i + n]

                    # Fetch predictive accuracy for sampled runs
                    for batch in tqdm(list(chunks(sampled, batch_size)),
                                      desc=f"Fetching accuracy for {algo} (dataset {dataset_id})",
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
                            print(f"⚠️ Accuracy fetch error: {e}")

                    sampled_df['predictive_accuracy'] = sampled_df.index.map(lambda x: accuracy_map.get(int(x)))
                    acc_path = os.path.join(acc_output_dir, f"{algo.lower().replace(' ', '_')}_accuracies.csv")
                    sampled_df.to_csv(acc_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


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
            if file.endswith("_accuracies.csv"):
                df = pd.read_csv(os.path.join(dataset_dir, file))
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
    """
    Build meta-learning targets from accuracies under the provided accuracies_dir.
    Writes to final/targets.csv by default.
    """
    merged = []
    for dataset_folder in os.listdir(accuracies_dir):
        if not dataset_folder.startswith("dataset_"):
            continue
        dataset_id = dataset_folder.split("_")[1]
        dataset_dir = os.path.join(accuracies_dir, dataset_folder)
        row = {'dataset_id': dataset_id}
        medians = {}
        for file in os.listdir(dataset_dir):
            if file.endswith("_accuracies.csv"):
                df = pd.read_csv(os.path.join(dataset_dir, file))
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

    print(f"✅ Final targets saved to {final_path}")
    print("🎯 Label encoding:")
    for algo, code in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {code} → {algo}")


def run_extract_runs_for_suite(suite_id, seed, api_key):
    """
    Orchestrates fetching runs, filtering, sampling (reproducible via `seed`),
    saving accuracies under runs/accuracies_{seed}, generating statistics, and
    building targets from those accuracies.
    """
    config.apikey = api_key

    # Seed for reproducible sampling
    random.seed(seed)

    accuracies_dir = f"runs/accuracies_{seed}"

    print("Setting up...")
    setup_dirs(accuracies_dir=accuracies_dir)

    print("Loading flow map...")
    flow_map, valid_flow_ids = load_flow_map()

    print("Downloading runs...")
    task_ids = openml.study.get_suite(suite_id).tasks
    download_runs(task_ids, valid_flow_ids)

    print("Filtering runs...")
    filter_runs(valid_flow_ids)

    print("Splitting and sampling runs...")
    split_and_sample(flow_map, seed=seed, accuracies_dir=accuracies_dir)

    print("Generating statistics...")
    generate_statistics(accuracies_dir=accuracies_dir)

    print("Building final target file...")
    build_targets(accuracies_dir=accuracies_dir, final_path="final/targets.csv")