#!/usr/bin/env python3
import os
import sys
import re
import ast
import csv
from pathlib import Path

def extract_metrics(log_file_path, output_csv_path):
    log_path = Path(log_file_path)
    if not log_path.exists():
        print(f"Error: Log file '{log_file_path}' not found.")
        sys.exit(1)

    print(f"Extracting metrics from {log_file_path} into {output_csv_path}...")

    # Regex pattern to capture the step number and the dictionary payload
    # Matches "perf <step_num>:" followed by the dictionary structure
    pattern = re.compile(r"train_metric_utils.+perf\s+(\d+):\s*(\{.*\})")

    all_data = []
    all_keys = set()

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = match.group(1)
                dict_str = match.group(2)
                try:
                    # Safely parse the string representation of the Python dictionary
                    metrics_dict = ast.literal_eval(dict_str)
                    
                    # Clean the keys by removing the "perf/" prefix
                    cleaned_metrics = {k.replace("perf/", ""): v for k, v in metrics_dict.items()}
                    cleaned_metrics["step"] = int(step) # Convert to int for proper sorting later
                    
                    all_data.append(cleaned_metrics)
                    all_keys.update(cleaned_metrics.keys())
                except (ValueError, SyntaxError):
                    # Skip any corrupted or half-written log lines silently
                    continue

    if not all_data:
        print("No matching performance metrics found in the log file.")
        return

    print(f"NK_DEBUG")
    print(f"all_data[0]: {all_data[0]}")

    # Sort data by step number to ensure chronologically ordered rows
    all_data.sort(key=lambda x: x["step"])

    # Organize headers: "step" first, followed by all other discovered metrics alphabetically
    headers = ["step"] + sorted([k for k in all_keys if k != "step"])

    # Write the compiled dictionary array out to the CSV
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)

    print(f"Successfully processed {len(all_data)} steps.")

if __name__ == "__main__":
    # Allow running from command line or falling back to default names
    base_path = "/mnt/lustre/gcp640426-lustre1/aisg/users/karthik/model_training_team/slime_test/logs"
    job_id = sys.argv[1] if len(sys.argv) > 1 else "1618"
    file_in = os.path.join(base_path, job_id, "node_0.log")
    file_out = os.path.join(base_path, job_id, "metrics.csv")
    
    if len(sys.argv) == 1 and not Path(file_in).exists():
        print("Usage: python3 extract_metrics.py <path_to_log_file> [output_csv_file]")
        sys.exit(1)
        
    extract_metrics(file_in, file_out)
