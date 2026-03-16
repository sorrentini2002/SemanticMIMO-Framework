import os
import json
import glob
import argparse
import yaml
import re

def extract_config_from_overrides(folder_path):
    """Parses .hydra/overrides.yaml to extract all run conditions generically."""
    config = {}
    
    overrides_path = os.path.join(folder_path, ".hydra", "overrides.yaml")
    if os.path.exists(overrides_path):
        with open(overrides_path, 'r') as f:
            try:
                overrides = yaml.safe_load(f)
                for item in overrides:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        # Clean key (e.g. method.parameters.desired_compression -> desired_compression)
                        clean_key = key.split(".")[-1]
                        # Try to convert to float/int if possible
                        try:
                            if "." in value: value = float(value)
                            else: value = int(value)
                        except:
                            pass
                        config[clean_key] = value
            except Exception as e:
                print(f"Warning: Could not parse overrides in {folder_path}: {e}")
                
    return config

def parse_results_to_dict(base_dir):
    """
    Scans the base directory for best_training_results.json files and aggregates them.
    Returns a list of dictionaries containing both config and metrics.
    """
    # 1. Find all best result files
    all_files = glob.glob(os.path.join(base_dir, "**", "best_training_results.json"), recursive=True)
    if not all_files:
        all_files = glob.glob(os.path.join(base_dir, "**", "final_training_results.json"), recursive=True)

    if not all_files:
        print(f"No result files found in {base_dir}")
        return []

    aggregated_data = []

    for fp in all_files:
        folder = os.path.dirname(fp)
        
        # Load the best metrics
        with open(fp, 'r') as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON in {fp}")
                continue

        # Extract config (conditions)
        config = extract_config_from_overrides(folder)
        
        # Fallback for non-multirun folders (extracting from folder name)
        if config["communication"] == "Unknown":
            folder_name = os.path.basename(folder)
            if "comm=" in folder_name:
                config["communication"] = folder_name.split("comm=")[-1].split("_")[0]

        # Combine into a single entry
        entry = {
            "source_path": fp,
            "conditions": config,
            "best_epoch_stats": metrics
        }
        
        aggregated_data.append(entry)

    return aggregated_data

def main():
    parser = argparse.ArgumentParser(description="Adaptive Aggregate MIMO Split Learning Results")
    parser.add_argument("--results_dir", type=str, default="multirun", help="Path to multirun or results directory")
    parser.add_argument("--base_output_dir", type=str, default="analysis_results", help="Base directory for the nested output")
    
    args = parser.parse_args()
    
    print(f"Scanning for results in: {args.results_dir}")
    all_data = parse_results_to_dict(args.results_dir)
    
    if not all_data:
        print("No data found to aggregate.")
        return

    # 1. Identify which parameters are constant and which are variables
    # We look through ALL loaded data to see how many unique values each parameter has
    all_keys = set()
    for entry in all_data:
        all_keys.update(entry["conditions"].keys())
    
    param_unique_values = {k: set() for k in all_keys}
    for entry in all_data:
        for k, v in entry["conditions"].items():
            param_unique_values[k].add(str(v))
            
    # Decision: 
    # - If value count == 1 -> HIERARCHY (Folder)
    # - If value count > 1 -> COMPARISON (Inside JSON)
    HIERARCHY_PARAMS = [k for k, v in param_unique_values.items() if len(v) == 1]
    COMPARISON_PARAMS = [k for k, v in param_unique_values.items() if len(v) > 1]
    
    print(f"Detected Hierarchy (Constants): {HIERARCHY_PARAMS}")
    print(f"Detected Comparison (Variables): {COMPARISON_PARAMS}")

    # 2. Group data by their hierarchy keys
    grouped_results = {}
    for entry in all_data:
        conds = entry["conditions"]
        path_parts = []
        # Create hierarchy based on constants
        # Sort HIERARCHY_PARAMS for consistent path structure
        for p in sorted(HIERARCHY_PARAMS):
            path_parts.append(f"{p}_{conds[p]}")
        
        hierarchy_path = os.path.join(*path_parts) if path_parts else "varied_runs"
        
        if hierarchy_path not in grouped_results:
            grouped_results[hierarchy_path] = []
        grouped_results[hierarchy_path].append(entry)

    # 3. Save each group into its nested directory
    for rel_path, items in grouped_results.items():
        target_dir = os.path.join(args.base_output_dir, rel_path)
        
        # Versioning: analysis, analysis_1, etc.
        base_analysis_dir = os.path.join(target_dir, "analysis")
        final_output_dir = base_analysis_dir
        counter = 1
        while os.path.exists(final_output_dir):
            final_output_dir = f"{base_analysis_dir}_{counter}"
            counter += 1
            
        os.makedirs(final_output_dir, exist_ok=True)
        
        output_file = os.path.join(final_output_dir, "comparison_results.json")
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=4)
            
        print(f"Saved {len(items)} runs to: {output_file}")

if __name__ == "__main__":
    main()
