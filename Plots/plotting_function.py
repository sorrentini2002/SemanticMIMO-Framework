import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================================
# USER CONFIGURATION
# =====================================================================
# Dataset and experiment settings
DATASET_NAME = "Flowers-102"              # Dataset name "CIFAR-100" or "Flowers-102"
SPLIT_INDEX = 3                            # Split index (1-3)
CHANNEL_TYPE = "MIMO"                      # Channel type (MIMO)
SEED = None                                  # Random seed

# Filter which methods and variants to display
INCLUDE_VARIANTS = ["ISW","DCT", "Base"]          # Communication variants: ["Base", "DCT", "ISW"]

# Lista di sotto-varianti da confrontare.
INCLUDE_SUB_VARIANTS = [""]    

INCLUDE_RANDOM = True                     # Include Random selection method?
INCLUDE_BASE_METHOD = True                 # Include Base compression method?

# Output settings
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_FILENAME_1 = "simple_comparison_CIFAR100.png"
OUTPUT_FILENAME_2 = "ablation_delta_comparison.png"

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def load_json(path: Path) -> dict | None:
    """Load JSON file with error handling."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def format_label(method_label: str, custom_title: str = None) -> str:
    """Format method labels for legend display."""
    label_map = {
        "Gumbel/DCT": "GS and DCT",
        "Gumbel/ISW": "GS and ISW",
        "Gumbel/Base": "GS and Base",
        "Random": "Random",
        "base": "Random"
    }

    if method_label in label_map:
        formatted = label_map[method_label]
    elif "/" in method_label:
        parts = method_label.split("/")
        if len(parts) >= 3:
            base_method = f"{parts[0]}/{parts[1]}"
            sub_variant = "/".join(parts[2:]) 
            formatted_base = label_map.get(base_method, base_method)
            formatted = f"Definitive"
        else:
            formatted = label_map.get(method_label, method_label)
    else:
        formatted = method_label

    if custom_title and custom_title not in ["Base", "DCT", "ISW", ""]:
        formatted = f"Definitive"

    return formatted

def discover_data(
    workspace_root: Path,
    dataset_name: str,
    split_index: int,
    channel_type: str,
    seed: int | None
) -> tuple[dict, list[float]]:
    """
    Discover and aggregate experimental results.
    """
    results_path = workspace_root / "results" / dataset_name
    data = defaultdict(lambda: defaultdict(list))
    baseline_clean_acc_list = []

    if seed is None:
        seed_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    else:
        seed_dirs = [results_path / f"seed_{seed}"]

    for seed_dir in seed_dirs:
        # Read baseline results
        baseline_dir = seed_dir / "Baseline"
        if baseline_dir.exists():
            res_file = baseline_dir / "best_training_results.json"
            if res_file.exists():
                result = load_json(res_file)
                if result and "best_val_accuracy" in result:
                    b_acc = result["best_val_accuracy"]
                    b_acc_val = float(b_acc.get("clean", list(b_acc.values())[0])) if isinstance(b_acc, dict) else float(b_acc)
                    baseline_clean_acc_list.append(b_acc_val)
                    for snr in [-5.0, 0.0, 10.0, 20.0]:
                        data["BASELINE/clean"][snr].append((192, b_acc_val))

        split_dir = seed_dir / f"split_{split_index}"
        if not split_dir.exists():
            continue

        channel_dir = split_dir / channel_type
        if not channel_dir.exists():
            continue

        for method_dir in channel_dir.iterdir():
            if not method_dir.is_dir(): continue
            method_type = method_dir.name
            
            if (method_type.lower() == "random" and not INCLUDE_RANDOM) or \
               (method_type.lower() == "base" and not INCLUDE_BASE_METHOD):
                continue

            for comp_dir in method_dir.iterdir():
                if not comp_dir.is_dir() or not comp_dir.name.lower().startswith("compression_"):
                    continue

                try:
                    comp_val = int(comp_dir.name.split("_")[1])
                except ValueError: continue

                variant_subdirs = [d for d in comp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

                if variant_subdirs:
                    for variant_dir in variant_subdirs:
                        if variant_dir.name not in INCLUDE_VARIANTS: continue
                        for sub_v in INCLUDE_SUB_VARIANTS:
                            target_dir = variant_dir / sub_v if sub_v else variant_dir
                            res_file = target_dir / "best_training_results.json"
                            if res_file.exists():
                                result = load_json(res_file)
                                if result and "best_val_accuracy" in result:
                                    method_label = f"{method_type}/{variant_dir.name}/{sub_v}" if sub_v else f"{method_type}/{variant_dir.name}"
                                    for snr_str, acc in result["best_val_accuracy"].items():
                                        if snr_str != "clean":
                                            data[method_label][float(snr_str)].append((comp_val, float(acc)))
                else:
                    res_file = comp_dir / "best_training_results.json"
                    if res_file.exists():
                        result = load_json(res_file)
                        if result and "best_val_accuracy" in result:
                            for snr_str, acc in result["best_val_accuracy"].items():
                                if snr_str != "clean":
                                    data[method_type][float(snr_str)].append((comp_val, float(acc)))

    return data, baseline_clean_acc_list

# =====================================================================
# MAIN PLOTTING
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Discovering results for:")
    print(f"    Dataset: {DATASET_NAME}, Split: {SPLIT_INDEX}, Seed: {SEED}, Channel: {CHANNEL_TYPE}")
    print(f"    Variants: {INCLUDE_VARIANTS}, Sub-Variants: {INCLUDE_SUB_VARIANTS}")

    data, baseline_clean_acc_list = discover_data(WORKSPACE_ROOT, DATASET_NAME, SPLIT_INDEX, CHANNEL_TYPE, SEED)

    if not data:
        print("[!] No data found!")
        exit(1)

    all_snrs = set()
    for method_data in data.values():
        all_snrs.update(method_data.keys())

    all_snrs = sorted(list(all_snrs))

    y_min = 1.0
    y_max = 0.0
    for method_data in data.values():
        for snr_points in method_data.values():
            for comp, acc in snr_points:
                y_min = min(y_min, acc)
                y_max = max(y_max, acc)
    y_min = max(0.0, y_min - 0.05)
    y_max = min(1.0, y_max + 0.05)

    print(f"\n[OK] Methods found: {list(data.keys())}")
    print(f"[OK] SNR levels: {all_snrs}")
    print(f"[OK] Y-axis range: [{y_min:.3f}, {y_max:.3f}]")
    if baseline_clean_acc_list:
        mean_base = np.mean(baseline_clean_acc_list)
        print(f"[OK] Pure Baseline Accuracy (Mean): {mean_base:.4f} across {len(baseline_clean_acc_list)} seeds")

    # =================================================================
    # ------------- IMAGE 1: PLOT LINEARE ASSOLUTO --------------------
    # =================================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    colors = {
        "baseline/clean": "red",
        "base": "red",
        "random": "gray",
        "gumbel/base": "blue",
        "gumbel/dct": "green",
        "gumbel/isw": "purple",
        "gumbel/dct/ablation": "darkgreen",
        "gumbel/isw/ablation": "indigo",
        "gumbel/base/ablation": "darkblue",
    }

    unique_methods = sorted([m for m in data.keys() if m not in ["base", "BASELINE/clean"]])
    method_styles = {}
    fallback_colors = ["#1f77b4", "#e377c2", "#17becf", "#8c564b", "#bcbd22", "#9467bd", "#2ca02c"]
    fallback_idx = 0
    markers_list = ["s", "o", "^", "D", "v", "P", "*", "X"]
    linestyles_list = ["--", ":", "-.", "-"]

    for m in unique_methods:
        m_lower = m.lower()
        parts = m_lower.split("/")
        sub_variant = "/".join(parts[2:]) if len(parts) >= 3 else ""
        sub_variant_lower = sub_variant.lower()
        sub_variants_lower = [v.lower() for v in INCLUDE_SUB_VARIANTS]
        
        if sub_variant_lower in sub_variants_lower:
            sub_v_idx = sub_variants_lower.index(sub_variant_lower)
        else:
            sub_v_idx = fallback_idx + 1

        if m_lower in colors:
            c = colors[m_lower]
        else:
            c = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
            
        ls = linestyles_list[sub_v_idx % len(linestyles_list)]
        mk = markers_list[sub_v_idx % len(markers_list)]
        method_styles[m] = {"color": c, "linestyle": ls, "marker": mk}

    for idx, snr in enumerate(all_snrs):
        ax = axes[idx]

        for method_label in sorted(data.keys()):
                if method_label in ["base", "BASELINE/clean"] or snr not in data[method_label]:
                    continue

                grouped_points = defaultdict(list)
                for comp_v, acc in data[method_label][snr]:
                    grouped_points[comp_v].append(acc)

                sorted_comp_vals = sorted(grouped_points.keys())
                comp_ratios = [cv / 192.0 for cv in sorted_comp_vals]
                
                mean_accs = [np.mean(grouped_points[cv]) for cv in sorted_comp_vals]
                min_accs = [np.min(grouped_points[cv]) for cv in sorted_comp_vals]
                max_accs = [np.max(grouped_points[cv]) for cv in sorted_comp_vals]

                style = method_styles.get(method_label, {"color": "black", "linestyle": "--", "marker": "s"})
                label = format_label(method_label)
                
                ax.plot(comp_ratios, mean_accs, marker=style["marker"], linestyle=style["linestyle"], linewidth=2.5, markersize=5, label=label, color=style["color"], alpha=1.0, zorder=5)
            
                ax.vlines(comp_ratios, min_accs, max_accs, color=style["color"], alpha=0.4, linewidth=1.5, zorder=3)
            
                ax.scatter(comp_ratios, min_accs, color=style["color"], marker='_', s=30, alpha=0.6, zorder=4)
                ax.scatter(comp_ratios, max_accs, color=style["color"], marker='_', s=30, alpha=0.6, zorder=4)

        if "BASELINE/clean" in data and snr in data["BASELINE/clean"]:
                b_accs = [p[1] for p in data["BASELINE/clean"][snr]]
                if b_accs:
                    mean_b_acc = np.mean(b_accs)
                    ax.axhline(y=mean_b_acc, color="red", linestyle="-", linewidth=2, label="baseline", zorder=1, alpha=0.3)

        if "base" in data and snr in data["base"]:
                base_accs = [p[1] for p in data["base"][snr]]
                if base_accs:
                    mean_base_acc = np.mean(base_accs)
                    ax.axhline(y=mean_base_acc, color="orange", linestyle=":", linewidth=3.5, label="without compression", zorder=1, alpha=0.3)

        ax.set_xlabel("Compression Ratio", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"SNR = {snr} dB", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    output_path_1 = WORKSPACE_ROOT / "Plots" / OUTPUT_FILENAME_1
    output_path_1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_1, dpi=150, bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close('all')
    print(f"[OK] Saved Original Plot 1 to {output_path_1}")

    # =================================================================
    # ------------- IMAGE 2: SEABORN BARPLOT DELTA --------------------
    # =================================================================

    df_list = []
    
    # --- Calcoliamo l'accuratezza di riferimento basandoci sul metodo "random" PER OGNI SNR E RATIO ---
    random_key = next((k for k in data.keys() if k.lower() == "random"), None)
    reference_acc_per_snr_ratio = defaultdict(lambda: defaultdict(list))
    reference_acc_per_snr = {} 
    
    if random_key:
        for snr, points in data[random_key].items():
            snr_accs = []
            for comp_val, acc in points:
                reference_acc_per_snr_ratio[snr][comp_val].append(acc)
                snr_accs.append(acc)
            if snr_accs:
                reference_acc_per_snr[snr] = np.mean(snr_accs)
                
        for snr in reference_acc_per_snr_ratio:
            for comp_val in reference_acc_per_snr_ratio[snr]:
                reference_acc_per_snr_ratio[snr][comp_val] = np.mean(reference_acc_per_snr_ratio[snr][comp_val])
    else:
        print("[!] 'random' method not found in data. Deltas will be calculated against 0.")
    # --------------------------------------------------------------------------------------
        
    strategy_colors = {}
    for m, style in method_styles.items():
        # Escludiamo "base" e "random" dalla mappa colori del secondo plot
        if m.lower() not in ["base", "random"]:
            strategy_colors[format_label(m)] = style["color"]

    for method_label, snr_dict in data.items():
        # Saltiamo la baseline pulita, "without compression" (base) e "random" (che è il riferimento)
        if method_label == "BASELINE/clean" or method_label.lower() in ["base", "random"]:
            continue
        
        strategy_name = format_label(method_label)
        
        for snr, points in snr_dict.items():
            for comp_val, acc in points:
                ref_acc = reference_acc_per_snr_ratio.get(snr, {}).get(comp_val, reference_acc_per_snr.get(snr, 0.0))
                
                delta = (acc - ref_acc) * 100
                
                comp_ratio = comp_val / 192.0
                ratio_str = f"{comp_ratio:.2f}"
                
                df_list.append({
                    "Strategy": strategy_name,
                    "Ratio": ratio_str,
                    "SNR": f"{int(snr)} dB",
                    "Delta": delta
                })

    df = pd.DataFrame(df_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calcolo dei limiti per l'asse Y (valori sia positivi che negativi)
    y_min_val = df["Delta"].min() if not df.empty else -5.0
    y_max_val = df["Delta"].max() if not df.empty else 5.0
    
    padding = max(abs(y_min_val), abs(y_max_val)) * 0.15 + 1.5
    y_limit_bottom = min(y_min_val - padding, -2.0)
    y_limit_top = max(y_max_val + padding, 2.0)
    
    # Sotto-Grafico 1: SNR
    sns.barplot(ax=ax1, data=df, x="SNR", y="Delta", hue="Strategy", errorbar=None, palette=strategy_colors)
    ax1.set_title("Mean Accuracy vs SNR", weight='bold')
    ax1.set_ylabel("Δ Accuracy (%)")
    
    ax1.set_ylim(y_limit_bottom, y_limit_top)
    ax1.axhline(0, color="black", linestyle="-", linewidth=1.2) 
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Etichette percentuali in VERTICALE (rotation=90)
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%+.1f%%', rotation=90, fontsize=7, padding=2)

    # Sotto-Grafico 2: Compression Ratio
    sns.barplot(ax=ax2, data=df, x="Ratio", y="Delta", hue="Strategy", errorbar=None, palette=strategy_colors)
    ax2.set_title("Mean Accuracy vs Compression Ratio", weight='bold')
    ax2.set_xlabel("Compression Ratio", labelpad=10, weight='semibold')
    ax2.set_ylabel("")
    
    ax2.set_ylim(y_limit_bottom, y_limit_top)
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Etichette percentuali in VERTICALE (rotation=90)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%+.1f%%', rotation=90, fontsize=7, padding=2)

    plt.tight_layout()
    output_path_2 = WORKSPACE_ROOT / "Plots" / OUTPUT_FILENAME_2
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Bar Plot 2 to {output_path_2}")