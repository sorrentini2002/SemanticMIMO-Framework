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
DATASET_NAME = "flowers-102"              # Dataset name
SPLIT_INDEX = 3                            # Split index (1-3)
CHANNEL_TYPE = "MIMO"                      # Channel type (MIMO)
SEED = 42                                  # Random seed

# Filter which methods and variants to display
INCLUDE_VARIANTS = ["DCT"]          # Communication variants: ["Base", "DCT", "ISW"]

# Lista di sotto-varianti da confrontare.
INCLUDE_SUB_VARIANTS = ["Definitive", "Ablation", "Abl_Alt", ""]    

INCLUDE_RANDOM = True                     # Include Random selection method?
INCLUDE_BASE_METHOD = True                 # Include Base compression method?

# Output settings
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_FILENAME_1 = "simple_comparison.png"
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
        "Gumbel/DCT": "Complex",
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
    seed: int
) -> tuple[dict, float | None]:
    """
    Discover and aggregate experimental results.
    """
    results_path = workspace_root / "results" / dataset_name
    data = defaultdict(lambda: defaultdict(list))
    baseline_clean_acc = None

    seed_dir = results_path / f"seed_{seed}"
    if not seed_dir.exists():
        return data, baseline_clean_acc

    # Read baseline results
    baseline_dir = seed_dir / "Baseline"
    if baseline_dir.exists():
        res_file = baseline_dir / "best_training_results.json"
        if res_file.exists():
            result = load_json(res_file)
            if result and "best_val_accuracy" in result:
                b_acc = result["best_val_accuracy"]
                if isinstance(b_acc, dict):
                    baseline_clean_acc = float(b_acc.get("clean", list(b_acc.values())[0]))
                else:
                    baseline_clean_acc = float(b_acc)
                
                for snr in [-5.0, 0.0, 10.0, 20.0]:
                    data["BASELINE/clean"][snr].append((192, baseline_clean_acc))

    split_dir = seed_dir / f"split_{split_index}"
    if not split_dir.exists():
        return data, baseline_clean_acc

    channel_dir = split_dir / channel_type
    if not channel_dir.exists():
        return data, baseline_clean_acc

    for method_dir in channel_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_type = method_dir.name

        if method_type.lower() == "random" and not INCLUDE_RANDOM:
            continue
        if method_type.lower() == "base" and not INCLUDE_BASE_METHOD:
            continue

        for comp_dir in method_dir.iterdir():
            if not comp_dir.is_dir() or not comp_dir.name.startswith("compression_"):
                continue

            try:
                comp_val = int(comp_dir.name.split("_")[1])
            except ValueError:
                continue

            variant_subdirs = [d for d in comp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

            if variant_subdirs:
                for variant_dir in variant_subdirs:
                    variant_name = variant_dir.name
                    if variant_name not in INCLUDE_VARIANTS:
                        continue

                    for sub_v in INCLUDE_SUB_VARIANTS:
                        if sub_v == "" or sub_v is None:
                            target_dir = variant_dir
                            actual_variant_name = variant_name
                        else:
                            target_dir = variant_dir / sub_v
                            actual_variant_name = f"{variant_name}/{sub_v}"

                        res_file = target_dir / "best_training_results.json"
                        if not res_file.exists():
                            continue

                        result = load_json(res_file)
                        if not result or "best_val_accuracy" not in result:
                            continue

                        accuracies = result["best_val_accuracy"]
                        method_label = f"{method_type}/{actual_variant_name}"

                        for snr_str, acc in accuracies.items():
                            if snr_str != "clean":
                                snr = float(snr_str)
                                data[method_label][snr].append((comp_val, float(acc)))
            else:
                res_file = comp_dir / "best_training_results.json"
                if not res_file.exists():
                    continue

                result = load_json(res_file)
                if not result or "best_val_accuracy" not in result:
                    continue

                accuracies = result["best_val_accuracy"]
                for snr_str, acc in accuracies.items():
                    if snr_str != "clean":
                        snr = float(snr_str)
                        data[method_type][snr].append((comp_val, float(acc)))

    return data, baseline_clean_acc

# =====================================================================
# MAIN PLOTTING
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Discovering results for:")
    print(f"    Dataset: {DATASET_NAME}, Split: {SPLIT_INDEX}, Seed: {SEED}, Channel: {CHANNEL_TYPE}")
    print(f"    Variants: {INCLUDE_VARIANTS}, Sub-Variants: {INCLUDE_SUB_VARIANTS}")

    data, baseline_clean_acc = discover_data(WORKSPACE_ROOT, DATASET_NAME, SPLIT_INDEX, CHANNEL_TYPE, SEED)

    if not data:
        print("[!] No data found!")
        exit(1)

    all_snrs = set()
    for method_data in data.values():
        all_snrs.update(method_data.keys())

    all_snrs = sorted(list(all_snrs))

    y_min = 1.0
    for method_data in data.values():
        for snr_points in method_data.values():
            for comp, acc in snr_points:
                y_min = min(y_min, acc)
    y_min = max(0.0, y_min - 0.05)

    print(f"\n[OK] Methods found: {list(data.keys())}")
    print(f"[OK] SNR levels: {all_snrs}")
    print(f"[OK] Y-axis range: [{y_min:.3f}, 1.0]")
    if baseline_clean_acc:
        print(f"[OK] Pure Baseline Accuracy: {baseline_clean_acc:.4f}")

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

            points = sorted(data[method_label][snr], key=lambda x: x[0])
            # Calcolo del Compression Ratio sull'asse X al posto dei Token assoluti
            comp_ratios = [p[0] / 192.0 for p in points]
            accs = [p[1] for p in points]

            style = method_styles.get(method_label, {"color": "black", "linestyle": "--", "marker": "s"})
            label = format_label(method_label)
            
            ax.plot(
                comp_ratios, accs, marker=style["marker"], linestyle=style["linestyle"],
                linewidth=2.5, markersize=5, label=label, color=style["color"], alpha=1.0, zorder=5
            )

        if "BASELINE/clean" in data and snr in data["BASELINE/clean"]:
            points = sorted(data["BASELINE/clean"][snr], key=lambda x: x[0])
            if points:
                b_acc = points[0][1]
                ax.axhline(y=b_acc, color="red", linestyle="-", linewidth=2, label="baseline", zorder=1, alpha=0.3)

        if "base" in data and snr in data["base"]:
            points = sorted(data["base"][snr], key=lambda x: x[0])
            if points:
                b_acc = points[0][1]
                ax.axhline(y=b_acc, color="orange", linestyle=":", linewidth=3.5, label="without compression", zorder=1, alpha=0.3)

        # Asse X rinominato in Compression Ratio per il Plot 1
        ax.set_xlabel("Compression Ratio", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"SNR = {snr} dB", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([y_min, 1.0])

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
    reference_acc = baseline_clean_acc if baseline_clean_acc is not None else 1.0

    # Genera automaticamente la mappa colori identica a quella usata nel Plot 1
    strategy_colors = {}
    for m, style in method_styles.items():
        strategy_colors[format_label(m)] = style["color"]
    
    # Forziamo il colore arancione per il caso senza compressione (come nel Plot 1)
    strategy_colors["Without Compression"] = "orange"

    for method_label, snr_dict in data.items():
        # Saltiamo SOLO la baseline pulita del database (clean), includiamo 'base'
        if method_label == "BASELINE/clean":
            continue
        
        # Mappatura uniforme delle legende con il Plot 1
        if method_label == "base":
            strategy_name = "Without Compression"
        else:
            strategy_name = format_label(method_label)
        
        for snr, points in snr_dict.items():
            for comp_val, acc in points:
                delta = (acc - reference_acc) * 100
                
                # Se è il caso 'base' il ratio è 1.00 (tutti i token), altrimenti lo calcola
                if method_label == "base":
                    comp_ratio = 1.00
                else:
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

    # Sotto-Grafico 1: SNR (Usa la mappa colori sincronizzata)
    sns.barplot(ax=ax1, data=df, x="SNR", y="Delta", hue="Strategy", errorbar=None, palette=strategy_colors)
    ax1.set_title("Mean Accuracy vs SNR", weight='bold')
    ax1.set_ylabel("Δ Accuracy (%)")
    y_limit = min(df["Delta"].min() - 2.0, -11)
    ax1.set_ylim(y_limit, 0)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Numeri inseriti alla fine delle colonne (SNR)
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%+.1f%%', rotation=90, fontsize=7, padding=2)

    # Sotto-Grafico 2: Compression Ratio (Usa la mappa colori sincronizzata)
    df_ratio = df[df["Strategy"] != "Without Compression"]
    sns.barplot(ax=ax2, data=df_ratio, x="Ratio", y="Delta", hue="Strategy", errorbar=None, palette=strategy_colors)
    ax2.set_title("Mean Accuracy vs Compression Ratio", weight='bold')
    ax2.set_xlabel("Compression Ratio", labelpad=10, weight='semibold')
    ax2.set_ylabel("")
    ax2.set_ylim(y_limit, 0)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    # Numeri inseriti alla fine delle colonne (Ratio)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%+.1f%%', rotation=90, fontsize=7, padding=2)

    plt.tight_layout()
    output_path_2 = WORKSPACE_ROOT / "Plots" / OUTPUT_FILENAME_2
    plt.savefig(output_path_2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Bar Plot 2 to {output_path_2}")