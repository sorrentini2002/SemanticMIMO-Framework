import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# USER CONFIGURATION
# =====================================================================
# Dataset and experiment settings
DATASET_NAME = "flowers-102"              # Dataset name
SPLIT_INDEX = 3                            # Split index (1-3)
CHANNEL_TYPE = "MIMO"                      # Channel type (MIMO)
SEED = 42                                  # Random seed

# Filter which methods and variants to display
INCLUDE_VARIANTS = ["DCT", "ISW"]          # Communication variants: ["Base", "DCT", "ISW"]

# NUOVA CONFIGURAZIONE: Lista di sotto-varianti da confrontare.
# Usa "" per la variante principale (senza sotto-cartelle) e "NomeCartella" per le sotto-varianti.
# Esempio per confrontare la variante normale con l'ablation: ["", "Ablation"]
INCLUDE_SUB_VARIANTS = [""]    

INCLUDE_RANDOM = False                     # Include Random selection method?
INCLUDE_BASE_METHOD = True                 # Include Base compression method?

# Output settings
WORKSPACE_ROOT = Path(__file__).parent.parent
OUTPUT_FILENAME = "simple_comparison.png"

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
    
    # Mappa base per i metodi principali
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
        
        # Caso con sub-variant (es. "Gumbel/DCT/Ablation")
        if len(parts) >= 3:
            base_method = f"{parts[0]}/{parts[1]}"
            sub_variant = "/".join(parts[2:]) # Gestisce anche eventuali sotto-sotto-cartelle
            
            formatted_base = label_map.get(base_method, base_method)
            formatted = f"{formatted_base} ({sub_variant})"
        else:
            formatted = label_map.get(method_label, method_label)
            
    else:
        formatted = method_label

    if custom_title and custom_title not in ["Base", "DCT", "ISW", ""]:
        formatted = f"{formatted} ({custom_title})"

    return formatted

def discover_data(
    workspace_root: Path,
    dataset_name: str,
    split_index: int,
    channel_type: str,
    seed: int
) -> dict:
    """
    Discover and aggregate experimental results from the hierarchical structure.
    """
    results_path = workspace_root / "results" / dataset_name
    data = defaultdict(lambda: defaultdict(list))

    seed_dir = results_path / f"seed_{seed}"
    if not seed_dir.exists():
        return data

    # Read baseline results (no channel, no compression)
    baseline_dir = seed_dir / "Baseline"
    if baseline_dir.exists():
        res_file = baseline_dir / "best_training_results.json"
        if res_file.exists():
            result = load_json(res_file)
            if result and "best_val_accuracy" in result:
                baseline_acc = result["best_val_accuracy"]
                for snr in [-5.0, 0.0, 10.0, 20.0]:
                    data["BASELINE/clean"][snr].append((1, float(baseline_acc)))

    # Read method results for the specified split and channel
    split_dir = seed_dir / f"split_{split_index}"
    if not split_dir.exists():
        return data

    channel_dir = split_dir / channel_type
    if not channel_dir.exists():
        return data

    # Process each method directory (Gumbel, Random, base, etc.)
    for method_dir in channel_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_type = method_dir.name

        # Apply filters
        if method_type.lower() == "random" and not INCLUDE_RANDOM:
            continue
        if method_type.lower() == "base" and not INCLUDE_BASE_METHOD:
            continue

        # Iterate over compression directories
        for comp_dir in method_dir.iterdir():
            if not comp_dir.is_dir() or not comp_dir.name.startswith("compression_"):
                continue

            try:
                comp_val = int(comp_dir.name.split("_")[1])
            except ValueError:
                continue

            # Check for variant subdirectories (Base/DCT/ISW)
            variant_subdirs = [
                d for d in comp_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]

            if variant_subdirs:
                for variant_dir in variant_subdirs:
                    variant_name = variant_dir.name
                    
                    if variant_name not in INCLUDE_VARIANTS:
                        continue

                    # NUOVA LOGICA MULTI SUB-VARIANT
                    for sub_v in INCLUDE_SUB_VARIANTS:
                        if sub_v == "" or sub_v is None:
                            # Variante base (senza sotto-cartella)
                            target_dir = variant_dir
                            actual_variant_name = variant_name
                        else:
                            # Sotto-variante (es. Ablation)
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
                # Structure without variants (e.g., Random)
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

    return data

# =====================================================================
# MAIN PLOTTING
# =====================================================================
if __name__ == "__main__":
    print(f"[*] Discovering results for:")
    print(f"    Dataset: {DATASET_NAME}, Split: {SPLIT_INDEX}, Seed: {SEED}, Channel: {CHANNEL_TYPE}")
    print(f"    Variants: {INCLUDE_VARIANTS}, Sub-Variants: {INCLUDE_SUB_VARIANTS}")

    data = discover_data(WORKSPACE_ROOT, DATASET_NAME, SPLIT_INDEX, CHANNEL_TYPE, SEED)

    if not data:
        print("[!] No data found!")
        exit(1)

    all_snrs = set()
    for method_data in data.values():
        all_snrs.update(method_data.keys())

    all_snrs = sorted(list(all_snrs))

    # Calculate global y-axis minimum
    y_min = 1.0
    for method_data in data.values():
        for snr_points in method_data.values():
            for comp, acc in snr_points:
                y_min = min(y_min, acc)

    y_min = max(0.0, y_min - 0.05)

    print(f"\n[OK] Methods found: {list(data.keys())}")
    print(f"[OK] SNR levels: {all_snrs}")
    print(f"[OK] Y-axis range: [{y_min:.3f}, 1.0]")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # MAPPA COLORI ESTESA: Gestisce in automatico sia le versioni base che quelle "Ablation"
    # Se aggiungi altre sotto-varianti, puoi mapparne il colore qui sotto.
    colors = {
        "baseline/clean": "red",
        "base": "red",
        "random": "gray",
        # Varianti Standard
        "gumbel/base": "blue",
        "gumbel/dct": "green",
        "gumbel/isw": "purple",
        # Varianti Ablation (sfumature diverse o stili diversi gestiti sotto)
        "gumbel/dct/ablation": "darkgreen",
        "gumbel/isw/ablation": "indigo",
        "gumbel/base/ablation": "darkblue",
    }

    # Plot each SNR scenario
    for idx, snr in enumerate(all_snrs):
        ax = axes[idx]

        # First pass: plot methods prominently
        for method_label in sorted(data.keys()):
            if method_label in ["base", "BASELINE/clean"]:
                continue

            if snr not in data[method_label]:
                continue

            points = sorted(data[method_label][snr], key=lambda x: x[0])
            comps = [p[0] for p in points]
            accs = [p[1] for p in points]

            color = colors.get(method_label.lower(), "black")
            
            # Cambia lo stile della linea dinamico se si tratta di un'Ablation
            if "ablation" in method_label.lower():
                linestyle = ":"
                marker = "o"
            else:
                linestyle = "--"
                marker = "s"

            label = format_label(method_label)
            ax.plot(
                comps,
                accs,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.5,
                markersize=5,
                label=label,
                color=color,
                alpha=1.0,
                zorder=5,
            )

        # Second pass: plot baselines
        if "BASELINE/clean" in data and snr in data["BASELINE/clean"]:
            points = sorted(data["BASELINE/clean"][snr], key=lambda x: x[0])
            if points:
                baseline_acc = points[0][1]
                ax.axhline(
                    y=baseline_acc,
                    color="red",
                    linestyle="-",
                    linewidth=2,
                    label="baseline",
                    zorder=1,
                    alpha=0.3,
                )

        if "base" in data and snr in data["base"]:
            points = sorted(data["base"][snr], key=lambda x: x[0])
            if points:
                baseline_acc = points[0][1]
                ax.axhline(
                    y=baseline_acc,
                    color="orange",
                    linestyle=":",
                    linewidth=3.5,
                    label="without compression",
                    zorder=1,
                    alpha=0.3,
                )

        ax.set_xlabel("Compression Tokens", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.set_title(f"SNR = {snr} dB", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([y_min, 1.0])

    plt.tight_layout()
    output_path = WORKSPACE_ROOT / "Plots" / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.cla()    # Pulisce gli assi
    plt.clf()    # Pulisce la figura
    plt.close('all') # Chiude tutte le finestre aperte