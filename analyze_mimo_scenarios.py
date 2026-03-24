# -*- coding: utf-8 -*-
import json
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configurazione logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_value(value_str: str) -> Any:
    """Tenta la conversione di una stringa in float o int."""
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str

def extract_config_from_overrides(folder_path: Path) -> Dict[str, Any]:
    """Svolge il parsing di .hydra/overrides.yaml catturando le eccezioni correttamente."""
    config = {}
    overrides_path = folder_path / ".hydra" / "overrides.yaml"
    
    if overrides_path.exists():
        try:
            with open(overrides_path, 'r', encoding='utf-8') as f:
                overrides = yaml.safe_load(f)
                
                if not overrides:
                    return config
                
                for item in overrides:
                    if "=" in item:
                        key, value = item.split("=", 1)
                        clean_key = key.split(".")[-1]
                        config[clean_key] = parse_value(value)
                        
        except yaml.YAMLError as e:
            logger.warning(f"Errore parsing YAML in {overrides_path}: {e}")
        except Exception as e:
            logger.error(f"Errore sconosciuto leggendo {overrides_path}: {e}")
            
    return config

def extract_fallback_config(folder_path: Path) -> Dict[str, str]:
    """Recupera la configurazione dai folder padre se overrides.yaml non è presente."""
    config = {}
    folder_name = folder_path.name
    if "=" in folder_name:
        parts = folder_name.split("=", 1)
        if len(parts) == 2:
            key, val = parts
            config[key] = val
    return config

def select_best_epoch_from_history(history: Dict[str, Any], criterion: str, snr_index: int = 0) -> Dict[str, Any]:
    """Seleziona l'epoca migliore dalla cronologia in base al criterio scelto."""
    accuracies = []
    
    if criterion == "average":
        accuracies = history.get("Val accuracies", [])
    elif criterion == "last":
        val_accs = history.get("Val accuracies", [])
        if not val_accs: return {}
        best_idx = len(val_accs) - 1
    elif criterion == "snr_index":
        snr_keys = [k for k in history.keys() if "Val accuracy (SNR" in k]
        if snr_keys:
            # Ordina chiavi SNR in modo numerico (es. -5, 0, 10, 20)
            sorted_keys = sorted(snr_acc_dict.keys(), key=lambda x: float(x)) if 'snr_acc_dict' in locals() else sorted(snr_keys, key=lambda x: float(x.split("(")[1].split(")")[0].split(" ")[-1]))
            if 0 <= snr_index < len(sorted_keys):
                target_key = sorted_keys[snr_index]
                accuracies = history.get(target_key, [])
            else:
                logger.warning(f"Indice SNR {snr_index} non valido. Fallback su average.")
                accuracies = history.get("Val accuracies", [])
        else:
            accuracies = history.get("Val accuracies", [])
    elif criterion == "max_noise":
        snr_keys = [k for k in history.keys() if "Val accuracy (SNR" in k]
        if snr_keys:
            target_key = sorted(snr_keys, key=lambda x: float(x.split("(")[1].split(")")[0].split(" ")[-1]))[0]
            accuracies = history.get(target_key, [])
        else:
            accuracies = history.get("Val accuracies", [])
    elif criterion == "min_noise":
        snr_keys = [k for k in history.keys() if "Val accuracy (SNR" in k]
        if snr_keys:
            target_key = sorted(snr_keys, key=lambda x: float(x.split("(")[1].split(")")[0].split(" ")[-1]), reverse=True)[0]
            accuracies = history.get(target_key, [])
        else:
            accuracies = history.get("Val accuracies", [])

    if not accuracies and criterion != "last":
        val_accs = history.get("Val accuracies", [])
        if not val_accs: return {}
        best_idx = len(val_accs) - 1
    elif criterion != "last":
        best_idx = accuracies.index(max(accuracies))

    best_stats = {"best_epoch": best_idx + 1}
    for k, v in history.items():
        if isinstance(v, list) and len(v) > best_idx:
            clean_key = k.replace("Train ", "best_train_").replace("Val ", "best_val_")
            if k == "Train accuracies": clean_key = "best_train_accuracy"
            if k == "Train losses": clean_key = "best_train_loss"
            if k == "Val accuracies": clean_key = "best_val_accuracy"
            if k == "Val losses": clean_key = "best_val_loss"
            if k == "Communication cost": clean_key = "best_communication_cost"
            best_stats[clean_key] = v[best_idx]
    
    snr_acc_dict = {}
    snr_loss_dict = {}
    snr_keys = [k for k in history.keys() if "(SNR " in k]
    for k in snr_keys:
        if len(history[k]) > best_idx:
            val = history[k][best_idx]
            snr_str = k.split("(SNR ")[1].split(")")[0]
            if "accuracy" in k: snr_acc_dict[snr_str] = val
            if "loss" in k: snr_loss_dict[snr_str] = val
    
    if snr_acc_dict: best_stats["best_val_accuracy"] = snr_acc_dict
    if snr_loss_dict: best_stats["best_val_loss"] = snr_loss_dict

    return best_stats

def parse_results_to_dict(base_dir: Path, force_reanalyze: bool = False, criterion: str = "average", snr_index: int = 0) -> List[Dict[str, Any]]:
    """Cerca e raccoglie tutti i risultati dalle cartelle specificate."""
    if force_reanalyze:
        logger.info(f"Modalità REANALYZE attiva. Ricerca final_training_results.json (criterio: {criterion})...")
        all_files = list(base_dir.rglob("final_training_results.json"))
    else:
        all_files = list(base_dir.rglob("best_training_results.json"))
        if not all_files:
            all_files = list(base_dir.rglob("final_training_results.json"))

    if not all_files:
        logger.warning(f"Nessun file risultato trovato in {base_dir}")
        return []

    aggregated_data = []
    for fp in all_files:
        folder = fp.parent
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Errore leggendo JSON in {fp}: {e}")
            continue

        if force_reanalyze or "Train losses" in data:
            metrics = select_best_epoch_from_history(data, criterion, snr_index)
        else:
            metrics = data

        if not metrics: continue

        config = extract_config_from_overrides(folder)
        if not config:
            current = folder
            while current != base_dir and current.name not in ["results", "multirun", "analysis_results"]:
                fb = extract_fallback_config(current)
                for k, v in fb.items():
                    if k not in config:
                        config[k] = v
                current = current.parent

        entry = {
            "source_path": str(fp),
            "conditions": config,
            "best_epoch_stats": metrics
        }
        aggregated_data.append(entry)

    return aggregated_data

def analyze_parameters(all_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    all_keys = set()
    for entry in all_data:
        all_keys.update(entry["conditions"].keys())
    
    param_unique_values = {k: set() for k in all_keys}
    for entry in all_data:
        for k, v in entry["conditions"].items():
            param_unique_values[k].add(str(v))
            
    hierarchy_params = [k for k, v in param_unique_values.items() if len(v) <= 1]
    comparison_params = [k for k, v in param_unique_values.items() if len(v) > 1]
    return sorted(hierarchy_params), sorted(comparison_params)

def save_flat_results(args: argparse.Namespace, all_data: List[Dict[str, Any]], hierarchy_params: List[str], comparison_params: List[str]):
    target_dir = Path(args.base_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = "flat"
    if args.force_reanalyze:
        suffix = f"reanalyze_{args.selection_criterion}"
        if args.selection_criterion == "snr_index":
            suffix += f"_idx{args.selection_snr_index}"

    final_output_dir = target_dir / f"analysis_{suffix}"
    counter = 1
    while final_output_dir.exists():
        final_output_dir = target_dir / f"analysis_{suffix}_{counter}"
        counter += 1
        
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = final_output_dir / "comparison_results.json"
    
    constant_metadata = {}
    if all_data:
        first_conditions = all_data[0]["conditions"]
        for p in hierarchy_params:
            if p in first_conditions:
                constant_metadata[p] = first_conditions[p]

    output_payload = {
        "metadata": {
            "constant_parameters": constant_metadata,
            "variable_parameters": comparison_params,
            "total_runs": len(all_data),
            "reanalyzed": args.force_reanalyze,
            "criterion": args.selection_criterion if args.force_reanalyze else "native"
        },
        "runs": all_data
    }
        
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, indent=4)
        logger.info(f"Salvato con successo l'aggregazione di {len(all_data)} runs in: {output_file}")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dei risultati in {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Adaptive Aggregate MIMO Split Learning Results")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to multirun or results directory")
    parser.add_argument("--base_output_dir", type=str, default="analysis_results", help="Base directory for the output")
    parser.add_argument("--force_reanalyze", action="store_true", help="Ignora i file 'best' e ricalcola l'epoca migliore dalla storia")
    parser.add_argument("--selection_criterion", type=str, default="average", 
                        choices=["average", "last", "max_noise", "min_noise", "snr_index"],
                        help="Criterio per scegliere l'epoca migliore durante la ri-analisi")
    parser.add_argument("--selection_snr_index", type=int, default=0, help="Indice SNR se si usa criterion='snr_index'")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    logger.info(f"Ricerca risultati in corso per: {results_dir}")
    all_data = parse_results_to_dict(results_dir, args.force_reanalyze, args.selection_criterion, args.selection_snr_index)
    
    if not all_data:
        logger.warning("Nessun dato trovato da aggregare. Uscita.")
        return

    hierarchy_params, comparison_params = analyze_parameters(all_data)
    save_flat_results(args, all_data, hierarchy_params, comparison_params)

if __name__ == "__main__":
    main()
