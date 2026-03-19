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
                        # Clean key (e.g. method.parameters.desired_compression -> desired_compression)
                        clean_key = key.split(".")[-1]
                        config[clean_key] = parse_value(value)
                        
        except yaml.YAMLError as e:
            logger.warning(f"Errore parsing YAML in {overrides_path}: {e}")
        except Exception as e:
            logger.error(f"Errore sconosciuto leggendo {overrides_path}: {e}")
            
    return config

def extract_fallback_config(folder_path: Path) -> Dict[str, str]:
    """Recupera la configurazione dai folder padre se overrides.yaml non � presente."""
    config = {}
    folder_name = folder_path.name
    # Generalizza il fallback (non solo 'comm=')
    if "=" in folder_name:
        parts = folder_name.split("=", 1)
        if len(parts) == 2:
            key, val = parts
            config[key] = val
    return config

def parse_results_to_dict(base_dir: Path) -> List[Dict[str, Any]]:
    """Cerca e raccoglie tutti i risultati dalle cartelle specificate."""
    # 1. Cerca 'best_training_results.json', fallback su 'final_training_results.json'
    all_files = list(base_dir.rglob("best_training_results.json"))
    if not all_files:
        all_files = list(base_dir.rglob("final_training_results.json"))

    if not all_files:
        logger.warning(f"Nessun file risultato trovato in {base_dir}")
        return []

    aggregated_data = []

    for fp in all_files:
        folder = fp.parent
        
        # Load the best metrics
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Decodifica JSON fallita in {fp}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Errore leggendo JSON in {fp}: {e}")
            continue

        # Extract config
        config = extract_config_from_overrides(folder)
        
        # Generalised Fallback
        if not config:
            current = folder
            # Scala l'albero fino alla root (multirun, results..) cercando folder key=value
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
    """Identifica parametri Costanti (Hierarchy) vs Variabili (Comparison)."""
    all_keys = set()
    for entry in all_data:
        all_keys.update(entry["conditions"].keys())
    
    param_unique_values = {k: set() for k in all_keys}
    for entry in all_data:
        for k, v in entry["conditions"].items():
            param_unique_values[k].add(str(v))
            
    # Decisione parametri:
    hierarchy_params = [k for k, v in param_unique_values.items() if len(v) <= 1]
    comparison_params = [k for k, v in param_unique_values.items() if len(v) > 1]
    
    return sorted(hierarchy_params), sorted(comparison_params)

def save_flat_results(args: argparse.Namespace, all_data: List[Dict[str, Any]], hierarchy_params: List[str], comparison_params: List[str]):
    """Salva i dati aggressati in una struttura piatta evitando Path too long e usando meta-dati."""
    target_dir = Path(args.base_output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Versioning unificato base
    final_output_dir = target_dir / "analysis_flat"
    counter = 1
    while final_output_dir.exists():
        final_output_dir = target_dir / f"analysis_flat_{counter}"
        counter += 1
        
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = final_output_dir / "comparison_results.json"
    
    # Estrae i parametri costanti come metadata
    constant_metadata = {}
    if all_data:
        # Essendo costanti (stesso valore per ogni run), guardiamo il primo entry
        first_conditions = all_data[0]["conditions"]
        for p in hierarchy_params:
            if p in first_conditions:
                constant_metadata[p] = first_conditions[p]

    output_payload = {
        "metadata": {
            "constant_parameters": constant_metadata,
            "variable_parameters": comparison_params,
            "total_runs": len(all_data)
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
    # Usa 'results' come default dato il folder context visibile (puoi passare --results_dir multirun)
    parser.add_argument("--results_dir", type=str, default="results", help="Path to multirun or results directory")
    parser.add_argument("--base_output_dir", type=str, default="analysis_results", help="Base directory for the output")
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    logger.info(f"Ricerca risultati in corso per: {results_dir}")
    
    all_data = parse_results_to_dict(results_dir)
    if not all_data:
        logger.warning("Nessun dato trovato da aggregare. Uscita.")
        return

    hierarchy_params, comparison_params = analyze_parameters(all_data)
    
    logger.info(f"Parametri Costanti rilevati (Metadata): {hierarchy_params}")
    logger.info(f"Parametri Variabili rilevati (Confronto): {comparison_params}")

    save_flat_results(args, all_data, hierarchy_params, comparison_params)

if __name__ == "__main__":
    main()
