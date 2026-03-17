# Libraries
import os
import hydra 
import torch 
import json 
from tqdm import tqdm

# Custom functions 
from omegaconf import OmegaConf


def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)
_safe_globals = {
    "__builtins__": None,   # disable all other builtins
    "round": round,
}

# Now eval("…") will have access to round()
OmegaConf.register_new_resolver(
    "eval",
    lambda expr: eval(expr, _safe_globals, {})
)
OmegaConf.register_new_resolver("flatten_params", flatten_params)


# Standard training phase 
def training_phase(model, train_data_loader, loss, optimizer, device, plot, max_communication):

    if plot:
        print("\nTraining phase: ")

    # Initialize train loss and accuracy
    train_loss = 0.0
    train_accuracy = 0.0
    epoch_stats = {}

    # Set the model into training mode
    model.train()

    # Counter   
    iterations = 0
    # Forward the train set
    for batch in tqdm(train_data_loader, disable=not plot):

        # Get input and labels from batch
        batch_input = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Get batch predictions
        batch_predictions = model(batch_input)

        # Collect communication/MIMO stats if available
        if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
            batch_stats = model.channel.get_last_info()
            for k, v in batch_stats.items():
                if isinstance(v, (int, float)):
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + v

        # Get batch accuracy + handle batch compression for labels
        if hasattr(model, "compressor_module"):
            num_classes = batch_predictions.shape[-1]
            batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)
            pred_classes = torch.argmax(batch_predictions, dim=1)
            target_classes = torch.argmax(batch_labels, dim=1)
            batch_accuracy = torch.mean((pred_classes == target_classes).float()).item()
        else:
            pred_classes = torch.argmax(batch_predictions, dim=1)
            batch_accuracy = torch.mean((batch_labels == pred_classes).float()).item()

        # Get batch loss
        batch_loss = loss(batch_predictions, batch_labels)

        iterations+=1

        # Store them
        train_loss += batch_loss.detach().cpu().item()
        train_accuracy += batch_accuracy

        # Compute gradients
        batch_loss.backward()  

        # Update and zero out previous gradients
        optimizer.step()
        optimizer.zero_grad()

        # Check communication
        if model.communication > max_communication: 
            break


    # Compute average loss and accuracy
    average_train_loss = train_loss / iterations
    average_train_accuracy = train_accuracy / iterations
    average_epoch_stats = {k: v / iterations for k, v in epoch_stats.items()}

    return average_train_loss, average_train_accuracy, average_epoch_stats

# Standard validation phase
def validation_phase(model, val_data_loader, loss, device, plot):
  if plot: 
    print("Validation phase: ")

  # Set the model to evaluation mode
  model.eval()

  # Initialize loss and accuracy
  val_loss = 0.0
  val_accuracy = 0.0
  epoch_stats = {}

  # Forward val set
  with torch.no_grad():
    for batch in tqdm(val_data_loader, disable=not plot):

      # Get input and labels from batch
      batch_input = batch[0].to(device)
      batch_labels = batch[1].to(device)

      # Get predictions
      batch_predictions = model(batch_input)

      # Collect communication/MIMO stats if available
      if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
          batch_stats = model.channel.get_last_info()
          for k, v in batch_stats.items():
              if isinstance(v, (int, float)):
                  epoch_stats[k] = epoch_stats.get(k, 0.0) + v

      # Keep validation consistent with training when batch compression is active.
      if hasattr(model, "compressor_module"):
          num_classes = batch_predictions.shape[-1]
          batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)

      # Get batch loss and accuracy
      batch_loss = loss(batch_predictions, batch_labels)
      pred_classes = torch.argmax(batch_predictions, dim=1)
      if batch_labels.ndim > 1:
          target_classes = torch.argmax(batch_labels, dim=1)
      else:
          target_classes = batch_labels
      batch_accuracy = torch.mean((target_classes == pred_classes).float()).item()

      # Update val_loss and val_accuracy
      val_loss += batch_loss.cpu().item()
      val_accuracy += batch_accuracy

  # Compute average loss and accuracy
  average_val_loss = val_loss / len(val_data_loader)
  average_val_accuracy = val_accuracy / len(val_data_loader)
  average_epoch_stats = {k: v / len(val_data_loader) for k, v in epoch_stats.items()}

  return average_val_loss, average_val_accuracy, average_epoch_stats

# Standard training / validation cicle
def training_schedule(model, train_data_loader, val_data_loader, optimizer, max_communication, device, hydra_output_dir, cfg, loss=torch.nn.CrossEntropyLoss(),  plot=True, save_model=True):

    # Lists to store results 
    train_losses, train_accuracies, val_losses, val_accuracies, communication_cost = [], [], [], [], []
    train_stats_history = {}
    val_stats_history = {}
    snr_sweep_history = {}

    best_val_accuracy = -1.0
    best_stats = {}
    final_results_file = os.path.join(hydra_output_dir, "final_training_results.json")
    best_results_file = os.path.join(hydra_output_dir, "best_training_results.json")
    results = {}
    try:
        for epoch in range(1, 1000):
            torch.cuda.empty_cache()
            if plot:
                print(f"\n\nEPOCH {epoch}")

            # Training phase 
            avg_train_loss, avg_train_accuracy, train_stats = training_phase(model, train_data_loader, loss, optimizer, device, plot, max_communication)
            
            # --- Validation and SNR Sweep ---
            snr_sweep = []
            if 'eval' in cfg and 'snr_sweep' in cfg.eval:
                snr_sweep = list(cfg.eval.snr_sweep)
            elif 'communication' in cfg and 'eval' in cfg.communication and 'snr_sweep' in cfg.communication.eval:
                snr_sweep = list(cfg.communication.eval.snr_sweep)

            val_acc_dict = {}
            val_loss_dict = {}
            val_stats_dict = {}

            if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                if plot:
                    print(f"\nPerforming Validation & SNR sweep for epoch {epoch}...")
                
                original_snr = cfg.communication.comm.channel.snr_db
                for snr in snr_sweep:
                    model.channel.reconfigure({'channel': {'snr_db': snr}})
                    v_loss, v_acc, v_stats = validation_phase(model, val_data_loader, loss, device, plot=False)
                    
                    val_loss_dict[str(snr)] = v_loss
                    val_acc_dict[str(snr)] = v_acc
                    val_stats_dict[str(snr)] = v_stats
                    
                    if plot:
                        print(f"  SNR: {snr} dB -> Val loss: {v_loss:.4f}; Val accuracy: {v_acc:.4f}")
                
                model.channel.reconfigure({'channel': {'snr_db': original_snr}})

                # Average across SNRs for threshold comparisons
                avg_val_accuracy = sum(val_acc_dict.values()) / len(val_acc_dict)
                avg_val_loss = sum(val_loss_dict.values()) / len(val_loss_dict)
                repr_v_stats = val_stats_dict[str(snr_sweep[0])]

                val_losses.append(val_loss_dict)
                val_accuracies.append(val_acc_dict)
            else:
                avg_val_loss, avg_val_accuracy, repr_v_stats = validation_phase(model, val_data_loader, loss, device, plot)
                val_losses.append(avg_val_loss)
                val_accuracies.append(avg_val_accuracy)

            # Store general results 
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)
            communication_cost.append(model.communication)

            # Update best stats if current validation accuracy is better
            if avg_val_accuracy >= best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                top_imp_frac = repr_v_stats.get("mode_alloc_top_imp_frac")
                if top_imp_frac is None:
                    top_imp_frac = repr_v_stats.get("stream_alloc_top_imp_frac")
                if top_imp_frac is None:
                    top_imp_frac = train_stats.get("mode_alloc_top_imp_frac", train_stats.get("stream_alloc_top_imp_frac", 0.0))

                best_stats = {
                    "best_epoch": epoch,
                    "best_train_accuracy": avg_train_accuracy,
                    "best_train_loss": avg_train_loss,
                    "best_communication_cost": model.communication,
                    "top_imp_frac": top_imp_frac,
                    "tokens_sent": repr_v_stats.get("tokens_sent", train_stats.get("tokens_sent", 0)),
                    "symbols_sent": repr_v_stats.get("mimo_L", train_stats.get("mimo_L", 0)),
                    "symbol_rate": repr_v_stats.get("rate_symbols", train_stats.get("rate_symbols", 0.0))
                }
                
                # Handling of dictionary vs scalar for multiple SNRs
                if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                    best_stats["best_val_accuracy"] = val_acc_dict
                    best_stats["best_val_loss"] = val_loss_dict
                    for snr_str, s_stats in val_stats_dict.items():
                        for k, v in s_stats.items(): 
                            best_stats[f"best_val_{k}_snr_{snr_str}"] = v
                else:
                    best_stats["best_val_accuracy"] = avg_val_accuracy
                    best_stats["best_val_loss"] = avg_val_loss
                    for k, v in repr_v_stats.items(): 
                        best_stats[f"best_val_{k}"] = v

                for k, v in train_stats.items(): best_stats[f"best_train_{k}"] = v
                
                # Save the specific best results file
                with open(best_results_file, "w") as f:
                    json.dump(best_stats, f, indent=4)

            # Store communication stats
            for k, v in train_stats.items():
                key = f"Train {k}"
                if key not in train_stats_history: train_stats_history[key] = []
                train_stats_history[key].append(v)
                
            if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                for snr_str, s_stats in val_stats_dict.items():
                    for k, v in s_stats.items():
                        key = f"Val {k} (SNR {snr_str})"
                        if key not in val_stats_history: val_stats_history[key] = []
                        val_stats_history[key].append(v)
            else:
                for k, v in repr_v_stats.items():
                    key = f"Val {k}"
                    if key not in val_stats_history: val_stats_history[key] = []
                    val_stats_history[key].append(v)

            # Plot main results 
            if plot:
                print(f"\nTrain loss: {avg_train_loss:.4f}; Val loss: {avg_val_loss:.4f}")
                print(f"Train accuracy: {avg_train_accuracy:.2f}; Val accuracy: {avg_val_accuracy:.2f}")

            # Collect results
            results = {
                "Train losses": train_losses,
                "Train accuracies": train_accuracies,
                "Val losses": val_losses,
                "Val accuracies": val_accuracies,
                "Communication cost": communication_cost,
                "Compression" : model.compression_ratio}
            
            # Add compatibility aliases and best epoch stats to results
            results.update(train_stats_history)
            results.update(val_stats_history)
            results.update(best_stats)
            results.update(snr_sweep_history)

            # Save the results dictionary as a JSON file periodically
            with open(final_results_file, "w") as f:
                json.dump(results, f, indent=4)

            # Check communication 
            if model.communication > max_communication:
                break

    finally:
        # Final save on completion or interruption
        if results:
            with open(final_results_file, "w") as f:
                json.dump(results, f, indent=4)

    return 
 


# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")
def main(cfg):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataset parameters
    batch_size = cfg.dataset.batch_size
    max_communication = cfg.dataset.max_communication
    num_workers = cfg.dataset.get('num_workers', 8)

    # Get datasets
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset = hydra.utils.instantiate(cfg.dataset.test)

    seeds = cfg.hyperparameters.get('seeds', [42, 51, 114])
    if OmegaConf.is_list(seeds):
        seeds = list(seeds)
    elif not isinstance(seeds, list):
        seeds = [seeds]

    # Set seed for reproducibility
    for seed in seeds:

        torch.manual_seed(seed)

        # Get dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Get model
        model = hydra.utils.instantiate(cfg.model)

        # Get channel
        channel = hydra.utils.instantiate(cfg.communication.channel)


        # Apply method to the model
        model = hydra.utils.instantiate(cfg.method.model,
                                        channel = channel,
                                        split_index = cfg.hyperparameters.split_index,
                                        model=model).to(device)

        # Get optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        # Print model, dataset and method
        print(f"\n\nTraining seed {seed}: \n\n  --model: {cfg.model.model_name} \n  --dataset: {cfg.dataset.name} \n  --communication: {cfg.communication.name} \n  --method: {cfg.method.name} \n  --compression: {model.compression_ratio} \n")
        print(f"\n\nParameters:  {cfg.method.parameters}\n\n  ")
        
        # Get the current Hydra output directory
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        print(hydra_output_dir)

        # legacy compatibility
        if seed != 42:
            hydra_output_dir = hydra_output_dir.replace('prova', f'prova_{seed}')
        
        # Ensure unique hydra_output_dir by appending a suffix if it already contains results
        base_dir = hydra_output_dir
        counter = 1
        while os.path.exists(os.path.join(hydra_output_dir, "final_training_results.json")) or \
              os.path.exists(os.path.join(hydra_output_dir, "training_results.json")):
            hydra_output_dir = f"{base_dir}_{counter}"
            counter += 1
        
        if hydra_output_dir != base_dir:
            print(f"Directory already exists, saving to: {hydra_output_dir}")
        else:
            print(hydra_output_dir)

        os.makedirs(hydra_output_dir, exist_ok=True)
        # Train
        training_schedule(model, train_dataloader, val_dataloader, optimizer, max_communication, device, hydra_output_dir, cfg,
                          save_model=seed == seeds[0])

    return


# At the very bottom
if __name__ == "__main__":
    main()
