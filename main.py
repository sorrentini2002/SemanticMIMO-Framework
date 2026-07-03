# Libraries
import os
import re
import random
import hydra 
import torch 
import json 
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn as nn

# Custom functions 
from omegaconf import OmegaConf


def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)
_safe_globals = {
    "__builtins__": None,
    "round": round,
}

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, _safe_globals, {}))
OmegaConf.register_new_resolver("flatten_params", flatten_params)


# ============================================================
# TRAINING PHASE
# ============================================================

def train_one_epoch(model, train_data_loader, loss, optimizer, device, plot,
                    current_epoch=1, num_epochs=1):

    if plot:
        print("\nTraining phase: ")

    train_loss = 0.0
    train_accuracy = 0.0
    epoch_stats = {}

    model.train()
    iterations = 0

    # --- SNR Range Decision (Accelerated Sampling) ---
    late_phase_start = max(1, num_epochs - num_epochs // 5)
    if current_epoch >= late_phase_start:
        snr_range = (10.0, 20.0)
    else:
        snr_range = (0.0, 20.0)

    for batch in tqdm(train_data_loader, disable=not plot):

        batch_input  = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # --- Accelerated SNR Training ---
        if hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
            dynamic_snr = random.uniform(snr_range[0], snr_range[1])
            model.channel.reconfigure({'channel': {'snr_db': dynamic_snr}})

        # --- Gumbel-Softmax step registration (tau annealing) ---
        if hasattr(model, 'compressor_module') and hasattr(model.compressor_module, 'register_step'):
            if not hasattr(model, '_gumbel_global_step'):
                model._gumbel_global_step = 0
            model.compressor_module.register_step(model._gumbel_global_step)
            model._gumbel_global_step += 1

        batch_predictions = model(batch_input)

        # Collect channel stats
        if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
            batch_stats = model.channel.get_last_info()
            for k, v in batch_stats.items():
                if isinstance(v, (int, float)):
                    epoch_stats[k] = epoch_stats.get(k, 0.0) + v

        # Labels compression
        if hasattr(model, "compressor_module"):
            num_classes  = batch_predictions.shape[-1]
            batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)
            pred_classes   = torch.argmax(batch_predictions, dim=1)
            target_classes = torch.argmax(batch_labels, dim=1)
            batch_accuracy = torch.mean((pred_classes == target_classes).float()).item()
        else:
            pred_classes   = torch.argmax(batch_predictions, dim=1)
            batch_accuracy = torch.mean((batch_labels == pred_classes).float()).item()

        batch_loss = loss(batch_predictions, batch_labels)

        # ================================================================
        # ENTROPY BOTTLENECK LOSS  (replaces old batch-diversity term)
        # ================================================================
        # gumbel_method.py stores the pre-computed bottleneck loss in
        # compressor_module.entropy_reg_loss.  In the new formulation this is:
        #
        #   entropy_bottleneck_weight * max(0, H_actual - H_target(epoch))
        #
        # When entropy is already below the target the term is zero (no penalty).
        # When it is above, we add a small push downward proportional to the excess.
        #
        # This replaces the old  loss -= weight * batch_entropy  logic, which was
        # too aggressive and caused the "eccessiva uniformità" symptom.
        #
        # NOTE: the weight is already baked into entropy_reg_loss by the wrapper,
        #       so we simply ADD it (no extra multiplier here).
        if hasattr(model, "compressor_module") and hasattr(model.compressor_module, "entropy_reg_loss"):
            ent_loss = model.compressor_module.entropy_reg_loss
            if torch.is_tensor(ent_loss) and ent_loss.requires_grad:
                batch_loss = batch_loss + ent_loss

        iterations += 1
        train_loss     += batch_loss.detach().cpu().item()
        train_accuracy += batch_accuracy

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    average_train_loss     = train_loss     / iterations
    average_train_accuracy = train_accuracy / iterations
    average_epoch_stats    = {k: v / iterations for k, v in epoch_stats.items()}

    return average_train_loss, average_train_accuracy, average_epoch_stats


# ============================================================
# VALIDATION PHASE  (unchanged)
# ============================================================

def evaluate_one_epoch(model, val_data_loader, loss, device, plot):
    if plot:
        print("Validation phase: ")

    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    epoch_stats = {}

    with torch.no_grad():
        for batch in tqdm(val_data_loader, disable=not plot):
            batch_input  = batch[0].to(device)
            batch_labels = batch[1].to(device)

            batch_predictions = model(batch_input)

            if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
                batch_stats = model.channel.get_last_info()
                for k, v in batch_stats.items():
                    if isinstance(v, (int, float)):
                        epoch_stats[k] = epoch_stats.get(k, 0.0) + v

            if hasattr(model, "compressor_module"):
                num_classes  = batch_predictions.shape[-1]
                batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)

            batch_loss   = loss(batch_predictions, batch_labels)
            pred_classes = torch.argmax(batch_predictions, dim=1)
            if batch_labels.ndim > 1:
                target_classes = torch.argmax(batch_labels, dim=1)
            else:
                target_classes = batch_labels
            batch_accuracy = torch.mean((target_classes == pred_classes).float()).item()

            val_loss     += batch_loss.cpu().item()
            val_accuracy += batch_accuracy

    average_val_loss     = val_loss     / len(val_data_loader)
    average_val_accuracy = val_accuracy / len(val_data_loader)
    average_epoch_stats  = {k: v / len(val_data_loader) for k, v in epoch_stats.items()}

    return average_val_loss, average_val_accuracy, average_epoch_stats

# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(model, val_loader, device, hydra_output_dir,
                       split_index, n_samples=4):
    """
    Generate *visual_comp_random.png*: a (n_samples × 2) grid comparing,
    for each sample image:
      - Column 1  – Original Image (Clean)
      - Column 2  – Original + GREEN overlay of Gumbel-selected patches
    """
    PATCH_GRID_COUNT = 14                       # 224 / 16 = 14 patches per side
    IMAGE_RESOLUTION_PX  = 224
    PATCH_SIZE_PX  = IMAGE_RESOLUTION_PX // PATCH_GRID_COUNT    # 16 px per patch

    # --- Collect randomly n_samples images ---------------------------
    dataset = val_loader.dataset
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), n_samples)
    
    images = []
    for idx in random_indices:
        img, _ = dataset[idx]  # Ignoriamo la label
        images.append(img)
        
    images = torch.stack(images, dim=0).to(device)  # [n, 3, 224, 224]

    # --- UNICO Forward pass (eval, no grad) -------------------------
    model.eval()
    with torch.no_grad():
        _ = model(images)

    # --- Extract Gumbel selection indices ---------------------------
    gumbel_indices = model.compressor_module.last_indices_sel.cpu()  # [n, 1+K]
    gumbel_patch_ids = gumbel_indices[:, 1:] - 1                     # [n, K]
    K = gumbel_patch_ids.shape[1]

    # --- Denormalise images for display -----------------------------
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_display = images.cpu() * std + mean
    imgs_display = imgs_display.clamp(0, 1)

    # --- Helper: build RGBA overlay from a set of patch indices -----
    def _build_overlay(patch_ids_1d, color_rgb, alpha=0.4):
        overlay = np.zeros((IMAGE_RESOLUTION_PX, IMAGE_RESOLUTION_PX, 4), dtype=np.float32)
        for pid in patch_ids_1d:
            pid = int(pid)
            if pid < 0 or pid >= PATCH_GRID_COUNT * PATCH_GRID_COUNT:
                continue
            row = pid // PATCH_GRID_COUNT
            col = pid % PATCH_GRID_COUNT
            y0, y1 = row * PATCH_SIZE_PX, (row + 1) * PATCH_SIZE_PX
            x0, x1 = col * PATCH_SIZE_PX, (col + 1) * PATCH_SIZE_PX
            overlay[y0:y1, x0:x1, :3] = color_rgb
            overlay[y0:y1, x0:x1, 3]  = alpha
        return overlay

    # --- Build the figure -------------------------------------------
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples),
                             constrained_layout=True)
    if n_samples == 1:
        axes = axes[np.newaxis, :]   # ensure 2-D indexing

    GREEN = np.array([0.0, 0.85, 0.25])

    for i in range(n_samples):
        img_np = imgs_display[i].permute(1, 2, 0).numpy()   # [H, W, 3]

        # --- Column 0: Original Image (Pulita) ---
        ax = axes[i, 0]
        ax.imshow(img_np) # Solo sfondo: immagine originale
        ax.set_title(f"Sample {i+1} — Original Image",
                     fontsize=10, fontweight='bold')
        ax.axis('off')

        # --- Column 1: Gumbel-selected patches (green) ---
        ax = axes[i, 1]
        ax.imshow(img_np) # Sfondo: immagine originale
        overlay_g = _build_overlay(gumbel_patch_ids[i], GREEN, alpha=0.4)
        ax.imshow(overlay_g)
        ax.set_title(f"Sample {i+1} — Gumbel selection (K={K})",
                     fontsize=10, fontweight='bold')
        ax.axis('off')

    # --- Legend ------------------------------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=GREEN, alpha=0.6, label='Gumbel-selected patches'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=1,
               fontsize=11, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Original Image vs Patch Selection',
                 fontsize=14, fontweight='bold', y=1.01)

    # --- Save --------------------------------------------------------
    save_path = os.path.join(hydra_output_dir, "visual_comp_random.png")
    fig.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

# ============================================================
# TRAINING SCHEDULE
# ============================================================

def training_schedule(model, train_data_loader, val_data_loader,
                      optimizer, num_epochs, device,
                      hydra_output_dir, cfg,
                      loss=torch.nn.CrossEntropyLoss(),
                      plot=True, save_model=True):

    train_losses, train_accuracies, val_losses, val_accuracies, communication_cost = [], [], [], [], []
    train_stats_history = {}
    val_stats_history   = {}
    snr_sweep_history   = {}

    best_val_accuracy = -1.0
    best_stats = {}
    final_results_file = os.path.join(hydra_output_dir, "final_training_results.json")
    best_results_file  = os.path.join(hydra_output_dir, "best_training_results.json")
    results = {}

    # --- Compressed Schedule: recalculate total steps for cosine tau annealing ---
    total_steps = len(train_data_loader) * num_epochs
    if hasattr(model, 'compressor_module') and hasattr(model.compressor_module, 'anneal_steps'):
        original_anneal_steps = model.compressor_module.anneal_steps
        model.compressor_module.anneal_steps = total_steps
        if plot:
            print(f"\n=== Gumbel Annealing: total_steps={total_steps} (was {original_anneal_steps})")
            print(f"    tau: {model.compressor_module.tau_max} → {model.compressor_module.tau_min} over {num_epochs} epochs")

    try:
        for epoch in range(1, num_epochs + 1):
            torch.cuda.empty_cache()
            if plot:
                print(f"\n\nEPOCH {epoch}")

            # ================================================================
            # CURRICULUM: notify the compressor of current epoch progress
            # ================================================================
            # This drives _compute_logit_scale() and _compute_entropy_target()
            # inside GumbelTokenSelectorBlockWrapper.
            if hasattr(model, 'compressor_module') and \
               hasattr(model.compressor_module, 'register_epoch'):
                model.compressor_module.register_epoch(epoch, num_epochs)
                if plot:
                    cm = model.compressor_module
                    alpha_now   = cm._compute_logit_scale()
                    h_target    = cm._compute_entropy_target()
                    print(f"  [Curriculum] epoch={epoch}/{num_epochs} | "
                          f"logit_alpha={alpha_now:.3f} | H_target={h_target:.3f}")

            # Training phase
            avg_train_loss, avg_train_accuracy, train_stats = train_one_epoch(
                model, train_data_loader, loss, optimizer, device, plot,
                current_epoch=epoch, num_epochs=num_epochs
            )

            # ================================================================
            # DIAGNOSTICS DUMP (extended with curriculum keys)
            # ================================================================
            if hasattr(model, 'compressor_module') and \
               hasattr(model.compressor_module, 'diagnostic_stats'):
                import numpy as np

                stats = model.compressor_module.diagnostic_stats
                epoch_diag_summary = {}
                for k, v_list in stats.items():
                    if len(v_list) > 0:
                        epoch_diag_summary[k] = float(np.mean(v_list))

                diag_file = os.path.join(hydra_output_dir, "diagnostic_gumbel.json")
                with open(diag_file, "a") as f:
                    epoch_diag_summary["epoch"] = epoch
                    f.write(json.dumps(epoch_diag_summary) + "\n")

                for k in stats.keys():
                    stats[k].clear()

            # --- Validation and SNR Sweep ---
            snr_sweep = []
            if 'eval' in cfg and 'snr_sweep' in cfg.eval:
                snr_sweep = list(cfg.eval.snr_sweep)
            elif 'communication' in cfg and 'eval' in cfg.communication and 'snr_sweep' in cfg.communication.eval:
                snr_sweep = list(cfg.communication.eval.snr_sweep)

            val_acc_dict  = {}
            val_loss_dict = {}
            val_stats_dict = {}

            if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                if plot:
                    print(f"\nPerforming Validation & SNR sweep for epoch {epoch}...")
                original_snr = cfg.communication.comm.channel.get('snr_db', 20.0)
                for snr in snr_sweep:
                    model.channel.reconfigure({'channel': {'snr_db': snr}})
                    v_loss, v_acc, v_stats = evaluate_one_epoch(model, val_data_loader, loss, device, plot=False)
                    val_loss_dict[str(snr)] = v_loss
                    val_acc_dict[str(snr)]  = v_acc
                    val_stats_dict[str(snr)] = v_stats
                    if plot:
                        print(f"  SNR: {snr} dB → Val loss: {v_loss:.4f}; Val acc: {v_acc:.4f}")
                model.channel.reconfigure({'channel': {'snr_db': original_snr}})

                avg_val_accuracy = sum(val_acc_dict.values()) / len(val_acc_dict)
                avg_val_loss     = sum(val_loss_dict.values()) / len(val_loss_dict)
                repr_v_stats = val_stats_dict[str(snr_sweep[0])]
                val_losses.append(val_loss_dict)
                val_accuracies.append(val_acc_dict)
            else:
                avg_val_loss, avg_val_accuracy, repr_v_stats = evaluate_one_epoch(
                    model, val_data_loader, loss, device, plot
                )
                val_losses.append(avg_val_loss)
                val_accuracies.append(avg_val_accuracy)

            # Best-epoch tracking
            criterion = cfg.dataset.get('selection_criterion', 'average')
            is_best = False
            current_comparison_metric = avg_val_accuracy

            if criterion == 'last':
                is_best = True
            elif criterion == 'max_noise' and snr_sweep:
                min_snr_val = min(snr_sweep)
                current_comparison_metric = val_acc_dict[str(min_snr_val)]
                is_best = (current_comparison_metric >= best_val_accuracy)
            elif criterion == 'min_noise' and snr_sweep:
                max_snr_val = max(snr_sweep)
                current_comparison_metric = val_acc_dict[str(max_snr_val)]
                is_best = (current_comparison_metric >= best_val_accuracy)
            elif criterion == 'snr_index' and snr_sweep:
                idx = cfg.dataset.get('selection_snr_index', 0)
                idx = min(max(0, idx), len(snr_sweep) - 1)
                snr_val = snr_sweep[idx]
                current_comparison_metric = val_acc_dict[str(snr_val)]
                is_best = (current_comparison_metric >= best_val_accuracy)
            else:
                current_comparison_metric = avg_val_accuracy
                is_best = (current_comparison_metric >= best_val_accuracy)

            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_accuracy)
            communication_cost.append(model.communication)

            if is_best:
                best_val_accuracy = current_comparison_metric
                top_imp_frac = repr_v_stats.get("mode_alloc_top_imp_frac")
                if top_imp_frac is None:
                    top_imp_frac = repr_v_stats.get("stream_alloc_top_imp_frac")
                if top_imp_frac is None:
                    top_imp_frac = train_stats.get("mode_alloc_top_imp_frac",
                                                   train_stats.get("stream_alloc_top_imp_frac", 0.0))
                
                best_stats = {
                    "best_epoch": epoch,
                    "best_train_accuracy": avg_train_accuracy,
                    "best_train_loss": avg_train_loss,
                    "best_communication_cost": model.communication,
                    "top_imp_frac": top_imp_frac,
                    "tokens_sent":  repr_v_stats.get("tokens_sent", train_stats.get("tokens_sent", 0)),
                    "symbols_sent": repr_v_stats.get("mimo_L", train_stats.get("mimo_L", 0)),
                    "symbol_rate":  repr_v_stats.get("rate_symbols", train_stats.get("rate_symbols", 0.0))
                }
                
                if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                    best_stats["best_val_accuracy"] = val_acc_dict
                    best_stats["best_val_loss"]     = val_loss_dict
                    for snr_str, s_stats in val_stats_dict.items():
                        for k, v in s_stats.items():
                            best_stats[f"best_val_{k}_snr_{snr_str}"] = v
                else:
                    best_stats["best_val_accuracy"] = avg_val_accuracy
                    best_stats["best_val_loss"]     = avg_val_loss
                    for k, v in repr_v_stats.items():
                        best_stats[f"best_val_{k}"] = v
                
                for k, v in train_stats.items():
                    best_stats[f"best_train_{k}"] = v

                with open(best_results_file, "w") as f:
                    json.dump(best_stats, f, indent=4)

                if hasattr(model, 'compressor_module') and \
                   hasattr(model.compressor_module, 'last_indices_sel'):
                    try:
                        split_idx_val = cfg.hyperparameters.split_index
                        save_visualization(
                            model, val_data_loader, device,
                            hydra_output_dir, split_idx_val,
                            n_samples=4,
                        )
                    except Exception as e:
                        print(f"  Visualisation failed: {e}")
                    finally:
                        model.train() 

            if plot:
                print(f"\nTrain loss: {avg_train_loss:.4f}; Val loss: {avg_val_loss:.4f}")
                print(f"Train accuracy: {avg_train_accuracy:.2f}; Val accuracy: {avg_val_accuracy:.2f}")

            for k, v in train_stats.items():
                key = f"Train {k}"
                if key not in train_stats_history:
                    train_stats_history[key] = []
                train_stats_history[key].append(v)

            if snr_sweep and hasattr(model, "channel") and hasattr(model.channel, "reconfigure"):
                for snr_str, s_stats in val_stats_dict.items():
                    for k, v in s_stats.items():
                        key = f"Val {k} (SNR {snr_str})"
                        if key not in val_stats_history:
                            val_stats_history[key] = []
                        val_stats_history[key].append(v)
            else:
                for k, v in repr_v_stats.items():
                    key = f"Val {k}"
                    if key not in val_stats_history:
                        val_stats_history[key] = []
                    val_stats_history[key].append(v)

            results = {
                "Train losses":       train_losses,
                "Train accuracies":   train_accuracies,
                "Val losses":         val_losses,
                "Val accuracies":     val_accuracies,
                "Communication cost": communication_cost,
                "Compression":        model.compression_ratio,
            }
            results.update(train_stats_history)
            results.update(val_stats_history)
            results.update(best_stats)
            results.update(snr_sweep_history)

            with open(final_results_file, "w") as f:
                json.dump(results, f, indent=4)

    finally:
        if results:
            with open(final_results_file, "w") as f:
                json.dump(results, f, indent=4)


def _build_param_groups(model, cfg):
    """
    Construct parameter groups for optimization with specific learning rates and weight decays.
    """
    base_lr = cfg.optimizer.lr
    base_wd = cfg.optimizer.weight_decay

    encoder_2d    = []
    decoder_2d    = []
    score_head_2d = []   # simplicial Gumbel params (2-D tensors → apply WD)

    encoder_1d    = []
    decoder_1d    = []
    score_head_1d = []   # simplicial biases / norms (1-D → no WD)

    split_idx = cfg.hyperparameters.split_index

    SCORE_HEAD_NAMES = [
        'compressor_module.w_u',
        'compressor_module.w_tri',
        'compressor_module.gamma',
        'compressor_module.beta',
        'compressor_module.gate',
        'compressor_module.branch_norm',
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # --- Score-head detection (Gumbel simplicial branch) ---
        is_score_head = any(tag in name for tag in SCORE_HEAD_NAMES)

        # --- Encoder detection ---
        is_encoder = False
        if not is_score_head and 'head' not in name:
            if any(x in name for x in ['patch_embed', 'cls_token', 'pos_embed']):
                is_encoder = True
            elif 'blocks.' in name:
                match = re.search(r'blocks\.(\d+)', name)
                if match and int(match.group(1)) < split_idx:
                    is_encoder = True

        is_decoder_or_head = not is_score_head and not is_encoder

        # --- 1-D / norm detection (no weight decay) ---
        is_1d_or_norm = param.ndim <= 1 or 'norm' in name or 'bias' in name

        if is_score_head:
            if is_1d_or_norm:
                score_head_1d.append(param)
            else:
                score_head_2d.append(param)
        elif is_encoder:
            if is_1d_or_norm:
                encoder_1d.append(param)
            else:
                encoder_2d.append(param)
        else:  # decoder / head
            if is_1d_or_norm:
                decoder_1d.append(param)
            else:
                decoder_2d.append(param)

    param_groups = [
        # ENCODER
        {'params': encoder_2d,    'lr': base_lr * 0.1, 'weight_decay': base_wd},
        {'params': encoder_1d,    'lr': base_lr * 0.1, 'weight_decay': 0.0},
        # DECODER & HEAD
        {'params': decoder_2d,    'lr': base_lr,       'weight_decay': base_wd},
        {'params': decoder_1d,    'lr': base_lr,       'weight_decay': 0.0},
        # SCORE HEAD (Gumbel simplicial) — moderate WD to cap logit saturation
        {'params': score_head_2d, 'lr': base_lr * 5.0, 'weight_decay': 1e-3},
        {'params': score_head_1d, 'lr': base_lr * 5.0, 'weight_decay': 0.0},
    ]
    return param_groups


def _build_output_dir(cfg, seed: int, model: nn.Module) -> str:
    """
    Construct the hierarchical output directory path according to the configuration and seed.
    """
    results_path = cfg.core.get('results_path', './results')
    dataset_name = cfg.dataset.name
    split_index = cfg.hyperparameters.split_index
    
    # Extract channel type from communication.name
    comm_name = cfg.communication.name.lower()
    if 'mimo' in comm_name:
        channel_type = 'MIMO'
    elif comm_name == 'clean':
        channel_type = 'None'
    else:
        channel_type = 'AWGN'
    
    # Extract method type from method.name
    method_name = cfg.method.name.lower()
    if 'gumbel' in method_name:
        method_type = 'Gumbel'
    elif 'random' in method_name:
        method_type = 'Random'
    else:
        method_type = cfg.method.name
    
    # ================================================================
    # COMMUNICATION VARIANT CLASSIFICATION
    # Classifica la variante di comunicazione basata su config
    # ================================================================
    comm_name = cfg.communication.name.lower()
    
    # Check for mode and power allocation flags
    has_mode_alloc = False
    has_power_alloc = False
    
    if hasattr(cfg.communication.comm, 'channel') and hasattr(cfg.communication.comm.channel, 'mode_alloc'):
        mode_alloc = cfg.communication.comm.channel.mode_alloc
        has_mode_alloc = mode_alloc.enabled if hasattr(mode_alloc, 'enabled') else False
    
    if hasattr(cfg.communication.comm, 'channel') and hasattr(cfg.communication.comm.channel, 'power_alloc'):
        power_alloc = cfg.communication.comm.channel.power_alloc
        has_power_alloc = power_alloc.enabled if hasattr(power_alloc, 'enabled') else False
    
    # Determine communication variant
    if 'dct' in comm_name or 'svd' in comm_name:
        if not has_mode_alloc and not has_power_alloc:
            comm_variant = 'Base'
        elif 'dct' in comm_name:
            comm_variant = 'DCT'
        else:
            comm_variant = 'ISW'
    elif 'dct' in comm_name or 'isw' in comm_name:
        # Has dct/isw but no mode/power allocation → Base
        if not has_mode_alloc and not has_power_alloc:
            comm_variant = 'Base'
        else:
            comm_variant = 'ISW' if 'isw' in comm_name else 'DCT'
    else:
        # Generic fallback: use communication name or "Other"
        comm_variant = comm_name.replace('baseline_mimo_', '').upper() if 'baseline' in comm_name else 'Other'
    
    # ================================================================
    # AUTOMATIC TOKEN TRANSMISSION CALCULATION
    # Tokens transmitted = input_dim (total tokens) × desired_compression (ratio)
    # Example: 192 × 0.1 = 19.2 ≈ 19 tokens transmitted
    # ================================================================
    
    # Get total tokens from config (communication.channel.input_dim)
    total_tokens = cfg.communication.channel.input_dim if hasattr(cfg.communication.channel, 'input_dim') else 192
    
    # Get compression ratio from config (method.parameters.desired_compression)
    compression_ratio = None
    if hasattr(cfg, 'method') and isinstance(cfg.method, dict) is False:
        # OmegaConf dict-like - use get
        compression_ratio = cfg.method.parameters.get('desired_compression') if cfg.method.parameters is not None else None
        if compression_ratio is None:
            compression_ratio = cfg.method.parameters.get('token_compression') if cfg.method.parameters is not None else None
    else:
        # Generic fallback for plain dict-like
        compression_ratio = cfg.get('desired_compression', None)

    if compression_ratio is None:
        compression_ratio = getattr(model, 'compression_ratio', None)

    # Calculate transmitted tokens: int(total_tokens × compression_ratio)
    try:
        if isinstance(compression_ratio, (int, float)) and compression_ratio > 0:
            # Assume compression_ratio is already a decimal (e.g., 0.1 = 10%)
            transmitted_tokens = int(round(total_tokens * float(compression_ratio)))
        else:
            # Fallback: treat as already being the transmitted token count
            transmitted_tokens = int(float(compression_ratio)) if compression_ratio is not None else total_tokens
    except Exception:
        transmitted_tokens = total_tokens

    compression_val = transmitted_tokens
    
    # Optional custom title from experiment_name (only if not 'test')
    experiment_name = cfg.hyperparameters.get('experiment_name', 'test')
    custom_title = experiment_name if experiment_name != 'test' else ''
    
    # Build hierarchical path matching results directory structure
    # Special handling for baseline: save at seed level for easy plotting

    # Special-case: if the channel has been replaced with an IdentityWrapper
    # (i.e., no channel active even at eval), treat the run as a Baseline
    # and save at seed level under 'Baseline'. This covers overrides like
    # `communication.channel._target_=comm.communication.IdentityWrapper`.
    ch_target = None
    try:
        if hasattr(cfg.communication, 'channel') and hasattr(cfg.communication.channel, '_target_'):
            ch_target = str(cfg.communication.channel._target_)
    except Exception:
        ch_target = None

    if ch_target and 'IdentityWrapper' in ch_target:
        hydra_output_dir = os.path.join(
            results_path,
            dataset_name,
            f'seed_{seed}',
            'Baseline'
        )
    else:
        if method_type.lower() == "baseline":
            hydra_output_dir = os.path.join(
                results_path,
                dataset_name,
                f'seed_{seed}',
                'Baseline'
            )
        elif method_type.lower() == "random":
            # Per il metodo Random saltiamo le varianti (ISW, Base, ecc.)
            # e usiamo la C maiuscola per "Compression" come da te richiesto
            hydra_output_dir = os.path.join(
                results_path,
                dataset_name,
                f'seed_{seed}',
                f'split_{split_index}',
                channel_type,
                method_type,
                f'Compression_{compression_val}'
            )
        else:
            if comm_variant == 'Base' and channel_type == 'MIMO':
                comp_str = str(int(compression_val)) if compression_val == int(compression_val) else str(compression_val)
                
                hydra_output_dir = os.path.join(
                    results_path,
                    dataset_name,
                    f'seed_{seed}',
                    f'split_{split_index}',
                    channel_type,       
                    'base',             
                    f'compression_1'
                )
            else:
                # Structure: {method_type}/compression_{value}/{comm_variant}
                # This matches the actual directory layout where variants are under compression
                hydra_output_dir = os.path.join(
                    results_path,
                    dataset_name,
                    f'seed_{seed}',
                    f'split_{split_index}',
                    channel_type,
                    method_type,
                    f'compression_{compression_val}',
                    comm_variant
                )
    
    if custom_title and method_type.lower() != "baseline":
        hydra_output_dir = os.path.join(hydra_output_dir, custom_title)
    
    # Handle duplicate runs by appending counter
    base_dir = hydra_output_dir
    counter = 1
    while os.path.exists(os.path.join(hydra_output_dir, "final_training_results.json")) or \
          os.path.exists(os.path.join(hydra_output_dir, "training_results.json")):
        hydra_output_dir = f"{base_dir}_{counter}"
        counter += 1

    os.makedirs(hydra_output_dir, exist_ok=True)
    return hydra_output_dir


@hydra.main(config_path="configs", version_base='1.2', config_name="default")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size  = cfg.dataset.batch_size
    num_epochs  = int(cfg.dataset.get('num_epochs', cfg.hyperparameters.get('num_epochs', 10)))
    if num_epochs <= 0:
        raise ValueError("dataset.num_epochs must be a positive integer")
    num_workers = cfg.dataset.get('num_workers', 8)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset   = hydra.utils.instantiate(cfg.dataset.test)

    seeds = cfg.hyperparameters.get('seeds', [42, 51, 114])
    if OmegaConf.is_list(seeds):
        seeds = list(seeds)
    elif not isinstance(seeds, list):
        seeds = [seeds]

    for seed in seeds:
        torch.manual_seed(seed)

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, shuffle=True, drop_last=True,
            batch_size=batch_size, num_workers=num_workers,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_dataset, shuffle=False,
            batch_size=batch_size, num_workers=num_workers,
        )

        model   = hydra.utils.instantiate(cfg.model)
        channel = hydra.utils.instantiate(cfg.communication.channel)
        model   = hydra.utils.instantiate(
            cfg.method.model,
            channel=channel,
            split_index=cfg.hyperparameters.split_index,
            model=model,
        ).to(device)

        # ======================================================================
        # PARAMETER GROUPS — Weight Decay Differenziato
        # ======================================================================
        # Groups:
        #   ENCODER        → lr*0.1,  wd=base_wd  (protect ImageNet weights)
        #   DECODER/HEAD   → lr*1.0,  wd=base_wd
        #   SCORE_HEAD_2D  → lr*5.0,  wd=1e-3     ← NEW: moderate WD keeps
        #                                              logit std in [1.5, 2.5]
        #   SCORE_HEAD_1D  → lr*5.0,  wd=0.0      (bias/scale: no WD)
        #
        # The score_head group covers the simplicial parameters of the Gumbel
        # compressor (w_u, w_tri, beta, gamma, gate_param, branch_norm).
        # A non-zero weight_decay prevents the logit norms from drifting to
        # saturation, naturally capping std at a healthy target of ~2.0.
        # ======================================================================
        param_groups = _build_param_groups(model, cfg)
        optimizer = torch.optim.AdamW(param_groups, eps=cfg.optimizer.eps)
        score_head_2d_params = param_groups[4]['params']

        print(
            f"\n\nTraining seed {seed}:"
            f"\n  model={cfg.model.model_name}"
            f"\n  dataset={cfg.dataset.name}"
            f"\n  method={cfg.method.name}"
            f"\n  compression={model.compression_ratio}"
            f"\n  score_head_2d params: {sum(p.numel() for p in score_head_2d_params)}"
            f"\n  logit_scale_start={cfg.method.parameters.get('logit_scale_start', 0.1)}"
            f"  → logit_scale_end={cfg.method.parameters.get('logit_scale_end', 1.0)}"
            f"\n  entropy_target: {cfg.method.parameters.get('entropy_target_start', 5.2)}"
            f"  → {cfg.method.parameters.get('entropy_target_end', 2.0)}\n"
        )

        # ================================================================
        # NEW: Hierarchical directory structure
        # Structure: results/{dataset}/seed_{seed}/split_{split}/{channel_type}/{method_type}/compression_{compression}/{optional_title}
        # ================================================================
        hydra_output_dir = _build_output_dir(cfg, seed, model)

        training_schedule(
            model, train_dataloader, val_dataloader,
            optimizer, num_epochs, device,
            hydra_output_dir, cfg,
            save_model=(seed == seeds[0]),
        )


if __name__ == "__main__":
    main()
