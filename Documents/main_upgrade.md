# Detailed Changelog — `main.py`

Document specifically and comprehensively describing all the modifications made in the updated file compared to the original version.

---

## 1. `training_phase` Function

### 1.1 Initialization of `epoch_stats`
**Original:** the function did not collect channel/MIMO statistics.
**Updated:** an `epoch_stats = {}` dictionary is initialized immediately following the `train_loss` and `train_accuracy` variables, to accumulate batch-by-batch additional metrics.

### 1.2 Channel statistics collection (new block)
**Original:** missing.
**Updated:** after computing `batch_predictions`, a block is added collecting communication/MIMO channel statistics, if available:

```python
if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
    batch_stats = model.channel.get_last_info()
    for k, v in batch_stats.items():
        if isinstance(v, (int, float)):
            epoch_stats[k] = epoch_stats.get(k, 0.0) + v
```

Stats are accumulated key-by-key within `epoch_stats`.

### 1.3 Accuracy computation — `compressor_module` branch
**Original:**
```python
batch_labels = model.compressor_module.compress_labels(batch_labels, len(train_data_loader.dataset.classes))
batch_accuracy = 0
```
Number of classes was fetched via dataset variables and accuracy permanently locked to zero.

**Updated:**
```python
num_classes = batch_predictions.shape[-1]
batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)
pred_classes = torch.argmax(batch_predictions, dim=1)
target_classes = torch.argmax(batch_labels, dim=1)
batch_accuracy = torch.mean((pred_classes == target_classes).float()).item()
```
- The number of classes derives accurately off model output dimensions (`batch_predictions.shape[-1]`), detaching logic from manual dataset polling.
- Accuracy is no longer fixed entirely on zero: it resolves accurately comparing `argmax` values against compressed targets (handling soft labeling structures properly).

### 1.4 Accuracy computation — `else` branch
**Original:**
```python
batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]
```

**Updated:**
```python
pred_classes = torch.argmax(batch_predictions, dim=1)
batch_accuracy = torch.mean((batch_labels == pred_classes).float()).item()
```
Equivalent mechanism but noticeably cleaner and robust: engages `torch.mean` across boolean arrays processed to `float`, exchanging previous structure relying strictly towards exact dataset dimensions divisors.

### 1.5 Returning Average Statistics
**Original:** function generated exclusively `(average_train_loss, average_train_accuracy)`.

**Updated:** appends channel calculation parameters yielding an broadened tuple structure:
```python
average_epoch_stats = {k: v / iterations for k, v in epoch_stats.items()}
return average_train_loss, average_train_accuracy, average_epoch_stats
```

---

## 2. `validation_phase` Function

### 2.1 Initialization of `epoch_stats`
**Original:** absent.
**Updated:** appended `epoch_stats = {}` directly underneath `val_loss` and `val_accuracy`, mimicking exactly variables established in `training_phase`.

### 2.2 Channel statistics collection (new block)
**Original:** absent.
**Updated:** same collection block implemented in `training_phase`, utilized inside validation segment:

```python
if hasattr(model, "channel") and hasattr(model.channel, "get_last_info"):
    batch_stats = model.channel.get_last_info()
    for k, v in batch_stats.items():
        if isinstance(v, (int, float)):
            epoch_stats[k] = epoch_stats.get(k, 0.0) + v
```

### 2.3 `compressor_module` management in validation
**Original:** the validation script remained unprepared encountering instances housing a `compressor_module`; incoming classes were digested completely raw.

**Updated:** corresponding condition mimicking formatting established previously:
```python
if hasattr(model, "compressor_module"):
    num_classes = batch_predictions.shape[-1]
    batch_labels = model.compressor_module.compress_labels(batch_labels, num_classes)
```

### 2.4 Accuracy computing in validation
**Original:**
```python
batch_accuracy = torch.sum(batch_labels == torch.argmax(batch_predictions, dim=1)).item() / batch_labels.shape[0]
```
Dismissed possibilities acknowledging multi-dimensional classes (soft labels).

**Updated:** embeds specific processing routines handling multi-dimensional classes properly (derived from `compress_labels`):
```python
pred_classes = torch.argmax(batch_predictions, dim=1)
if batch_labels.ndim > 1:
    target_classes = torch.argmax(batch_labels, dim=1)
else:
    target_classes = batch_labels
batch_accuracy = torch.mean((target_classes == pred_classes).float()).item()
```

### 2.5 Returning average stats
**Original:** returns exclusively `(average_val_loss, average_val_accuracy)`.

**Updated:**
```python
average_epoch_stats = {k: v / len(val_data_loader) for k, v in epoch_stats.items()}
return average_val_loss, average_val_accuracy, average_epoch_stats
```

---

## 3. `training_schedule` Function

### 3.1 New parameter `cfg`
**Original:** signature appeared as `training_schedule(model, ..., hydra_output_dir, loss=..., plot=True, save_model=True)`.
**Updated:** includes `cfg` (the Hydra config) parameter, fundamentally required accessing SNR sweep rules extending further downstream configurations.

### 3.2 Advanced stat timeline formatting
**Original:** stored raw arrays corresponding strictly to losses and accuracy tracking variables.
**Updated:** introduced three comprehensive map variables:
```python
train_stats_history = {}
val_stats_history  = {}
snr_sweep_history  = {}
```
Allows accumulating variables per active sweep accurately for each iteration across both metrics environments.

### 3.3 Default trigger of `best_val_accuracy`
**Original:** `best_val_accuracy = 0`
**Updated:** `best_val_accuracy = -1.0`
Improves best validation logic by allowing models mapping effectively at exact zeroes properly instead of failing initial comparisons.

### 3.4 Append `best_stats` and `best_results_file`
**Original:** did not output specific files indexing metrics associated exclusively to peak models.
**Updated:**
```python
best_stats = {}
best_results_file = os.path.join(hydra_output_dir, "best_training_results.json")
```
Generates distinct output index `best_training_results.json` locking stats attached completely against model iteration demonstrating pinnacle accuracy traits.

### 3.5 Elimination of `results_file` (partial logging)
**Original:** `training_results.json` dumped sequentially alongside each iteration as a partial log file.
**Updated:** eliminated. Maintains purely `final_training_results.json`, rendering iteration data permanently correctly instead of fragmenting it uselessly.

### 3.6 Loop `try/finally` handling
**Original:** central `for epoch` mechanism ran unshielded from breaks.
**Updated:** completely enclosed leveraging a `try/finally` barrier block:
```python
try:
    for epoch in range(1, 1000):
        ...
finally:
    if results:
        with open(final_results_file, "w") as f:
            json.dump(results, f, indent=4)
```
Secures generation output cleanly saving results uninterrupted regardless of sudden halts (`KeyboardInterrupt`, general exceptions).

### 3.7 SNR Sweep integration (new block)
**Original:** single instance evaluating strictly constrained fixed SNR figures running once per cycle.
**Updated:** launches extensive sweeps testing range limits seamlessly via parameters configured alongside `cfg`:
```python
snr_sweep = []
if 'eval' in cfg and 'snr_sweep' in cfg.eval:
    snr_sweep = list(cfg.eval.snr_sweep)
elif 'communication' in cfg and 'eval' in cfg.communication and 'snr_sweep' in cfg.communication.eval:
    snr_sweep = list(cfg.communication.eval.snr_sweep)
```
If the array operates actively, processes re-run matching each parameter array against target files while reverting configurations gracefully at conclusion.

### 3.8 Optimal Model evaluation upgrade
**Original:** logic processing saving models left largely deactivated.
**Updated:** system dynamically selects models leveraging variables connected inside `is_best`:
- Calculates broad spectrum accurate mapping values crossing entirely against available datasets whether sweeps remain active or constrained normally.

When peak iteration models trigger correctly, writes stats comprehensively out including elements alongside `epoch, accuratezza, loss, comunicazione, top_imp_frac, token inviati, simboli MIMO`, logging parameters carefully toward `best_training_results.json`.

### 3.9 Final mapping `results` adjustments
**Original:** compiled exclusively simple arrays.
**Updated:** encompasses expansive mapping elements accumulating arrays securely inside final payload sequences:
```python
results.update(train_stats_history)
results.update(val_stats_history)
results.update(best_stats)
results.update(snr_sweep_history)
```

### 3.10 Repositioning logic `max_communication`
**Original:** checking system evaluating `if model.communication > max_communication: break` resolved aggressively stopping results from processing perfectly into storage files.
**Updated:** relocated downwards preventing interruptions while correctly guaranteeing saving steps occur sequentially without delays.

---

## 4. `main` Function

### 4.1 Adjustable parameter `num_workers`
**Original:** defined absolutely towards `16`.
**Updated:** imported efficiently handling backup settings cleanly:
```python
num_workers = cfg.dataset.get('num_workers', 8)
```
Applies default variable sets starting at `8` handling loads significantly cleaner unless directed dynamically via Hydra files directly.

### 4.2 Handling array objects robustly inside `seeds`
**Original:**
```python
if not isinstance(seeds, list):
    seeds = [seeds]
```

**Updated:**
```python
if OmegaConf.is_list(seeds):
    seeds = list(seeds)
elif not isinstance(seeds, list):
    seeds = [seeds]
```
Checks list formats cleanly specifically aligning around OmegaConf array types bypassing random list processing failures efficiently.

### 4.3 DataLoader restructuring
**Original:** compiled continuously spanning wide single structures globally.
**Updated:** spaced out enhancing generalized reading experiences utilizing precise variable names dynamically.

### 4.4 Automated directory indexing duplicates
**Original:** skipped overlapping output directories bypassing runs executing entirely using exact strings entirely.

**Updated:** integrates logic hunting effectively alternative namespaces stepping numerically sequentially without corruptions:
```python
base_dir = hydra_output_dir
counter = 1
while os.path.exists(os.path.join(hydra_output_dir, "final_training_results.json")) or       os.path.exists(os.path.join(hydra_output_dir, "training_results.json")):
    hydra_output_dir = f"{base_dir}_{counter}"
    counter += 1
```
Removes skip sequences allowing concurrent configurations writing cleanly separating overlapping paths safely perfectly.

### 4.5 Executing `training_schedule` with `cfg`
**Original:** executed solely parameters extending partially `training_schedule(model, ..., hydra_output_dir, save_model=...)`.
**Updated:** sends targeted variables including broad spectrum maps accurately passing `cfg` seamlessly forwards:
```python
training_schedule(model, train_dataloader, val_dataloader, optimizer,
                  max_communication, device, hydra_output_dir, cfg,
                  save_model=seed == seeds[0])
```

---

## Modification Summary

| Sector | Upgrade |
|---|---|
| `training_phase` | Active statistics gathering; reliable dimensional classes resolving directly; outputs `epoch_stats` |
| `validation_phase` | Gathers accurate variables safely; integrated `compressor_module`; processes ND labels successfully; outputs `epoch_stats` |
| `training_schedule` | Appends `cfg`; advanced SNR tracking; tracks robust variables compiling best arrays cleanly; `try/finally`; integrated files securely |
| `main` | Configurable `num_workers`; OmegaConf seeds processing formats reliably; integrates numerical folders dodging collisions completely; integrates maps properly |
