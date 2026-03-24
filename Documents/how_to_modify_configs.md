# Configuration Guide for the Project (Split Learning & MIMO)

This guide explains how to configure and run Split Learning experiments on MIMO channels. The system uses **Hydra** to manage configurations in a modular way.

---

## 1. Main Configuration (`configs/default.yaml`)
This is the entry point. Here you choose the modules to combine for the run.

| Parameter | Description |
| :--- | :--- |
| **`communication`** | Chooses the radio profile (e.g. `baseline_mimo_svd`, `clean`, `noisy`). |
| **`dataset`** | Chooses the dataset (e.g. `cifar_100`, `imagenette`). |
| **`model`** | Chooses the model architecture (e.g. `deit_tiny_patch16_224`). |
| **`method`** | Configures the Split Learning logic (e.g. `proposal`). |
| **`hyperparameters.split_index`** | Defines at which layer to split the network between Client and Server. |
| **`hyperparameters.experiment_name`** | Name of the folder where results will be saved. |
| **`hyperparameters.seeds`** | List of seeds for reproducibility. |

---

## 2. Communication Module (`configs/communication/`)
Regulates the end-to-end transmission from the Client to the Server, handling compression, physical noise, and spatial resource allocation.

### General Configuration and Evaluation
*   **`eval/snr_sweep`**: Indicates the SNR values (dB) tested during the evaluation phase. An analysis is performed for each parameter in the list. If omitted, the maximum value defined in `snr_db` is used.
*   **`channel/input_dim`**: Defines the dimension of the last client-side layer before passing to the communication channel (e.g. 192 for tiny architectures).
*   **`bottleneck/out_dim`**: Indicates the format in which images are sent to the channel. A compression layer (Joint Source-Channel Coding) is applied to reduce the size of the original data.

### MIMO Channel Configuration
*   **`comm/type`**: Identifies the type of physical channel (e.g. `"mimo"`).
*   **`comm/n_tx`**: Number of transmitting antennas sending the signal from the Client side.
*   **`comm/n_rx`**: Number of receiving antennas picking up the signal from the Server side.
*   **`comm/fading`**: Method for generating the channel matrix $H$:
    *   `identity`: The matrix $H$ is the identity (perfect channel).
    *   `random`: Random matrix based on the specified distribution parameters.
    *   `diagonal`: Diagonal matrix with independent gains.
    *   `rayleigh`: Realistic channel with Rayleigh fading (i.i.d. Gaussian model).
*   **`comm/equalizer`**: Algorithm used by the receiver to restore the original signal to solve the optimization problem:
    *   `mmse`: *Minimum Mean Square Error* (optimal balance between noise and interference).
    *   `zf`: *Zero Forcing* (total cancellation of interference at the expense of noise).
    *   `none`: No active receiver (pass-through transmission).

### Noise Management and Normalization
*   **`comm/snr_db`**: Noise value or range encountered by the signal during training.
*   **`comm/train_mode`**: 
    *   `sampled`: SNR is sampled randomly within the defined range.
    *   `fixed`: A single constant SNR value is used throughout the training.
*   **`comm/sample_mode`**: SNR sampling criterion: `per sample` (each image has its own noise) or `per batch`.
*   **`comm/normalization_mode`**: Power normalization criterion: `sample` or `batch`.

### Power Allocation (Power Allocation)
Implements the *Waterfilling* criterion to optimize data resilience.
*   **`power_alloc/source`**: Source of scores for allocation:
    *   `selection_scores`: Waterfilling based on semantic importance (attention scores).
    *   `channel_capacity`: Waterfilling based on instantaneous channel capacity.
    *   `uniform`: Uniform power allocation.
*   **`power_alloc/alpha`**: Proportion between total energy and the chosen criterion (higher values give more weight to important data).
*   **`power_alloc/eps`**: Minimum amount of energy guaranteed to each antenna/token to avoid switching it off.

### Structured Cases (Advanced Fading)
*   **`diagonal/gains`**: Defines the specific values observed along the diagonal of the matrix $H$.
*   **`random/distribution`**: Distribution of values on the diagonal: `uniform` or `gaussian`.
*   **`random/min_gain` / `max_gain`**: Limits for sampling the gains.
*   **`random/seed`**: Seed to ensure reproducibility of the channel sampling.

### Stream Allocation (Stream Allocation)
Defines how to apply *Semantic Waterfilling* within the MIMO channel, unlike `power_alloc` which operates ignoring the physical channel.
*   **`stream_alloc/strategy`**: Strategy for mapping tokens to antennas:
    *   `importance_to_gain`: Maps important tokens to streams with higher gain.
    *   `importance_to_capacity`: Maps important tokens to higher capacity.
    *   `importance_to_uniform`: Uniform and sequential distribution.
*   **`assignment/granularity`**: Symbol sending criterion:
    *   `token`: Each token is treated individually.
    *   `chunk`: Tokens are grouped into blocks and sent on the same antenna.
*   **`assignment/chunk_size`**: Grouping size if granularity is set to `chunk`.

---

## 3. Difference between power_alloc and stream_alloc_power
It is essential to distinguish the two levels of energy optimization:

| Technique | Where it is applied | Objective |
| :--- | :--- | :--- |
| **`power_alloc`** | On the original data (**Transformer Tokens**). | Semantic optimization before data is packed for antennas. |
| **`stream_alloc_power`** | On data already **Packed** for the antennas. | Optimization based on the physical characteristics of the channel and antenna-token mapping. |

---

## 4. Compression Method (`configs/method/proposal.yaml`)
Manages how to select and drop tokens or samples to stay within the communication budget.

| Parameter | Description |
| :--- | :--- |
| **`desired_compression`** | Global goal for data reduction (e.g. `0.1` = keep 10% of original data). |
| **`token_compression`** | Percentage of tokens (patches) to keep for each image. |
| **`batch_compression`** | Percentage of images to keep within a batch through clustering. |
| **`pooling`**| Importance calculation method: `attention` (self-attention scores), `average`, `cls`. |


---

## 5. Dataset and Pre-processing (`configs/dataset/`)
Configures data loading.

*   **`name`**: Dataset identifier.
*   **`num_classes`**: Number of classes for final classification.
*   **`batch_size`**: Images loaded per iteration.
*   **`max_communication`**: Transmittable sample limit (data budget).
*   **`selection_criterion`**: Criterion for choosing the "best epoch" saved in the results:
    *   `average`: Average accuracy across all SNRs of the eval sweep (default).
    *   `last`: Always saves the last completed epoch.
    *   `max_noise`: Best accuracy detected at the lowest SNR (maximum noise).
    *   `min_noise`: Best accuracy detected at the highest SNR (minimum noise).
    *   `snr_index`: Uses the accuracy detected at a specific index of the sweep (indicated by `selection_snr_index`).
*   **`selection_snr_index`**: Index (0, 1, 2...) of the SNR sweep to use if the criterion is `snr_index`.
*   **`train, test / transform`**: Image augmentation (Resize, RandAugment, Normalize).
---
