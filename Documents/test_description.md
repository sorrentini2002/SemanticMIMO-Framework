# Current Experimental Test Description

This document provides a detailed description of the experimental test configuration and objectives currently underway, based on the configurations in the `configs` folder. The experiment focuses on image classification using a **Split-Computing** architecture applied to a vision Transformer.

## 1. Model Architecture and Task
The test uses a **DeiT** (Data-efficient Image Transformers) model, specifically the **Tiny** version with 16-pixel patches and a 224x224 input resolution.

- **Base Model**: `deit_tiny_patch16_224.fb_in1k` (Pre-trained on ImageNet-1k).
- **Dataset**: **CIFAR-100**, which includes 100 distinct classes of high-resolution images.
- **Task**: Multi-class classification.

## 2. Split-Computing Configuration
The model architecture is not executed entirely on a single node but is divided (**Split**) to allow for distributed execution.

- **Split Index**: 3.
- **Logic**: The network is split at index 3. The initial parts of the model process the original image and generate intermediate representations (feature maps or latent tokens). These representations are then processed by the compression method before being sent to the remaining part of the network to complete the classification.

## 3. Compression Method (Gumbel)
The test evaluates the **Gumbel** method, which implements a learnable token selection mechanism using Gumbel-Softmax to perform data compression at the split point.

### Core Gumbel-Softmax Parameters
- **`desired_compression` (0.1)**: The target selection ratio. For a ViT-Tiny with 197 total tokens (196 patches + 1 CLS token), the model is constrained to select approximately 20 tokens for transmission to the server.
- **`compression_enabled` (true)**: Toggles the token selection module. When active, only a subset of tokens is transmitted based on the selection logic.
- **`hard` (true)**: During the forward pass, this forces the Gumbel-Softmax distribution to be sampled as a discrete k-hot vector. This ensures that the model operates in a true "selection" mode (token is either sent or not) rather than a weighted mixing of tokens.
- **`straight_through` (true)**: Enables the Straight-Through Estimator (STE). Since discrete sampling is non-differentiable, STE uses the discrete selection in the forward pass but uses the gradient of the continuous "soft" scores in the backward pass, allowing the selection policy to be trained via backpropagation.
- **Temperature Annealing**:
  - **`tau_start` (2.0) / `tau_end` (0.1)**: Controls the "stiffness" of the selection distribution. A high temperature (2.0) makes the selection almost uniform and stochastic, promoting exploration. A low temperature (0.1) makes the distribution peaky and deterministic.
  - **`schedule` (linear) / `steps` (10000)**: The temperature $\tau$ decays linearly from start to end over 10,000 optimization steps. This "cools down" the system, transitioning from an exploratory phase to a stable selection policy as training progresses.

### Regularization and Diversity
- **`entropy_reg_weight` (0.1)**: Penalizes the model if the selection probability distribution becomes too deterministic too early (low entropy). This forces the model to explore different token combinations and prevents "collapse" where it might always pick the same spatial locations regardless of image content.
- **`cov_reg_weight` (0.5) / `margin` (0.3)**: Covariance Regularization. It penalizes pairs of selected tokens that are highly correlated. By enforcing a margin of 0.3, it effectively forces the model to select a *diverse* and *non-redundant* set of patches, maximizing the information content transmitted within the 10% budget.
- **`cov_reg_max_tokens` (64)**: Limits the covariance penalty calculation to a subset of tokens to reduce the $O(N^2)$ computational complexity during training.

### Inference and Semantic Prioritization
- **`gumbel_mc_enabled` (true)** / **`gumbel_mc_samples` (16)**: During evaluation, the model performs 16 Monte-Carlo stochastic draws. The results are aggregated (strategy: `mean`) to produce a more stable and robust token selection, reducing the impact of random noise in the selection scores during inference.
- **`semantic_waterfilling` (true)**: Integrates the ViT's internal attention scores into the selection process. It ensures that the most "semantically important" tokens (those with high class-token attention) are prioritized for transmission, especially over the communication channel's limited modes.
- **`channel_eval_only` (false)**: When false, the compression logic is active and trained end-to-end. If true, compression would only be applied during the validation phase on the communication channel.

## 4. Data Pipeline and Preprocessing
The CIFAR-100 dataset is processed using the following transformations:

### Training
- **Data Augmentation**: `RandAugment` (num_ops: 2, magnitude: 9) and `ColorJitter` (brightness, contrast, and saturation adjusted to 0.4).
- **Resizing**: Images are resized to 224x224 pixels.
- **Flip**: Random horizontal flip.
- **Normalization**: Implemented with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.

### Verification (Test/Validation)
- **Resizing**: 224x224 pixels.
- **Normalization**: Identical to training for data consistency.
- **Batch Size**: 128 images per batch.

## 5. Optimization Parameters
The system is trained/validated with the following optimization configuration:

- **Algorithm**: **Adam**.
- **Learning Rate**: $1 \times 10^{-4}$ (0.0001).
- **Epsilon**: $1 \times 10^{-08}$.
- **Weight Decay**: 0 (no weight decay regularization applied directly in the optimizer).
- **Seed**: 42 (to ensure reproducibility).

## 6. Evaluation Criterion
Model performance is monitored using the accuracy metric. The primary criterion for selecting the "best" model is based on the **average** of the performance recorded (parameter `selection_criterion: average`).

## 7. Communication Channel Configurations

The logic for communication and channel simulation is managed through two main configuration profiles, which define the physical layer parameters and the dimensionality reduction techniques applied at the split point.

### 7.1. MIMO Rayleigh Channel (`baseline_mimo_svd.yaml`)
This profile simulates a realistic multi-antenna communication environment.
- **System Dimensions (`n_tx: 4`, `n_rx: 4`)**: A 4x4 MIMO setup allowing up to 4 parallel spatial streams (eigen-modes) for data transmission.
- **Fading (`rayleigh`)**: Simulates a stochastic, rich-scattering environment. The channel gain matrix is complex-valued and follows a Rayleigh distribution, representing non-line-of-sight propagation.
- **Equalizer (`mmse`)**: Minimum Mean Square Error equalizer. It attempts to recover the transmitted signal by minimizing both noise and inter-stream interference.
- **Mode Allocation (`importance_to_modes`)**:
  - Uses SVD ($H = U\Sigma V^H$) to decompose the channel into independent flat-fading modes.
  - Features are mapped directly to these modes based on their importance.
  - **`per_sample: true`**: Recalculates the SVD for every individual sample in the batch, adapting to fast-fading channel fluctuations.
- **Assignment Logic**:
  - **`granularity: token`**: Features are assigned to channel modes at the individual token level.
  - **`prioritize_cls: true`**: Ensures the Class Token is always sent over the strongest available eigen-mode (highest singular value) for maximum reliability.
- **Power Allocation**: Explicitly disabled (`enabled: false`) for these tests to focus purely on the effectiveness of spatial mode allocation.
- **Note on Bottleneck**: Mode allocation is automatically disabled in scenarios where only the bottleneck compression is active without token selection.

### 7.2. AWGN-like Channel (`baseline_mimo.yaml`)
This profile is used as a baseline to simulate standard noisy environments without spatial diversity.
- **Antenna Config**: Number of antennas is set to **1** (SISO), effectively deactivating spatial multiplexing.
- **Fading (`identity`)**: Spatiotemporal fading is disabled. The signal only undergoes Additive White Gaussian Noise (AWGN).
- **Bottleneck Layer**:
  - **`enabled: true`**: Activates a linear projection at the split point.
  - **`out_dim: 128`**: Reduces the latent feature vector size from 192 (ViT-Tiny) to 128 dimensions.
- **Signal Normalization**:
  - **`normalize: true`**: Ensures the transmitted signal has unit average power.
  - **`normalization_mode: sample`**: Normalization is performed on a per-sample basis.
- **Training SNR (`snr_db: [0, 20]`)**: Noise levels are randomly sampled between 0 and 20 dB during training, forcing the model to learn noise-resilient features.
- **Evaluation SNR (`snr_sweep: [-5, 0, 10, 20]`)**: Standard set of noise levels used during validation to characterize performance degradation under adverse conditions.

