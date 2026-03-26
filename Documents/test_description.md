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

## 3. Compression Method (Proposal)
The test evaluates a specific method called **Proposal**, whose primary purpose is to reduce bandwidth usage (data compression) at the split point.

- **Desired Compression Factor**: 0.1 (i.e., a reduction to 10% of the original size).
- **Pooling Mechanism**: The method uses an **Attention Pooling** system to synthesize relevant information from the tokens produced by the ViT before transmission.
- **Parameters**:
  - `desired_compression`: 0.1
  - `pooling`: attention
  - `token_compression`: disabled (null)
  - `batch_compression`: disabled (null)

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
The experimental setup supports multiple communication scenarios to evaluate the robustness of the split-model inference under various channel conditions. These are managed via the `CommModuleWrapper` and configured through different profiles:

### 7.1. Clean Profile (Reference)
Used as a baseline to measure performance without transmission artifacts.
- **Channel**: Disabled (noise-free transmission).
- **Bottleneck**: Can be enabled to test the impact of dimensionality reduction alone (e.g., reducing feature dimension to 128).

### 7.2. Noisy Profile (AWGN Simulation)
Simulates a classic noisy environment using an AWGN (Additive White Gaussian Noise) model.
- **Setup**: Configured as a 1x1 MIMO system with identity fading.
- **Bottleneck**: Active, reducing the input dimension (192) to a bottleneck dimension (128).
- **Training SNR**: Noise is sampled in the range of [0, 20] dB during training to improve robustness.

### 7.3. Diagonal MIMO Profile (Advanced Allocation)
Tests a 4x4 MIMO setup with fixed channel gains, focusing on importance-aware stream allocation.
- **Configuration**: 4 Transmitters (TX) and 4 Receivers (RX).
- **Fading**: Diagonal with fixed gains: `[1.0, 0.8, 0.5, 0.3]`.
- **Stream Allocation**: Uses the `importance_to_gain` strategy, where data chunks are assigned to specific spatial streams based on their measured importance.
- **Power Allocation**: Dynamically adjusted based on `selection_scores` to prioritize critical features.

### 7.4. SVD MIMO Profile (Fading Robustness)
Evaluates performance over a realistic Rayleigh fading channel using Singular Value Decomposition (SVD).
- **Configuration**: 4x4 MIMO setup.
- **Fading**: Rayleigh (stochastic fading).
- **Mode Allocation**: Uses the `importance_to_modes` strategy. It decomposes the channel matrix using SVD and maps the most important tokens to the strongest eigen-modes of the channel.
- **Equalizer**: MMSE (Minimum Mean Square Error) is used for signal recovery.

### 7.5. Evaluation Sweep
Across all noisy scenarios, the model is evaluated over a wide Signal-to-Noise Ratio (SNR) sweep, typically including **-5, 0, 5, 10, and 20 dB**, to characterize the performance degradation under increasingly adverse conditions.
