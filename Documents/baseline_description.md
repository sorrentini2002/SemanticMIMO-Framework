# Baseline Description and Operational Scenarios

This document extends [test_description.md](test_description.md) and describes the operational scenarios to test in a controlled way on the same split-learning ViT pipeline.

The reference setup (model, dataset, split index, optimization, and evaluation logic) remains aligned with the current test description:
- Model: DeiT Tiny (`deit_tiny_patch16_224.fb_in1k`)
- Dataset: CIFAR-100
- Split index: 3
- Optimizer: Adam
- Main metric: validation accuracy (with SNR sweep when channel is enabled)

For the scenarios below, batch selection is intentionally ignored for now. The focus is on:
- channel ON/OFF
- bottleneck compression ON/OFF
- token selection ON/OFF

## 1. Scope and Goal

The goal is to compare clean and noisy communication regimes in a progressive way, separating:
- pure model fine-tuning effects
- channel robustness effects (AWGN-like vs MIMO)
- compression effects (bottleneck, then token selection)

This allows a fair comparison from the simplest baseline to the most constrained communication setting.

## 2. Scenario Matrix

## S1. Clean baseline (no channel, no compression) in train and validation

- Train: channel OFF, bottleneck OFF, token selection OFF
- Validation: channel OFF, bottleneck OFF, token selection OFF

Use case:
- Reference upper-bound for model behavior without transmission artifacts.

Expected outcome:
- Best semantic performance, no communication-induced degradation.

---

## S2. Train clean, validate with open channel (no compression)

Updated target for S2 (as currently requested):
- Train: channel OFF, bottleneck OFF, token selection OFF
- Validation: channel ON, bottleneck OFF, token selection OFF

Sub-scenarios:
- S2-A (AWGN-like): channel configured as MIMO 1x1 with identity fading
- S2-B (MIMO): channel configured as true multi-antenna setting (for example 4x4 diagonal or rayleigh)

Use case:
- Measure generalization gap between clean training and noisy/no-channel-mismatch validation.
- Isolate channel impact without any compression confounder.

Expected outcome:
- Accuracy drop vs S1, larger at lower SNR.
- Difference between AWGN-like and MIMO depends on equalizer and fading profile.

---

## S3. Channel ON, no compression in train and validation

- Train: channel ON, bottleneck OFF, token selection OFF
- Validation: channel ON, bottleneck OFF, token selection OFF

Sub-scenarios:
- S3-A: AWGN-like channel
- S3-B: MIMO channel

Use case:
- Robust training directly in the same communication regime used at validation.

Expected outcome:
- Better robustness than S2 at low SNR due to train/val matching.
- Usually lower clean-performance ceiling than S1.

![Validation Accuracy Evolution per Epoch](../Plots/baseline/plt1.png)
![Validation Accuracy Evolution per Epoch](../Plots/baseline/plt2.png)

---

## S4a. Channel ON + bottleneck only (no token selection)

- Train: channel ON, token selection OFF, bottleneck ON
- Validation: channel ON, token selection OFF, bottleneck ON

Use case:
- Quantify the isolated effect of token compression under channel constraints.

Expected outcome:
- Trade-off between communication efficiency and accuracy.
- Sensitivity to bottleneck output dimension and SNR.

---

## S4b. Channel ON + token selection only (no bottleneck)

- Train: channel ON, token selection ON, bottleneck OFF
- Validation: channel ON, token selection ON, bottleneck OFF

Use case:
- Measure the isolated effect of semantic token filtering without dimensionality reduction.
- Contrast with S4a to understand the relative benefit of token selection alone.

Expected outcome:
- Communication savings from token selection alone, without bottleneck compression.
- Performance likely better than S4a if token selection is well-tuned.
- Baseline for understanding token selection effectiveness under channel constraints.

---

## S4c. Channel ON + bottleneck + token selection

- Train: channel ON, bottleneck ON, token selection ON
- Validation: channel ON, bottleneck ON, token selection ON

Use case:
- Full constrained setting with both semantic token filtering and bottleneck dimensionality reduction.

Expected outcome:
- Strongest communication savings combining both compression mechanisms.
- Accuracy depends on quality of token importance estimation and channel allocation strategy.

Note:
- This scenario combines maximal compression constraints.

![Validation Accuracy Evolution per Epoch](../Plots/baseline/plt3.png)
![Best Validation Accuracy Heatmap](../Plots/baseline/plt4.png)
![Measured Post-Equalization SNR Heatmap](../Plots/baseline/plt5.png)

## 3. Recommended Evaluation Protocol

For each scenario:
- Keep model, dataset, split index, and optimizer fixed.
- Evaluate on the same SNR sweep (for channel-enabled validation).
- Save both final and best metrics with consistent naming.
- Compare against S1 as the clean anchor baseline.

Minimum reporting set:
- Validation accuracy per SNR
- Average validation accuracy over sweep
- Communication stats (symbols, tokens, rates) when available

## 4. Practical Comparison Path

Suggested order:
1. S1 (clean reference)
2. S2-A and S2-B (clean-train vs noisy-val mismatch)
3. S3-A and S3-B (matched noisy train/val)
4. S4a (bottleneck only)
5. S4b (token selection only)
6. S4c (bottleneck + token selection)

This progression gives a clear attribution of performance changes to:
- channel mismatch,
- channel robustness training,
- bottleneck compression effect,
- token selection effect,
- combined compression constraints.

## 5. Results Folder Naming (Current)

- S1 -> `results/S1_clean_baseline/`
- S2-A -> `results/S2A_clean_train_noisy_val_awgn_like/`
- S2-B -> `results/S2B_clean_train_noisy_val_mimo/`
- S3-A -> `results/S3A_channel_on_awgn_like_no_compression/`
- S3-B -> `results/S3B_channel_on_mimo_no_compression/`
- S4a -> `results/S4A_channel_on_bottleneck_on_no_token_selection/`
- S4b -> `results/S4B_channel_on_token_selection_on_no_bottleneck/`
- S4c -> `results/S4C_channel_on_bottleneck_on_token_selection_on/`

## 6. Current Priority

Current implementation priority is S2 with the updated requirement:
- no compression in training,
- channel activated only in validation,
- no token selection in both phases.

This should be treated as the first target extension over the existing baseline workflow.
