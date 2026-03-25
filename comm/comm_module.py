import logging
from collections.abc import Mapping

import torch
import torch.nn as nn
import copy
from .bottleneck import Bottleneck
from .mimo import MIMOAWGNChannel, pack_tokens_to_mimo_symbols, unpack_mimo_symbols_to_tokens

logger = logging.getLogger(__name__)


def _to_plain_dict(cfg):
    """Recursively convert Hydra DictConfig / ListConfig to plain Python
    dicts and lists so that isinstance(x, dict) checks always work."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    if isinstance(cfg, Mapping):
        return {k: _to_plain_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return type(cfg)(_to_plain_dict(v) for v in cfg)
    return cfg

class CommModule(nn.Module):
    """
    Communication Module that orchestrates:
    Input (D) -> [Bottleneck (D->O)] -> [Channel] -> [Bottleneck (O->D)] -> Output (D)
    
    Configuration is passed via a dictionary for flexibility.
    """
    def __init__(self, input_dim, config):
        super().__init__()
        
        # Parse Config
        # Expected structure:
        # comm:
        #   enabled: bool
        #   bottleneck:
        #     enabled: bool
        #     out_dim: int
        #   channel:
        #     enabled: bool
        #     snr_db: float
        #     normalize: bool
        
        comm_cfg = config.get("comm", {})
        self.enabled = comm_cfg.get("enabled", False)
        
        # Bottleneck
        bn_cfg = comm_cfg.get("bottleneck", {})
        self.use_bottleneck = bn_cfg.get("enabled", False)
        bn_out_dim = bn_cfg.get("out_dim", input_dim)
        
        # Channel
        ch_cfg = _to_plain_dict(comm_cfg.get("channel", {}))
        self.channel_cfg = copy.deepcopy(ch_cfg)
        self.use_channel = ch_cfg.get("enabled", False)
        self.channel_type = ch_cfg.get("type", "mimo")
        self.power_alloc_cfg = ch_cfg.get("power_alloc", {})
        self.stream_alloc_cfg = ch_cfg.get("stream_alloc", {})
        self.mode_alloc_cfg = ch_cfg.get("mode_alloc", {})
        
        # Components
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(input_dim, bn_out_dim)
        else:
            self.bottleneck = None
            bn_out_dim = input_dim 
            
        if self.use_channel:
            self._build_channel(config.get("comm", {}).get("channel", {}))
        else:
            self.channel = None
            
        self.bn_out_dim = bn_out_dim
        
    def _build_channel(self, ch_cfg):
        ch_cfg = _to_plain_dict(ch_cfg)
        self.channel_cfg = copy.deepcopy(ch_cfg)
        ch_type = ch_cfg.get("type", "mimo")
        self.channel_type = ch_type
        self.power_alloc_cfg = ch_cfg.get("power_alloc", {})
        self.stream_alloc_cfg = ch_cfg.get("stream_alloc", {})
        self.mode_alloc_cfg = ch_cfg.get("mode_alloc", {})
        
        if ch_type == "mimo":
            sample_mode = str(ch_cfg.get("sample_mode", "")).lower()
            if sample_mode:
                if sample_mode not in {"per_sample", "per_batch"}:
                    raise ValueError(f"Unknown MIMO sample_mode: {sample_mode}")
                sample_h_per_batch = (sample_mode == "per_batch")
            else:
                sample_h_per_batch = bool(ch_cfg.get("sample_h_per_batch", False))

            mmse_cfg = ch_cfg.get("mmse", {}) or {}
            mmse_eps = float(mmse_cfg.get("eps", 1e-6))

            self.channel = MIMOAWGNChannel(
                n_tx=ch_cfg.get("n_tx", 2),
                n_rx=ch_cfg.get("n_rx", 2),
                snr_db=ch_cfg.get("snr_db", 10.0),
                train_mode=ch_cfg.get("train_mode", "fixed"),
                normalize=ch_cfg.get("normalize", True),
                normalization_mode=ch_cfg.get("normalization_mode", "sample"),
                fading=ch_cfg.get("fading", "rayleigh"),
                equalizer=ch_cfg.get("equalizer", "mmse"),
                sample_h_per_batch=sample_h_per_batch,
                diagonal_cfg=ch_cfg.get("diagonal", {}),
                diagonal_gains=ch_cfg.get("diagonal_gains", None),
                mmse_eps=mmse_eps,
            )

    def _resolve_power_scores(self, tx_signal, selection_indices, selection_scores, source):
        bsz, n_tokens, _ = tx_signal.shape
        n_patches = max(0, n_tokens - 1)
        if n_patches == 0:
            return tx_signal.new_ones((bsz, 0))

        if source == "uniform":
            return tx_signal.new_ones((bsz, n_patches))

        # source in {"selection_scores", "attention_cls"} share the same score path.
        if selection_scores is None:
            return tx_signal.new_ones((bsz, n_patches))

        if not torch.is_tensor(selection_scores):
            return tx_signal.new_ones((bsz, n_patches))

        scores = selection_scores.to(device=tx_signal.device, dtype=tx_signal.dtype)
        if scores.dim() != 2 or scores.shape[0] != bsz or scores.shape[1] <= 0:
            return tx_signal.new_ones((bsz, n_patches))

        if selection_indices is not None and torch.is_tensor(selection_indices) and selection_indices.shape[1] >= n_tokens:
            patch_indices = selection_indices[:, 1:n_tokens] - 1
            patch_indices = patch_indices.clamp(min=0, max=scores.shape[1] - 1)
            return torch.gather(scores, 1, patch_indices)

        if scores.shape[1] >= n_patches:
            return scores[:, :n_patches]

        # Fallback if score tensor is shorter than requested patch tokens.
        pad = tx_signal.new_ones((bsz, n_patches - scores.shape[1]))
        return torch.cat([scores, pad], dim=1)

    def _apply_power_allocation(self, tx_signal, selection_indices=None, selection_scores=None):
        cfg = self.power_alloc_cfg or {}
        enabled = bool(cfg.get("enabled", False))
        alpha = float(cfg.get("alpha", 1.0))
        source = str(cfg.get("source", "selection_scores"))
        eps = float(cfg.get("eps", 1e-4))
        apply_to_cls = bool(cfg.get("apply_to_cls", False))

        stats = {
            "power_alloc_enabled": enabled,
            "power_alloc_alpha": alpha,
            "power_alloc_source": source,
            "p_mean": 1.0,
            "p_min": 1.0,
            "p_max": 1.0,
        }
        if not enabled:
            return tx_signal, stats

        bsz, n_tokens, _ = tx_signal.shape
        weights = tx_signal.new_ones((bsz, n_tokens))
        if n_tokens > 1:
            patch_scores = self._resolve_power_scores(tx_signal, selection_indices, selection_scores, source)
            patch_weights = (torch.clamp(patch_scores, min=0.0) + eps) ** alpha
            patch_weights = patch_weights / patch_weights.mean(dim=1, keepdim=True).clamp_min(1e-9)
            weights[:, 1:] = patch_weights

        if apply_to_cls:
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-9)

        scaled = tx_signal * torch.sqrt(weights).unsqueeze(-1)
        active = weights if apply_to_cls else (weights[:, 1:] if n_tokens > 1 else weights)

        stats["p_mean"] = float(active.mean().item())
        stats["p_min"] = float(active.min().item())
        stats["p_max"] = float(active.max().item())
        return scaled, stats

    def _resolve_stream_alloc_scores(self, tx_signal, selection_indices, selection_scores, source, prioritize_cls):
        bsz, n_tokens, _ = tx_signal.shape
        scores = tx_signal.new_ones((bsz, n_tokens))
        if n_tokens > 1:
            scores[:, 1:] = self._resolve_power_scores(
                tx_signal,
                selection_indices,
                selection_scores,
                source,
            )
        if prioritize_cls and n_tokens > 0:
            if n_tokens > 1:
                cls_score = scores[:, 1:].max(dim=1).values + 1.0
            else:
                cls_score = tx_signal.new_ones((bsz,))
            scores[:, 0] = cls_score
        return scores

    def _build_stream_src_order(self, token_scores, d_sent, granularity, chunk_size, prioritize_cls):
        n_tokens = token_scores.shape[0]
        units = []
        if n_tokens > 0 and prioritize_cls:
            units.append((float("inf"), 0, 1))
            patch_start = 1
        else:
            patch_start = 0

        step = 1 if granularity == "token" else chunk_size
        for start in range(patch_start, n_tokens, step):
            end = min(start + step, n_tokens)
            score = float(token_scores[start:end].mean().item())
            units.append((score, start, end))

        if prioritize_cls:
            prefix = units[:1]
            body = units[1:]
        else:
            prefix = []
            body = units

        body.sort(key=lambda item: (-item[0], item[1]))
        ordered = prefix + body

        flat_indices = []
        for _, start, end in ordered:
            for token_idx in range(start, end):
                base = token_idx * d_sent
                flat_indices.extend(range(base, base + d_sent))
        return flat_indices

    def _build_stream_alloc_positions(self, gains, t):
        bsz, n_tx = gains.shape
        if t == 0:
            return gains.new_zeros((bsz, 0), dtype=torch.long)

        orders = []
        for batch_idx in range(bsz):
            ranked = sorted(
                range(n_tx),
                key=lambda row_idx: (-float(gains[batch_idx, row_idx].item()), row_idx),
            )
            orders.append(ranked)
        stream_order = torch.tensor(orders, device=gains.device, dtype=torch.long)
        cols = torch.arange(t, device=gains.device, dtype=torch.long).view(1, 1, t)
        row_major = stream_order.unsqueeze(-1) * t + cols
        return row_major.reshape(bsz, -1)

    def _pack_mimo_symbols(
        self,
        tx_signal,
        *,
        selection_indices=None,
        selection_scores=None,
        generator=None,
    ):
        packed_default, pack_stats = pack_tokens_to_mimo_symbols(tx_signal, n_tx=self.channel.n_tx)
        cfg = self.stream_alloc_cfg or {}
        enabled = bool(cfg.get("enabled", False))

        # Inizializziamo i default sicuri per i log (tutto spento)
        strategy = "none"
        assignment_enabled = False
        stream_power_enabled = False

        if enabled:
            # Leggiamo le sub-configurazioni SOLO se il modulo principale è attivo
            strategy = str(cfg.get("strategy", "importance_to_gain"))
            assignment_cfg = cfg.get("assignment", {}) or {}
            stream_power_cfg = cfg.get("power", {}) or {}
            
            assignment_enabled = bool(assignment_cfg.get("enabled", True))
            stream_power_enabled = bool(stream_power_cfg.get("enabled", False))

        alloc_stats = {
            "stream_alloc_enabled": float(enabled),
            "stream_alloc_strategy": strategy,
            "stream_alloc_assignment_enabled": float(assignment_enabled),
            "stream_alloc_power_enabled": float(stream_power_enabled),
        }
        if not enabled:
            return packed_default, pack_stats, None, alloc_stats, None
        if strategy != "importance_to_gain":
            raise ValueError(f"Unknown stream allocation strategy: {strategy}")

        bsz, n_tokens, d_sent = tx_signal.shape
        l = pack_stats["mimo_L"]
        t = pack_stats["mimo_T"]
        l_pad = pack_stats["mimo_L_pad"]
        device = tx_signal.device
        dtype = tx_signal.dtype

        source = str(cfg.get("source", self.power_alloc_cfg.get("source", "selection_scores")))
        prioritize_cls = bool(assignment_cfg.get("prioritize_cls", True))
        granularity = str(assignment_cfg.get("granularity", "token"))
        if granularity not in {"token", "chunk"}:
            raise ValueError(f"Unknown stream allocation granularity: {granularity}")
        chunk_size = int(assignment_cfg.get("chunk_size", 1))
        if chunk_size <= 0:
            raise ValueError(f"stream_alloc.assignment.chunk_size must be > 0, got {chunk_size}")

        if self.channel_type == "mimo" and getattr(self.channel, "fading", None) == "diagonal":
            diagonal_gains = self.channel.sample_diagonal_gains(
                bsz,
                device=device,
                generator=generator,
                dtype=dtype,
            )
        else:
            diagonal_gains = None

        gains = (
            diagonal_gains
            if diagonal_gains is not None
            else tx_signal.new_ones((bsz, self.channel.n_tx))
        )
        positions = self._build_stream_alloc_positions(gains, t)[:, :l]

        flat = tx_signal.reshape(bsz, l)
        if assignment_enabled and l > 0:
            token_scores = self._resolve_stream_alloc_scores(
                tx_signal,
                selection_indices,
                selection_scores,
                source,
                prioritize_cls,
            )
            src_orders = []
            for batch_idx in range(bsz):
                src_order = self._build_stream_src_order(
                    token_scores[batch_idx],
                    d_sent,
                    granularity,
                    chunk_size,
                    prioritize_cls,
                )
                src_orders.append(src_order)
            src_order = torch.tensor(src_orders, device=device, dtype=torch.long)
            ordered_flat = flat.gather(1, src_order)
        else:
            src_order = torch.arange(l, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            ordered_flat = flat

        packed_flat = tx_signal.new_zeros((bsz, l_pad))
        if l > 0:
            packed_flat.scatter_(1, positions, ordered_flat)

        packed = packed_flat.reshape(bsz, self.channel.n_tx, t)

        if stream_power_enabled and l > 0:
            alpha = float(stream_power_cfg.get("alpha", 1.0))
            eps   = float(stream_power_cfg.get("eps",   1e-4))
            gain_alpha = float(stream_power_cfg.get("gain_alpha", 1.0))
            max_power_ratio = float(stream_power_cfg.get("max_power_ratio", 10.0))

            # --- Build per-stream (per-antenna) importance weights ---
            # Strategy: for each antenna row, average the importance scores
            # of the tokens that were mapped to it via 'positions'.
            # positions shape: [B, l] — each entry is a flat index in l_pad.
            # Flat index k corresponds to antenna row k // t, column k % t.
            if assignment_enabled and "token_scores" in locals():
                # token_scores: [B, N_tokens], one score per token
                # Expand token scores to flat dimension l (pixels)
                # by repeating each token score d_sent times.
                flat_token_scores = token_scores.unsqueeze(-1).expand(
                    bsz, n_tokens, d_sent
                ).reshape(bsz, l)                          # [B, l]

                # Scatter token scores into the padded grid using positions,
                # then average per antenna row.
                score_grid = tx_signal.new_zeros((bsz, l_pad))  # [B, n_tx*T]
                if positions.shape[1] > 0:
                    score_grid.scatter_(1, positions, flat_token_scores)
                score_grid = score_grid.reshape(bsz, self.channel.n_tx, t)   # [B, n_tx, T]

                # Mean score per antenna (ignore padded zeros via count)
                count_grid = tx_signal.new_zeros((bsz, l_pad))
                ones_flat  = tx_signal.new_ones((bsz, l))
                if positions.shape[1] > 0:
                    count_grid.scatter_(1, positions, ones_flat)
                count_grid = count_grid.reshape(bsz, self.channel.n_tx, t).sum(dim=2)  # [B, n_tx]
                sum_grid   = score_grid.sum(dim=2)                                     # [B, n_tx]
                weights    = sum_grid / count_grid.clamp_min(1.0)                      # [B, n_tx]
                weights    = (weights.clamp(min=0.0) + eps) ** alpha
            else:
                # Fallback: uniform weights across all antennas
                weights = tx_signal.new_ones((bsz, self.channel.n_tx))

            # Couple stream power with channel quality so weak streams are
            # naturally down-weighted before MMSE inversion.
            if gain_alpha != 0.0:
                gain_weights = gains.clamp_min(1e-6) ** gain_alpha
                weights = weights * gain_weights

            # Enforce a true per-sample max/min ratio cap before normalization.
            # Dividing by the mean does not change max/min, so this guarantee
            # is preserved after normalization.
            weights = weights.clamp_min(1e-9)
            if max_power_ratio > 1.0:
                max_w = weights.max(dim=1, keepdim=True).values
                min_allowed = max_w / max_power_ratio
                weights = torch.maximum(weights, min_allowed)

            # Normalise so mean power across antennas stays constant
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-9)
            pre_power  = packed.pow(2).mean(dim=(1, 2), keepdim=True)
            packed     = packed * torch.sqrt(weights).unsqueeze(-1)
            post_power = packed.pow(2).mean(dim=(1, 2), keepdim=True)
            # Rescale to preserve total power budget
            packed = packed * torch.sqrt(pre_power / post_power.clamp_min(1e-9))
            alloc_stats["stream_alloc_power_alpha"] = alpha
            alloc_stats["stream_alloc_power_gain_alpha"] = gain_alpha
            alloc_stats["stream_alloc_power_max_ratio"] = max_power_ratio
            alloc_stats["stream_alloc_power_min"]   = float(weights.min().item())
            alloc_stats["stream_alloc_power_max"]   = float(weights.max().item())
            sample_ratio = weights.max(dim=1).values / weights.min(dim=1).values.clamp_min(1e-9)
            alloc_stats["stream_alloc_power_ratio"] = float(sample_ratio.mean().item())
            alloc_stats["stream_alloc_power_ratio_max"] = float(sample_ratio.max().item())

        # Stream-level alignment metric: how often top-importance tokens
        # are mapped to strongest streams (gain-ranked).
        stream_top_imp_frac = 0.0
        if assignment_enabled and l > 0:
            token_scores_eval = self._resolve_stream_alloc_scores(
                tx_signal,
                selection_indices,
                selection_scores,
                source,
                prioritize_cls,
            )

            row_ids = torch.div(positions, t, rounding_mode="floor").clamp(min=0, max=self.channel.n_tx - 1)
            symbol_gains = torch.gather(gains, 1, row_ids)
            token_ids = torch.div(src_order, d_sent, rounding_mode="floor").clamp(min=0, max=n_tokens - 1)

            token_gain_sum = tx_signal.new_zeros((bsz, n_tokens))
            token_gain_cnt = tx_signal.new_zeros((bsz, n_tokens))
            ones = tx_signal.new_ones((bsz, l))
            token_gain_sum.scatter_add_(1, token_ids, symbol_gains)
            token_gain_cnt.scatter_add_(1, token_ids, ones)

            token_gain_mean = token_gain_sum / token_gain_cnt.clamp_min(1.0)
            token_gain_mean = token_gain_mean.masked_fill(token_gain_cnt <= 0, float("-inf"))

            overlaps = []
            for b in range(bsz):
                n_sent_tokens = int((token_gain_cnt[b] > 0).sum().item())
                if n_sent_tokens <= 0:
                    continue
                k_eval = max(1, min(n_sent_tokens, n_tokens // 2))
                top_imp_tokens = token_scores_eval[b].topk(k_eval).indices.tolist()
                top_gain_tokens = token_gain_mean[b].topk(k_eval).indices.tolist()
                overlap = len(set(top_imp_tokens) & set(top_gain_tokens)) / float(k_eval)
                overlaps.append(overlap)

            if overlaps:
                stream_top_imp_frac = float(sum(overlaps) / len(overlaps))

        alloc_ctx = {
            "positions": positions,
            "src_order": src_order,
            "tokens_sent": n_tokens,
            "d_sent": d_sent,
            "original_l": l,
        }
        alloc_stats["stream_alloc_gain_min"] = float(gains.min().item())
        alloc_stats["stream_alloc_gain_mean"] = float(gains.mean().item())
        alloc_stats["stream_alloc_gain_max"] = float(gains.max().item())
        alloc_stats["stream_alloc_top_imp_frac"] = stream_top_imp_frac
        return packed, pack_stats, alloc_ctx, alloc_stats, diagonal_gains

    # ------------------------------------------------------------------
    # SVD Mode Allocation (Rayleigh MIMO)
    # ------------------------------------------------------------------

    def _compute_svd_modes(self, h, eps=1e-6):
        """Compute SVD of H. Returns (V, sigma) or None on failure."""
        try:
            u, sigma, vh = torch.linalg.svd(h, full_matrices=False)
            if torch.isnan(sigma).any() or torch.isinf(sigma).any():
                logger.warning("SVD produced NaN/Inf singular values; disabling mode_alloc for this batch.")
                return None
            return vh.transpose(-2, -1), sigma  # V [B, n_tx, K], sigma [B, K]
        except RuntimeError as e:
            logger.warning(f"SVD failed ({e}); disabling mode_alloc for this batch.")
            return None

    def _apply_mode_alloc(
        self,
        packed,
        h,
        *,
        tx_signal,
        selection_indices,
        selection_scores,
    ):
        """Apply SVD mode allocation to packed signal.

        1. SVD(H) → V, Σ
        2. S_mode = V^T @ S
        3. Assignment + optional power in mode domain (using Σ as gains)
        4. S' = V @ S_mode'

        Returns (s_out, stats) or (packed, empty_stats) on SVD failure.
        """
        cfg = self.mode_alloc_cfg or {}
        svd_cfg = cfg.get("svd", {}) or {}
        eps = float(svd_cfg.get("eps", 1e-6))
        prune_cfg = cfg.get("prune", {}) or {}
        prune_enabled = bool(prune_cfg.get("enabled", True))
        prune_rel_threshold = float(prune_cfg.get("sigma_rel_threshold", 0.1))
        assignment_cfg = cfg.get("assignment", {}) or {}
        power_cfg = cfg.get("power", {}) or {}
        assignment_enabled = bool(assignment_cfg.get("enabled", True))
        power_enabled = bool(power_cfg.get("enabled", False))
        num_modes = cfg.get("num_modes", None)

        bsz, n_tx, t = packed.shape
        device = packed.device
        dtype = packed.dtype

        empty_stats = {"mode_alloc_enabled": False}

        svd_result = self._compute_svd_modes(h, eps=eps)
        if svd_result is None:
            # Bug 1 fix: return 3 values so the caller's unpack always works.
            return packed, empty_stats, None

        v_mat, sigma = svd_result  # V [B, n_tx, K], sigma [B, K]
        k = sigma.shape[1]

        # Optionally restrict to top-m modes
        if num_modes is not None:
            m = min(int(num_modes), k)
            v_mat = v_mat[:, :, :m]
            sigma = sigma[:, :m]
            k = m

        # Relative-sigma pruning mask used for assignment prioritization and
        # for zeroing power on weak modes.
        prune_mask = torch.zeros_like(sigma, dtype=torch.bool)
        sigma_for_assignment = sigma
        if prune_enabled and k > 0:
            sigma_max = sigma.max(dim=1, keepdim=True).values.clamp_min(1e-9)
            prune_mask = sigma < (prune_rel_threshold * sigma_max)
            # Keep at least one mode active per sample.
            all_pruned = prune_mask.all(dim=1)
            if all_pruned.any():
                best_mode = sigma.argmax(dim=1)
                prune_mask[all_pruned, best_mode[all_pruned]] = False
            sigma_for_assignment = sigma.masked_fill(prune_mask, 0.0)

        # Transform to mode domain: S_mode = V^T @ S
        vt = v_mat.transpose(-2, -1)  # [B, K, n_tx]
        s_mode = torch.matmul(vt, packed)  # [B, K, T]

        # Assignment: reorder symbols so important tokens → strong modes
        if assignment_enabled and t > 0 and tx_signal is not None:
            source = str(cfg.get("source", self.power_alloc_cfg.get("source", "selection_scores")))
            prioritize_cls = bool(assignment_cfg.get("prioritize_cls", True))
            granularity = str(assignment_cfg.get("granularity", "token"))
            chunk_size = int(assignment_cfg.get("chunk_size", 1))

            _, n_tokens, d_sent = tx_signal.shape
            l = n_tokens * d_sent

            # Build mode positions: map strongest modes to earliest flat positions
            positions = self._build_stream_alloc_positions(sigma_for_assignment[:, :k], t)[:, :l]

            # Build source order from importance scores
            token_scores = self._resolve_stream_alloc_scores(
                tx_signal, selection_indices, selection_scores, source, prioritize_cls,
            )
            src_orders = []
            for batch_idx in range(bsz):
                src_order = self._build_stream_src_order(
                    token_scores[batch_idx], d_sent, granularity, chunk_size, prioritize_cls,
                )
                src_orders.append(src_order)
            src_order_t = torch.tensor(src_orders, device=device, dtype=torch.long)

            # Bug 2 fix: reorder operates on flat_mode (mode domain), not on
            # packed (antenna domain).  flat_orig / flat_orig.gather was using
            # the wrong domain — the scatter into the K×T grid must come from
            # the already-projected s_mode so the V @ s_mode multiplication
            # at the end is mathematically coherent.
            # Flatten mode-domain signal, reorder, re-pack
            # Bug 1 fix: if k * t < l (user chose num_modes < n_tx), we can't fit
            # all L original symbols into the K*T mode capacity. We must truncate
            # the assignment to the top min(l, k*t) most important tokens.
            l_assign = min(l, k * t)

            # Limit sources and positions to available capacity
            src_order_trunc = src_order_t[:, :l_assign]
            positions_trunc = positions[:, :l_assign]

            flat_mode = s_mode.reshape(bsz, -1)          # [B, K*T]  mode domain
            ordered_flat = flat_mode.gather(1, src_order_trunc)  # [B, l_assign]

            l_pad = k * t
            packed_mode = packed.new_zeros((bsz, l_pad))
            if l_assign > 0 and positions_trunc.shape[1] > 0:
                packed_mode.scatter_(1, positions_trunc, ordered_flat)
            s_mode = packed_mode.reshape(bsz, k, t)

            # Save context for unpack — include V and k so _unpack_mode_alloc
            # can apply the inverse V^T projection after equalization.
            mode_alloc_ctx = {
                "positions": positions_trunc,
                "src_order": src_order_trunc,
                "l_assign": l_assign,
                "v_mat": v_mat,   # [B, n_tx, K]  — needed for V^T in unpack
                "k": k,
            }
        else:
            mode_alloc_ctx = None

        # Optional power allocation in mode domain (importance-based, no CSI)
        if power_enabled and t > 0:
            alpha     = float(power_cfg.get("alpha", 1.0))
            power_eps = float(power_cfg.get("eps",   1e-4))

            # Bug 4 fix: compute per-mode weights by aggregating importance
            # scores of the tokens assigned to each mode row, using the same
            # positions scatter as the assignment step.
            # Previously: mean_imp was a scalar expanded uniformly → all modes
            # identical. Now: each mode k gets the average score of its tokens.
            if tx_signal is not None and mode_alloc_ctx is not None:
                source = str(cfg.get("source", self.power_alloc_cfg.get("source", "selection_scores")))
                imp_scores = self._resolve_stream_alloc_scores(
                    tx_signal, selection_indices, selection_scores, source,
                    bool((cfg.get("assignment", {}) or {}).get("prioritize_cls", True)),
                )
                _, n_tokens_tx, d_sent_tx = tx_signal.shape
                l_imp = n_tokens_tx * d_sent_tx
                # Expand per-token score to flat token*dim space
                flat_imp = imp_scores.unsqueeze(-1).expand(
                    bsz, n_tokens_tx, d_sent_tx
                ).reshape(bsz, l_imp)                         # [B, l]
                # Re-use positions from mode_alloc_ctx (same scatter grid)
                pos_imp = mode_alloc_ctx["positions"]          # [B, l]
                score_grid = packed.new_zeros((bsz, k * t))
                count_grid = packed.new_zeros((bsz, k * t))
                ones_flat   = packed.new_ones((bsz, l_imp))
                if pos_imp.shape[1] > 0:
                    score_grid.scatter_(1, pos_imp, flat_imp)
                    count_grid.scatter_(1, pos_imp, ones_flat)
                score_grid = score_grid.reshape(bsz, k, t).sum(dim=2)   # [B, k]
                count_grid = count_grid.reshape(bsz, k, t).sum(dim=2)   # [B, k]
                per_mode_imp = score_grid / count_grid.clamp_min(1.0)    # [B, k]
                weights = (per_mode_imp.clamp(min=0.0) + power_eps) ** alpha
            elif tx_signal is not None:
                # Assignment disabled: fall back to mean importance uniformly
                source = str(cfg.get("source", self.power_alloc_cfg.get("source", "selection_scores")))
                imp_scores = self._resolve_stream_alloc_scores(
                    tx_signal, selection_indices, selection_scores, source,
                    bool((cfg.get("assignment", {}) or {}).get("prioritize_cls", True)),
                )
                mean_imp = imp_scores.mean(dim=1, keepdim=True).expand(bsz, k)
                weights  = (torch.clamp(mean_imp, min=0.0) + power_eps) ** alpha
            else:
                weights = packed.new_ones((bsz, k))

            # Force zero power on weak singular modes.
            if prune_enabled and k > 0:
                weights = weights.masked_fill(prune_mask, 0.0)
                active_mask = ~prune_mask
                active_count = active_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                active_mean = (weights * active_mask).sum(dim=1, keepdim=True) / active_count
                weights = torch.where(
                    active_mask,
                    weights / active_mean.clamp_min(1e-9),
                    torch.zeros_like(weights),
                )

            weights    = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-9)
            pre_power  = s_mode.pow(2).mean(dim=(1, 2), keepdim=True)
            s_mode     = s_mode * torch.sqrt(weights).unsqueeze(-1)
            post_power = s_mode.pow(2).mean(dim=(1, 2), keepdim=True)
            s_mode     = s_mode * torch.sqrt(pre_power / post_power.clamp_min(1e-9))

        # Transform back: S' = V @ S_mode'
        s_out = torch.matmul(v_mat[:, :, :k], s_mode)  # [B, n_tx, T]

        # Compute quality stats
        with torch.no_grad():
            top_imp_frac = 0.0
            if assignment_enabled and mode_alloc_ctx is not None and tx_signal is not None:
                source = str(cfg.get("source", self.power_alloc_cfg.get("source", "selection_scores")))
                imp = self._resolve_stream_alloc_scores(
                    tx_signal, selection_indices, selection_scores, source,
                    bool((cfg.get("assignment", {}) or {}).get("prioritize_cls", True)),
                )
                _, n_tok, d_sent = tx_signal.shape
                positions = mode_alloc_ctx["positions"]  # [B, l_assign]
                src_order = mode_alloc_ctx["src_order"]  # [B, l_assign]
                l_assign = int(mode_alloc_ctx["l_assign"])

                if l_assign > 0:
                    # Each transmitted flat symbol has a source token id and a destination mode id.
                    token_ids = torch.div(src_order, d_sent, rounding_mode="floor")
                    mode_ids = torch.div(positions, t, rounding_mode="floor").clamp(min=0, max=k - 1)
                    mode_gains = torch.gather(sigma, 1, mode_ids)

                    token_gain_sum = packed.new_zeros((bsz, n_tok))
                    token_gain_cnt = packed.new_zeros((bsz, n_tok))
                    ones = packed.new_ones((bsz, l_assign))
                    token_gain_sum.scatter_add_(1, token_ids, mode_gains)
                    token_gain_cnt.scatter_add_(1, token_ids, ones)

                    token_gain_mean = token_gain_sum / token_gain_cnt.clamp_min(1.0)
                    token_gain_mean = token_gain_mean.masked_fill(token_gain_cnt <= 0, float("-inf"))

                    overlaps = []
                    for b in range(bsz):
                        n_sent_tokens = int((token_gain_cnt[b] > 0).sum().item())
                        if n_sent_tokens <= 0:
                            continue
                        k_eval = max(1, min(n_sent_tokens, n_tok // 2))
                        top_imp_tokens = imp[b].topk(k_eval).indices.tolist()
                        top_gain_tokens = token_gain_mean[b].topk(k_eval).indices.tolist()
                        overlap = len(set(top_imp_tokens) & set(top_gain_tokens)) / float(k_eval)
                        overlaps.append(overlap)

                    if overlaps:
                        top_imp_frac = float(sum(overlaps) / len(overlaps))

        stats = {
            "mode_alloc_enabled": True,
            "mode_alloc_strategy": str(cfg.get("strategy", "importance_to_modes")),
            "svd_per_sample": bool(svd_cfg.get("per_sample", True)),
            "sigma_min": float(sigma.min().item()),
            "sigma_mean": float(sigma.mean().item()),
            "sigma_max": float(sigma.max().item()),
            "mode_alloc_prune_enabled": prune_enabled,
            "mode_alloc_prune_rel_threshold": prune_rel_threshold,
            "mode_alloc_pruned_frac": float(prune_mask.float().mean().item()) if k > 0 else 0.0,
            "mode_alloc_assignment_enabled": assignment_enabled,
            "mode_alloc_assignment_granularity": str(assignment_cfg.get("granularity", "token")),
            "mode_alloc_assignment_chunk_size": int(assignment_cfg.get("chunk_size", 1)),
            "mode_alloc_power_enabled": power_enabled,
            "mode_alloc_power_alpha": float(power_cfg.get("alpha", 1.0)) if power_enabled else 0.0,
            "mode_alloc_top_imp_frac": top_imp_frac,
        }
        if num_modes is not None:
            stats["mode_alloc_num_modes"] = int(num_modes)

        return s_out, stats, mode_alloc_ctx

    def _unpack_mode_alloc(self, rx_packed, mode_alloc_ctx, tx_signal_shape, pack_stats):
        """
        Reverse mode-alloc: un-scatter in mode domain, restore token order,
        project back to antenna/token domain with V.

        Bug 2 fix: the equalised signal is in antenna domain [B, n_tx, T].
        To invert the V-projection done at the transmitter we must:
          1. Project to mode domain:  s_mode_rx = V^T @ rx_packed
          2. Un-scatter using saved positions (reverse the assignment)
          3. Restore token order (argsort of src_order)
          4. Reshape to [B, N, D]
        """
        if mode_alloc_ctx is None:
            return rx_packed

        bsz      = rx_packed.shape[0]
        n_tokens = tx_signal_shape[1]
        d_sent   = tx_signal_shape[2]
        l        = pack_stats["mimo_L"]
        v_mat    = mode_alloc_ctx["v_mat"]   # [B, n_tx, K]
        k        = mode_alloc_ctx["k"]

        # Step 1 — Project received (equalised, antenna domain) into mode domain
        # rx_packed shape: [B, n_tx, T]
        vt = v_mat.transpose(-2, -1)                  # [B, K, n_tx]
        s_mode_rx = torch.matmul(vt, rx_packed)        # [B, K, T]

        # Step 2/3 — Un-scatter and restore token assignment order
        # We only transmitted l_assign symbols in mode domain
        l_assign     = mode_alloc_ctx["l_assign"]
        positions    = mode_alloc_ctx["positions"]     # [B, l_assign]
        src_order    = mode_alloc_ctx["src_order"]     # [B, l_assign]
        flat_mode_rx = s_mode_rx.reshape(bsz, -1)      # [B, K*T]

        rx_ordered    = flat_mode_rx.gather(1, positions)          # [B, l_assign]
        restored_flat = rx_ordered.new_zeros((bsz, l))             # [B, L] full size containing zeros for dropped data
        
        if l_assign > 0 and positions.shape[1] > 0:
            restored_flat.scatter_(1, src_order, rx_ordered)

        # Step 4 — Reshape to token domain [B, N, D]
        return restored_flat.reshape(bsz, n_tokens, d_sent)

    # ------------------------------------------------------------------

    def _unpack_mimo_symbols(self, rx_packed, tx_signal_shape, pack_stats, alloc_ctx):
        if alloc_ctx is None:
            return unpack_mimo_symbols_to_tokens(
                rx_packed,
                tokens_sent=tx_signal_shape[1],
                d_sent=tx_signal_shape[2],
                original_l=pack_stats["mimo_L"],
            )

        bsz = rx_packed.shape[0]
        flat = rx_packed.reshape(bsz, -1)
        original_l = alloc_ctx["original_l"]
        rx_ordered = flat.gather(1, alloc_ctx["positions"])
        restore_order = torch.argsort(alloc_ctx["src_order"], dim=1)
        restored = rx_ordered.gather(1, restore_order)
        return restored.reshape(bsz, alloc_ctx["tokens_sent"], alloc_ctx["d_sent"])

    def reconfigure(self, config_snippet):
        """
        Reconfigures the communication module (specifically the channel) in-place.
        Only supports SNR-only overrides (eval-time SNR sweep).

        Args:
            config_snippet: dict part of 'comm' config (e.g. {'channel': {'snr_db': 10}})

        Raises:
            ValueError: if the override contains keys other than 'snr_db'.
        """
        if 'channel' not in config_snippet:
            return

        override_cfg = _to_plain_dict(config_snippet['channel'])
        if not isinstance(override_cfg, dict):
            raise TypeError(
                f"reconfigure: expected dict for channel override, got {type(override_cfg).__name__}"
            )

        non_snr_keys = set(override_cfg.keys()) - {'snr_db'}
        if non_snr_keys:
            raise ValueError(
                f"reconfigure: only 'snr_db' overrides are supported, "
                f"got unexpected keys: {non_snr_keys}"
            )

        if self.channel is None:
            raise RuntimeError("reconfigure: channel is None, cannot update snr_db")

        if 'snr_db' in override_cfg:
            self.channel.snr_db = override_cfg['snr_db']
            if isinstance(self.channel_cfg, dict):
                self.channel_cfg['snr_db'] = override_cfg['snr_db']
            
    def forward(self, x, selection_indices=None, selection_scores=None, generator=None):
        """
        Args:
            x: [B, N, D]
            selection_indices: [B, N] original token indices (optional)
            generator: torch.Generator for sampling.
        Returns:
            out: [B, N, D]
            info: dict with stats
        """
        if not self.enabled:
            return x, {}
            
        info = {}
        curr = x
        
        # 1. Compress
        if self.use_bottleneck:
            tx_signal = self.bottleneck.compressor(curr)
        else:
            tx_signal = curr
            
        # 2. Channel
        if self.use_channel:
            tx_signal, power_stats = self._apply_power_allocation(
                tx_signal,
                selection_indices=selection_indices,
                selection_scores=selection_scores,
            )
            info.update(power_stats)

            if self.channel_type == "mimo":
                packed, pack_stats, alloc_ctx, alloc_stats, diagonal_gains = self._pack_mimo_symbols(
                    tx_signal,
                    selection_indices=selection_indices,
                    selection_scores=selection_scores,
                    generator=generator,
                )

                # SVD mode allocation for non-diagonal MIMO
                mode_alloc_cfg = self.mode_alloc_cfg or {}
                mode_alloc_enabled = (
                    bool(mode_alloc_cfg.get("enabled", False))
                    and getattr(self.channel, "fading", None) != "diagonal"
                )
                mode_alloc_ctx = None
                h_override = None

                if mode_alloc_enabled:
                    # Pre-sample H so SVD and channel use the same realization
                    bsz_ch = packed.shape[0]
                    device_ch = packed.device
                    compute_dtype = self.channel._compute_dtype(packed.dtype)
                    h_pre = self.channel._sample_h(
                        bsz_ch, device_ch, generator, compute_dtype,
                    )
                    packed_ma, ma_stats, mode_alloc_ctx = self._apply_mode_alloc(
                        packed,
                        h_pre,
                        tx_signal=tx_signal,
                        selection_indices=selection_indices,
                        selection_scores=selection_scores,
                    )
                    if ma_stats.get("mode_alloc_enabled", False):
                        packed = packed_ma
                        h_override = h_pre
                    info.update(ma_stats)
                else:
                    info["mode_alloc_enabled"] = False

                ch_kwargs = {"selection_indices": selection_indices, "generator": generator}
                if h_override is not None:
                    ch_kwargs["h_override"] = h_override
                elif diagonal_gains is not None:
                    ch_kwargs["diagonal_gains"] = diagonal_gains

                rx_packed, ch_stats = self.channel(packed, **ch_kwargs)

                # Unpack: mode_alloc reordering → standard MIMO unpack
                if mode_alloc_ctx is not None:
                    rx_signal = self._unpack_mode_alloc(
                        rx_packed, mode_alloc_ctx, tx_signal.shape, pack_stats,
                    )
                else:
                    rx_signal = self._unpack_mimo_symbols(
                        rx_packed, tx_signal.shape, pack_stats, alloc_ctx,
                    )
                pack_stats["mimo_n_rx"] = int(self.channel.n_rx)
                info.update(pack_stats)
                info.update(alloc_stats)
                info.update(ch_stats)
            else:
                rx_signal, ch_stats = self.channel(
                    tx_signal,
                    selection_indices=selection_indices,
                    generator=generator,
                )
                info.update(ch_stats)
        else:
            rx_signal = tx_signal
            
        # 3. Decompress
        if self.use_bottleneck:
            out = self.bottleneck.decompressor(rx_signal)
        else:
            out = rx_signal
            
        # Info
        # Note: input x shape might differ from output out shape if erasure happened
        n_tokens_eff = out.shape[1]
        info['n_tokens_effective'] = n_tokens_eff
        info['n_tokens_full'] = x.shape[1] # Original tokens
        info['dim_channel'] = self.bn_out_dim
        
        # Rate Accounting
        # rate_symbols_factor is a multiplier (1.0 for analog, 0 for pure digital).
        # The actual numeric rate_symbols is: factor * tokens_sent_excl_cls * d_sent.
        
        # Check channel type/stats for overrides
        bits_per_dim = info.get('bits_per_dim', 0)
        # rate_symbols_factor: 1.0 (analog), 0 (digital), 1.0 (hybrid residual)
        if 'rate_symbols' not in info:
             # Default to analog factor if not specified by channel
             rate_symbols_factor = 1.0
        else:
             rate_symbols_factor = info['rate_symbols']
        
        # Custom Primitive Tracking
        tokens_selected = max(0, x.shape[1] - 1) # Assumes 1 CLS
        tokens_sent = max(0, n_tokens_eff - 1)
        
        info['tokens_selected'] = tokens_selected
        info['tokens_sent'] = tokens_sent
        info['tokens_capacity'] = tokens_sent
        info['d_sent'] = self.bn_out_dim
        info['tokens_sent_excl_cls'] = tokens_sent
        info['rate_symbols_multiplier'] = float(rate_symbols_factor)
        
        # Compute actual numeric rate_symbols
        rate_symbols_value = float(rate_symbols_factor) * tokens_sent * self.bn_out_dim
        info['rate_symbols'] = rate_symbols_value
        info['rate_symbols_def'] = f"{rate_symbols_factor} * tokens_sent_excl_cls * d_sent"
        
        if bits_per_dim > 0:
             # Digital or Hybrid
             info['rate_bits'] = tokens_sent * self.bn_out_dim * bits_per_dim
        else:
             info['rate_bits'] = 0.0
              
        # Symbols: n_tokens * dim * factor
        # If AWGN (default): factor=1. If Quant(pure): factor=0.
        info['q_symbols'] = n_tokens_eff * self.bn_out_dim * rate_symbols_factor
            
        return out, info
