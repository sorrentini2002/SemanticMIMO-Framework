import math
from collections.abc import Mapping
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from omegaconf import ListConfig
except Exception:  # pragma: no cover
    ListConfig = ()


def _randn(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Device-safe random normal sampling that respects CPU generators."""
    if generator is not None and generator.device.type == "cpu" and device.type != "cpu":
        return torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _rand(
    shape: Tuple[int, ...],
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Device-safe random uniform sampling that respects CPU generators."""
    if generator is not None and generator.device.type == "cpu" and device.type != "cpu":
        return torch.rand(shape, generator=generator, device="cpu", dtype=dtype).to(device=device, dtype=dtype)
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def pack_tokens_to_mimo_symbols(tokens: torch.Tensor, n_tx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Pack [B, tokens_sent, d_sent] to [B, n_tx, T] with zero-padding.
    """
    if n_tx <= 0:
        raise ValueError(f"n_tx must be > 0, got {n_tx}")

    bsz, tokens_sent, d_sent = tokens.shape
    l = tokens_sent * d_sent
    t = math.ceil(l / n_tx) if l > 0 else 0
    l_pad = n_tx * t
    pad_symbols = l_pad - l

    flat = tokens.reshape(bsz, l)
    if pad_symbols > 0:
        flat = torch.cat([flat, torch.zeros(bsz, pad_symbols, device=tokens.device, dtype=tokens.dtype)], dim=1)

    packed = flat.reshape(bsz, n_tx, t)
    info = {
        "mimo_n_tx": int(n_tx),
        "mimo_T": int(t),
        "mimo_L": int(l),
        "mimo_L_pad": int(l_pad),
        "mimo_pad_symbols": int(pad_symbols),
    }
    return packed, info


def unpack_mimo_symbols_to_tokens(
    symbols: torch.Tensor,
    *,
    tokens_sent: int,
    d_sent: int,
    original_l: int,
) -> torch.Tensor:
    """
    Unpack [B, n_tx, T] back to [B, tokens_sent, d_sent], trimming padded symbols.
    """
    bsz = symbols.shape[0]
    flat = symbols.reshape(bsz, -1)
    if original_l > flat.shape[1]:
        raise ValueError(f"original_l={original_l} exceeds packed size={flat.shape[1]}")
    flat = flat[:, :original_l]
    return flat.reshape(bsz, tokens_sent, d_sent)


class MIMOAWGNChannel(nn.Module):
    """
    Real-valued MIMO channel:
      Y = H S + N
    with linear equalization (ZF/MMSE) at the receiver.
    """

    def __init__(
        self,
        *,
        n_tx: int = 2,
        n_rx: int = 2,
        snr_db=10.0,
        train_mode: str = "fixed",
        normalize: bool = True,
        normalization_mode: str = "sample",
        fading: str = "rayleigh",
        equalizer: str = "mmse",
        sample_h_per_batch: bool = False,
        diagonal_cfg=None,
        diagonal_gains=None,
        mmse_eps: float = 1e-6,
    ):
        super().__init__()
        if n_tx <= 0 or n_rx <= 0:
            raise ValueError(f"n_tx and n_rx must be > 0, got n_tx={n_tx}, n_rx={n_rx}")
        if equalizer not in {"zf", "mmse"}:
            raise ValueError(f"Unknown equalizer: {equalizer}")
        if fading not in {"rayleigh", "identity", "diagonal"}:
            raise ValueError(f"Unknown fading: {fading}")
        if normalization_mode not in {"sample", "batch"}:
            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

        self.n_tx = int(n_tx)
        self.n_rx = int(n_rx)
        self.snr_db = snr_db
        self.train_mode = train_mode
        self.normalize = bool(normalize)
        self.normalization_mode = normalization_mode
        self.fading = fading
        self.equalizer = equalizer
        self.sample_h_per_batch = bool(sample_h_per_batch)
        self.diagonal_cfg = dict(diagonal_cfg or {})
        if diagonal_gains is not None and "gains" not in self.diagonal_cfg:
            self.diagonal_cfg["gains"] = diagonal_gains
        self.diagonal_gains = self.diagonal_cfg.get("gains", None)
        self.mmse_eps = float(mmse_eps)

        if self.fading == "diagonal" and self.n_rx != self.n_tx:
            raise ValueError(
                f"Diagonal MIMO requires n_rx == n_tx, got n_tx={self.n_tx}, n_rx={self.n_rx}"
            )
        if self.fading == "diagonal":
            self._validate_diagonal_config()

    def _resolve_snr(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> float:
        is_snr_sequence = isinstance(self.snr_db, (list, tuple, ListConfig))
        if self.training and self.train_mode == "sampled":
            if is_snr_sequence:
                gen_device = generator.device if generator is not None else x.device
                r = torch.rand(1, generator=generator, device=gen_device).to(x.device).item()
                if len(self.snr_db) == 2:
                    low, high = self.snr_db
                    return float(low + (high - low) * r)
                idx = int(r * len(self.snr_db))
                idx = min(idx, len(self.snr_db) - 1)
                return float(self.snr_db[idx])
            return float(self.snr_db)
        if is_snr_sequence:
            return float(max(self.snr_db))
        return float(self.snr_db)

    def _get_diagonal_random_cfg(self) -> dict:
        random_cfg = self.diagonal_cfg.get("random", {})
        if random_cfg is None:
            return {}
        if isinstance(random_cfg, Mapping):
            return dict(random_cfg)
        raise ValueError("comm.channel.diagonal.random must be a mapping when provided.")

    def _validate_diagonal_config(self) -> None:
        random_cfg = self._get_diagonal_random_cfg()

        random_enabled = bool(random_cfg.get("enabled", False))
        if random_enabled:
            distribution = str(random_cfg.get("distribution", "uniform")).lower()
            if distribution not in {"uniform", "lognormal"}:
                raise ValueError(f"Unknown diagonal gain distribution: {distribution}")
            min_gain = float(random_cfg.get("min_gain", 0.2))
            max_gain = float(random_cfg.get("max_gain", 1.0))
            if min_gain <= 0.0 or max_gain <= 0.0:
                raise ValueError("Diagonal random gains must be strictly positive.")
            if min_gain > max_gain:
                raise ValueError(
                    f"Diagonal random gains require min_gain <= max_gain, got {min_gain} > {max_gain}"
                )
            return

        gains = self._fixed_diagonal_gains(device=torch.device("cpu"), dtype=torch.float32)
        if gains.numel() != self.n_tx:
            raise ValueError(
                f"Diagonal MIMO requires exactly {self.n_tx} configured gains, got {gains.numel()}"
            )

    def _fixed_diagonal_gains(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gains_cfg = self.diagonal_cfg.get("gains", self.diagonal_gains)
        if gains_cfg is None:
            gains = torch.ones(self.n_tx, device=device, dtype=dtype)
        else:
            gains = torch.as_tensor(gains_cfg, device=device, dtype=dtype).flatten()

        if gains.numel() != self.n_tx:
            raise ValueError(
                f"Diagonal MIMO requires exactly {self.n_tx} configured gains, got {gains.numel()}"
            )
        if torch.any(gains <= 0):
            raise ValueError("Diagonal MIMO gains must be strictly positive.")
        return gains

    def sample_diagonal_gains(
        self,
        bsz: int,
        *,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.fading != "diagonal":
            raise ValueError("sample_diagonal_gains is only valid when fading='diagonal'.")

        random_cfg = self._get_diagonal_random_cfg()
        random_enabled = bool(random_cfg.get("enabled", False))
        base_shape = (1, self.n_tx) if self.sample_h_per_batch else (bsz, self.n_tx)

        if not random_enabled:
            gains = self._fixed_diagonal_gains(device=device, dtype=dtype).unsqueeze(0)
            return gains.expand(bsz, -1)

        distribution = str(random_cfg.get("distribution", "uniform")).lower()
        min_gain = float(random_cfg.get("min_gain", 0.2))
        max_gain = float(random_cfg.get("max_gain", 1.0))
        seed = random_cfg.get("seed", None)
        sample_gen = generator
        if sample_gen is None and seed is not None:
            sample_gen = torch.Generator(device="cpu").manual_seed(int(seed))

        if distribution == "uniform":
            draws = _rand(base_shape, device=device, generator=sample_gen, dtype=dtype)
            gains = min_gain + (max_gain - min_gain) * draws
        else:
            log_min = math.log(min_gain)
            log_max = math.log(max_gain)
            mu = 0.5 * (log_min + log_max)
            sigma = max((log_max - log_min) / 4.0, 1e-6)
            z = _randn(base_shape, device=device, generator=sample_gen, dtype=dtype)
            gains = torch.exp(z * sigma + mu).clamp(min=min_gain, max=max_gain)

        if self.sample_h_per_batch:
            gains = gains.expand(bsz, -1)
        return gains

    def _sample_h(
        self,
        bsz: int,
        device: torch.device,
        generator: Optional[torch.Generator],
        dtype: torch.dtype,
        diagonal_gains: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fading == "identity":
            base = torch.zeros(self.n_rx, self.n_tx, device=device, dtype=dtype)
            diag_n = min(self.n_tx, self.n_rx)
            idx = torch.arange(diag_n, device=device)
            base[idx, idx] = 1.0
            return base.unsqueeze(0).expand(bsz, -1, -1)

        if self.fading == "diagonal":
            if diagonal_gains is None:
                gains = self.sample_diagonal_gains(
                    bsz,
                    device=device,
                    generator=generator,
                    dtype=dtype,
                )
            else:
                gains = torch.as_tensor(diagonal_gains, device=device, dtype=dtype)
                if gains.dim() == 1:
                    gains = gains.unsqueeze(0).expand(bsz, -1)
                if gains.shape != (bsz, self.n_tx):
                    raise ValueError(
                        f"Diagonal gains must have shape [{bsz}, {self.n_tx}] or [{self.n_tx}], got {tuple(gains.shape)}"
                    )

            h = torch.zeros(bsz, self.n_rx, self.n_tx, device=device, dtype=dtype)
            idx = torch.arange(self.n_tx, device=device)
            h[:, idx, idx] = gains
            return h

        # Rayleigh
        std = math.sqrt(1.0 / self.n_tx)
        if self.sample_h_per_batch:
            h = _randn((1, self.n_rx, self.n_tx), device=device, generator=generator, dtype=dtype) * std
            return h.expand(bsz, -1, -1)
        return _randn((bsz, self.n_rx, self.n_tx), device=device, generator=generator, dtype=dtype) * std

    def _normalize_signal(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = s.shape[0]
        s_flat = s.reshape(bsz, -1)
        if not self.normalize:
            p_signal = torch.mean(s_flat ** 2, dim=1, keepdim=True)
            return s, p_signal

        if self.normalization_mode == "sample":
            p_signal = torch.mean(s_flat ** 2, dim=1, keepdim=True)
            scale = torch.rsqrt(p_signal + 1e-9)
            s_norm = (s_flat * scale).reshape_as(s)
            return s_norm, torch.ones_like(p_signal)

        # batch mode
        p_signal_batch = torch.mean(s_flat ** 2)
        scale = torch.rsqrt(p_signal_batch + 1e-9)
        s_norm = (s_flat * scale).reshape_as(s)
        return s_norm, torch.ones((bsz, 1), device=s.device, dtype=s.dtype)

    def _mmse_use_inverse_path(self, h: torch.Tensor) -> bool:
        return h.device.type == "mps" and torch.is_grad_enabled()

    def _mmse_solve_with_jitter(self, a: torch.Tensor, rhs: torch.Tensor, i_eye: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.solve(a, rhs)
        except RuntimeError:
            # Avoid pinv/SVD fallback on MPS by re-trying with diagonal jitter.
            scale = max(float(a.detach().abs().amax().item()), 1.0)
            base_jitter = max(torch.finfo(a.dtype).eps * scale, 1e-6)
            for factor in (1.0, 1e2, 1e4):
                try:
                    return torch.linalg.solve(a + (base_jitter * factor) * i_eye, rhs)
                except RuntimeError:
                    continue
            raise

    def _mmse_inverse_with_jitter(self, a: torch.Tensor, rhs: torch.Tensor, i_eye: torch.Tensor) -> torch.Tensor:
        try:
            return torch.matmul(torch.linalg.inv(a), rhs)
        except RuntimeError:
            scale = max(float(a.detach().abs().amax().item()), 1.0)
            base_jitter = max(torch.finfo(a.dtype).eps * scale, 1e-6)
            for factor in (1.0, 1e2, 1e4):
                try:
                    return torch.matmul(torch.linalg.inv(a + (base_jitter * factor) * i_eye), rhs)
                except RuntimeError:
                    continue
            raise

    def _equalize(self, h: torch.Tensor, y: torch.Tensor, sigma2_vec: torch.Tensor) -> torch.Tensor:
        if self.equalizer == "zf":
            h_pinv = torch.linalg.pinv(h)
            return torch.matmul(h_pinv, y)

        # MMSE: (H^T H + (sigma^2 + eps) I)^-1 H^T Y
        ht = h.transpose(1, 2)
        hth = torch.matmul(ht, h).to(dtype=h.dtype)
        i_eye = torch.eye(self.n_tx, device=h.device, dtype=h.dtype).unsqueeze(0).expand(h.shape[0], -1, -1)
        a = hth + (sigma2_vec.view(-1, 1, 1) + self.mmse_eps) * i_eye
        rhs = torch.matmul(ht, y).to(dtype=a.dtype)
        if self._mmse_use_inverse_path(h):
            return self._mmse_inverse_with_jitter(a, rhs, i_eye)
        return self._mmse_solve_with_jitter(a, rhs, i_eye)

    def _compute_dtype(self, in_dtype: torch.dtype) -> torch.dtype:
        # MPS linear algebra kernels used by solve/pinv require float32.
        # We also keep float32 for mixed precision stability.
        if in_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return in_dtype

    def forward(self, s: torch.Tensor, selection_indices=None, generator: Optional[torch.Generator] = None, **kwargs):
        """
        Args:
            s: [B, n_tx, T] packed transmit symbols.
            h_override: optional pre-sampled H [B, n_rx, n_tx] to skip internal sampling.
        Returns:
            s_hat: [B, n_tx, T] equalized symbols.
            stats: dict
        """
        if s.dim() != 3:
            raise ValueError(f"MIMO channel expects [B, n_tx, T], got shape={tuple(s.shape)}")
        if s.shape[1] != self.n_tx:
            raise ValueError(f"Input n_tx mismatch: expected {self.n_tx}, got {s.shape[1]}")

        curr_snr = self._resolve_snr(s, generator)
        bsz, _, t = s.shape
        device = s.device
        in_dtype = s.dtype
        compute_dtype = self._compute_dtype(in_dtype)
        s = s.to(dtype=compute_dtype)
        dtype = s.dtype

        s_norm, _p_signal = self._normalize_signal(s)
        snr_linear = float("inf") if math.isinf(curr_snr) else (10 ** (curr_snr / 10.0))

        # Sample / override channel matrix H
        h_override = kwargs.get("h_override", None)
        if h_override is not None:
            h = h_override.to(device=device, dtype=dtype)
        else:
            diagonal_gains = kwargs.get("diagonal_gains", None)
            h = self._sample_h(bsz, device, generator, dtype, diagonal_gains=diagonal_gains)

        # Received signal (noiseless)
        y_signal = torch.matmul(h, s_norm)

        # Compute noise variance from *received* signal power so that
        # SNR_pre = E[||H S||^2] / E[||N||^2] = snr_linear (by construction).
        if math.isinf(snr_linear):
            sigma2_vec = torch.zeros((bsz,), device=device, dtype=dtype)
        else:
            p_received = torch.mean(
                y_signal.detach().reshape(bsz, -1) ** 2, dim=1, keepdim=True,
            )  # [B, 1]
            sigma2_vec = (p_received.view(-1) / snr_linear).to(dtype=dtype)

        # Additive noise
        noise = _randn((bsz, self.n_rx, t), device=device, generator=generator, dtype=dtype)
        noise = noise * torch.sqrt(sigma2_vec.view(-1, 1, 1))
        y = y_signal + noise

        s_hat = self._equalize(h, y, sigma2_vec)

        with torch.no_grad():
            p_sig_pre = torch.mean(y_signal ** 2)
            p_noi_pre = torch.mean(noise ** 2)
            snr_pre = 10.0 * torch.log10(p_sig_pre / (p_noi_pre + 1e-9))

            eq_err = s_hat - s_norm
            p_sig_post = torch.mean(s_norm ** 2)
            p_noi_post = torch.mean(eq_err ** 2)
            snr_post = 10.0 * torch.log10(p_sig_post / (p_noi_post + 1e-9))

        stats = {
            "channel_type": "mimo",
            "fading": self.fading,
            "mimo_n_tx": self.n_tx,
            "mimo_n_rx": self.n_rx,
            "mimo_fading": self.fading,
            "mimo_equalizer": self.equalizer,
            "mimo_mmse_eps": self.mmse_eps,
            "snr_db_target": curr_snr,
            "snr_db_measured_pre_eq": float(snr_pre.item()),
            "snr_db_measured_post_eq": float(snr_post.item()),
            "snr_db_measured": float(snr_post.item()),
            "signal_power_pre_eq": float(p_sig_pre.item()),
            "noise_power_pre_eq": float(p_noi_pre.item()),
            "signal_power_post_eq": float(p_sig_post.item()),
            "noise_power_post_eq": float(p_noi_post.item()),
            "mimo_sigma2": float(sigma2_vec.mean().item()),
        }
        if self.fading == "diagonal":
            diag_gains = torch.diagonal(h, dim1=1, dim2=2)
            stats["mimo_diagonal_gain_min"] = float(diag_gains.min().item())
            stats["mimo_diagonal_gain_mean"] = float(diag_gains.mean().item())
            stats["mimo_diagonal_gain_max"] = float(diag_gains.max().item())
            random_cfg = self._get_diagonal_random_cfg()
            if not bool(random_cfg.get("enabled", False)):
                stats["mimo_diagonal_gains"] = [float(v) for v in diag_gains[0].tolist()]
        return s_hat.to(dtype=in_dtype), stats
