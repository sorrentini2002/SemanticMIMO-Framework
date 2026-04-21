import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import VisionTransformer

from comm.comm_module_wrapper import CommModuleWrapper


class Store_Class_Token_Attn_Wrapper(nn.Module):
    """
    Wraps a timm attention module and stores CLS-row attention scores.
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape

        qkv = (
            self.attn.qkv(x)
            .reshape(bsz, n_tokens, 3, self.attn.num_heads, self.attn.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Shape [B, N]: mean over heads of CLS row attention.
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1)

        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(bsz, n_tokens, channels)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class Random_Token_Selection_Block_Wrapper(nn.Module):
    """
    Selects random patch tokens while always keeping CLS.

    It also stores the true importance scores (from CLS attention) of the
    selected tokens in `last_adc_scores` for semantic mode/power allocation.
    """

    def __init__(self, block: nn.Module, method_cfg: dict):
        super().__init__()

        self.block = block

        self.compression_enabled = method_cfg.get("compression_enabled", True)
        self.token_compression = method_cfg.get("token_compression", 1.0)
        self.eval_k = method_cfg.get("eval_k", None)

        # Interface variable used by CommModuleWrapper.
        self.last_adc_scores = None
        object.__setattr__(self, "_model_ref", None)

        self.n_new_tokens = 0

    def _random_select(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        num_patches = n_tokens - 1
        device = x.device

        if num_patches <= 0:
            self.n_new_tokens = 1
            self.last_adc_scores = torch.ones(
                (bsz, 1), dtype=x.dtype, device=device
            )
            return x

        if not self.training and self.eval_k is not None:
            n_alpha = max(1, min(int(self.eval_k), num_patches))
        else:
            n_alpha = max(1, min(int(self.token_compression * num_patches), num_patches))

        self.n_new_tokens = 1 + n_alpha

        patch_indices = []
        for _ in range(bsz):
            # Sample patch indices in [1, N-1], then sort for stable ordering.
            sampled = torch.randperm(num_patches, device=device)[:n_alpha] + 1
            sampled, _ = torch.sort(sampled)
            patch_indices.append(sampled)
        patch_indices = torch.stack(patch_indices, dim=0)  # [B, n_alpha]

        cls_indices = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        indices_sel = torch.cat([cls_indices, patch_indices], dim=1)  # [B, 1+n_alpha]

        gather_idx = indices_sel.unsqueeze(-1).expand(-1, -1, dim)
        tokens_sel = torch.gather(x, dim=1, index=gather_idx)

        # Importance scores are still computed from CLS attention, even if
        # token selection itself is random.
        cls_attn = self.block.attn.class_token_attention  # [B, N]
        if cls_attn is None:
            selected_patch_scores = torch.ones(
                (bsz, n_alpha), dtype=x.dtype, device=device
            )
        else:
            patch_scores = cls_attn[:, 1:]  # [B, N-1]
            selected_patch_scores = torch.gather(
                patch_scores,
                dim=1,
                index=(patch_indices - 1).clamp(min=0, max=max(0, num_patches - 1)),
            ).to(dtype=x.dtype, device=device)

        cls_dummy = torch.ones((bsz, 1), dtype=x.dtype, device=device)
        self.last_adc_scores = torch.cat([cls_dummy, selected_patch_scores], dim=1)
        
        # --- Store indices for Server Spatial Reconstruction ---
        self.last_indices_sel = indices_sel
        self.last_original_N = x.shape[1]

        return tokens_sel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        clean_val = False
        if not self.training and self._model_ref is not None:
            clean_val = getattr(self._model_ref, "clean_validation", False)

        if self.compression_enabled and not clean_val:
            x = self._random_select(x)

        return x

    def compress_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(labels, num_classes=num_classes).float()


class model(nn.Module):
    """
    Split-learning model with random token selection at split point.
    """

    def __init__(
        self,
        model: VisionTransformer,
        channel,
        split_index,
        method_cfg: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.method_cfg = method_cfg

        compression_enabled = method_cfg.get("compression_enabled", True)
        desired_compression = method_cfg.get("desired_compression", None)
        token_compression = method_cfg.get("token_compression", 1.0)
        self.channel_eval_only = method_cfg.get("channel_eval_only", False)
        self.semantic_waterfilling = method_cfg.get("semantic_waterfilling", True)

        if not compression_enabled:
            self.compression_ratio = 1.0
        else:
            if desired_compression is not None:
                assert token_compression is None or token_compression == 1.0, (
                    "When desired_compression is set, token_compression should be None"
                )
                self.compression_ratio = desired_compression
                self.method_cfg["token_compression"] = desired_compression
            else:
                if token_compression is None:
                    token_compression = 1.0
                self.compression_ratio = token_compression
                self.method_cfg["token_compression"] = token_compression

        self.compressor_module = None
        self.clean_validation = False

        self.model = self.build_model(model, channel, split_index, self.method_cfg)
        self.channel = channel
        self.communication = 0
        self.name = "RandomMethod"

    def build_model(
        self,
        model: VisionTransformer,
        channel,
        split_index: int,
        method_cfg: dict,
    ):
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(
            model.blocks[split_index - 1].attn
        )

        model.blocks[split_index - 1] = Random_Token_Selection_Block_Wrapper(
            block=model.blocks[split_index - 1],
            method_cfg=method_cfg,
        )
        self.compressor_module = model.blocks[split_index - 1]

        object.__setattr__(self.compressor_module, "_model_ref", weakref.proxy(self))

        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        if isinstance(channel, CommModuleWrapper):
            channel.set_score_source(self.compressor_module)
            if hasattr(channel, "set_channel_eval_only"):
                channel.set_channel_eval_only(self.channel_eval_only)
            if hasattr(channel, "set_semantic_waterfilling"):
                channel.set_semantic_waterfilling(self.semantic_waterfilling)

            if not method_cfg.get("compression_enabled", True) and hasattr(channel, "comm"):
                channel.comm.use_bottleneck = False

        return model

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
        return self.model.forward(x)