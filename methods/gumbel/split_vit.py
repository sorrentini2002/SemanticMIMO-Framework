import torch
import torch.nn as nn
import timm
import logging
from ..comm.comm_module import CommModule
from ..selection import apply_selection
from .simplicial_attention import SimplicialAttention

logger = logging.getLogger(__name__)

class SplitViT(nn.Module):
    """
    Split ViT model that wraps a timm Vision Transformer.
    It splits the transformer blocks into Encoder and Decoder parts,
    and inserts a Communication Module in between.
    """
    def __init__(self, config, num_classes):
        super().__init__()
        
        model_name = config.get("model", {}).get("name", "vit_base_patch16_224")
        pretrained = config.get("model", {}).get("pretrained", True)
        patch_size = config.get("model", {}).get("patch_size", None)
        
        logger.info(f"Creating SplitViT base: {model_name}")
        
        # Prepare kwargs
        image_size = config.get("data", {}).get("image_size", 224)
        model_kwargs = {"num_classes": num_classes, "img_size": image_size}
        if patch_size is not None:
            logger.info(f"Overriding patch_size to {patch_size}. Note: This may trigger patch embedding re-initialization.")
            model_kwargs["patch_size"] = patch_size
            
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        
        # Log token count
        if hasattr(self.vit, 'patch_embed'):
            num_patches = self.vit.patch_embed.num_patches
            logger.info(f"Model initialized with {num_patches} patches (Total tokens: {num_patches + 1} with CLS).")
        else:
            logger.warning("Could not determine number of patches (no patch_embed attribute).")
        
        # Extract components from timm ViT
        if not hasattr(self.vit, 'blocks'):
            raise ValueError(f"Model {model_name} does not seem to have 'blocks' attribute. SplitViT currently supports standard ViT only.")
            
        self.split_block = config.get("model", {}).get("split_block", -1)
        
        # --- Simplicial Attention Integration ---
        simplicial_cfg = config.get("model", {}).get("simplicial", {})
        self.simplicial_indices = []

        if simplicial_cfg.get("enabled", False):
            # Parse block config
            blocks_cfg = simplicial_cfg.get("blocks", {})
            # If "blocks" is missing but enabled is True, default to "split_only" for safety/convenience
            mode = blocks_cfg.get("mode", "split_only")
            
            num_blocks = len(self.vit.blocks)
            target_indices = []

            # Resolve indices based on mode
            if mode == "split_only":
                # Default behavior: last encoder block (before split)
                # If split_block is -1, it means all blocks are encoder, so -1 maps to the last one.
                idx = self.split_block - 1 if self.split_block >= 0 else num_blocks - 1
                if idx < 0: idx += num_blocks # Handle wrapping if needed, though split_block=0 -> idx=-1 -> last block is weird.
                # If split_block=0, strictly speaking encoder is empty. 
                # But let's assume valid split_block >= 1 or -1.
                target_indices = [idx]
            
            elif mode == "list":
                target_indices = blocks_cfg.get("indices", [])
                
            elif mode in ["all", "every_n"]:
                start = blocks_cfg.get("start", 0)
                end = blocks_cfg.get("end", num_blocks)
                if end is None: end = num_blocks
                
                stride = blocks_cfg.get("every_n", 1) if mode == "every_n" else 1
                target_indices = list(range(start, end, stride))
                
            else:
                raise ValueError(f"Unknown simplicial block mode: {mode}")

            # Validate and Swap
            unique_indices = sorted(list(set(target_indices)))
            
            for idx in unique_indices:
                # Handle negative indices
                if idx < 0: idx += num_blocks
                
                if 0 <= idx < num_blocks:
                    target_block = self.vit.blocks[idx]
                    
                    if not hasattr(target_block, 'attn'):
                        logger.warning(f"Block {idx} does not have 'attn'. Skipping.")
                        continue
                        
                    old_attn = target_block.attn
                    
                    # Prevent double wrapping if swapped already (unlikely with set/sorted, but safe)
                    if isinstance(old_attn, SimplicialAttention):
                        continue

                    # Create new SimplicialAttention
                    new_attn = SimplicialAttention(
                        base_attn=old_attn,
                        variant=simplicial_cfg.get("variant", "triangle_marginal"),
                        beta_init=simplicial_cfg.get("beta_init", 1.0),
                        gate_init=simplicial_cfg.get("gate_init", -6.0),
                        branch_norm=simplicial_cfg.get("branch_norm", "layernorm"),
                        allow_pure=simplicial_cfg.get("allow_pure", True),
                        beta_trainable=simplicial_cfg.get("beta_trainable", True),
                        gate_trainable=simplicial_cfg.get("gate_trainable", True),
                        tri_scale=simplicial_cfg.get("tri_scale", "none"),
                        gamma_init=simplicial_cfg.get("gamma_init", 1.0),
                        gamma_trainable=simplicial_cfg.get("gamma_trainable", True),
                        pair_attention_kwargs=simplicial_cfg.get("pair_attention", {})
                    )
                    
                    # Handle force_pure override
                    if simplicial_cfg.get("force_pure", False):
                        # Apply force pure logic once
                        with torch.no_grad():
                            new_attn.beta.fill_(0.0)
                            new_attn.gate_param.fill_(3.0) 
                        new_attn.beta.requires_grad = False

                    target_block.attn = new_attn
                    self.simplicial_indices.append(idx)
                    logger.info(f"Swapped block {idx} attention with SimplicialAttention.")
                else:
                    raise ValueError(f"Invalid block index {idx} for Simplicial Attention swap. Num blocks: {num_blocks}")
            
            logger.info(f"Simplicial Attention enabled on {len(self.simplicial_indices)} blocks: {self.simplicial_indices}")
        
        # Continuation of original logic...
        num_blocks = len(self.vit.blocks)
        
        # If split_block is -1 or >= num_blocks, effectively everything is Encoder.
        if self.split_block < 0 or self.split_block >= num_blocks:
            self.encoder_blocks = self.vit.blocks
            self.decoder_blocks = nn.Sequential()
            logger.info(f"SplitViT: No split (all blocks in encoder). Num blocks: {num_blocks}")
            self.last_encoder_block = self.vit.blocks[-1]
        else:
            self.encoder_blocks = self.vit.blocks[:self.split_block]
            self.decoder_blocks = self.vit.blocks[self.split_block:]
            logger.info(f"SplitViT: Split at block {self.split_block}. Encoder: {len(self.encoder_blocks)}, Decoder: {len(self.decoder_blocks)}")
            # The last block of the encoder is the one before split
            self.last_encoder_block = self.encoder_blocks[-1]
            
        # Communication Module
        embed_dim = self.vit.embed_dim
        self.comm = CommModule(embed_dim, config)
        
        # Token Selection Config
        self.selection_config = config.get("selection", {})
        self.selection_enabled = self.selection_config.get("enabled", False)
        self.selection_method = self.selection_config.get("method", "topk_cls")
        
        # Pure baseline mode: when both channel and selection are disabled, 
        # avoid registering hooks and disabling fused attention to match standard ViT behavior
        comm_enabled = config.get("comm", {}).get("enabled", False)
        vis_enabled = config.get("vis", {}).get("enabled", False)
        self.pure_baseline_mode = not comm_enabled and not self.selection_enabled and not vis_enabled
        
        self.captured_attn = None
        self.hook_handle = None
        
        if not self.pure_baseline_mode:
            logger.info("Registering attention hook for potential token selection.")
            all_blocks = list(self.encoder_blocks) + list(self.decoder_blocks)
            for blk in all_blocks:
                if hasattr(blk, 'attn'):
                    if hasattr(blk.attn, 'fused_attn'):
                        blk.attn.fused_attn = False
            
            if hasattr(self.last_encoder_block, 'attn') and hasattr(self.last_encoder_block.attn, 'attn_drop'):
                if isinstance(self.last_encoder_block.attn, SimplicialAttention):
                    # SimplicialAttention already stores `last_attn` internally.
                    logger.info("Last encoder block uses SimplicialAttention. Skipping timm monkey patch and using internal attention context.")
                else:
                    if hasattr(self.last_encoder_block.attn, 'fused_attn'):
                        self.last_encoder_block.attn.fused_attn = False

                    required_attrs = [
                        "qkv", "num_heads", "head_dim", "q_norm", "k_norm",
                        "scale", "attn_drop", "attn_dim", "norm", "proj", "proj_drop"
                    ]
                    if all(hasattr(self.last_encoder_block.attn, name) for name in required_attrs):
                        # Monkey patch forward to explicitly save the attention weights
                        import types
                        def patched_forward(self_attn, x, attn_mask=None):
                            B, N, C = x.shape
                            qkv = self_attn.qkv(x).reshape(B, N, 3, self_attn.num_heads, self_attn.head_dim).permute(2, 0, 3, 1, 4)
                            q, k, v = qkv.unbind(0)
                            q, k = self_attn.q_norm(q), self_attn.k_norm(k)

                            if getattr(self_attn, 'fused_attn', False):
                                import torch.nn.functional as F
                                x_out = F.scaled_dot_product_attention(
                                    q, k, v,
                                    attn_mask=attn_mask,
                                    dropout_p=self_attn.attn_drop.p if self_attn.training else 0.,
                                )
                                self_attn.last_attn = None
                            else:
                                q = q * self_attn.scale
                                attn = q @ k.transpose(-2, -1)
                                # We skip maybe_add_mask to keep it simple, since baseline ViT doesn't use it for CIFAR
                                attn = attn.softmax(dim=-1)
                                self_attn.last_attn = attn # Expose here!
                                attn = self_attn.attn_drop(attn)
                                x_out = attn @ v

                            x_out = x_out.transpose(1, 2).reshape(B, N, self_attn.attn_dim)
                            x_out = self_attn.norm(x_out)
                            x_out = self_attn.proj(x_out)
                            x_out = self_attn.proj_drop(x_out)
                            return x_out

                        self.last_encoder_block.attn.forward = types.MethodType(patched_forward, self.last_encoder_block.attn)
                    else:
                        logger.warning(
                            "Last encoder attention module does not expose the expected timm attributes for monkey patching. "
                            "Attention-based selection may be unavailable for this model."
                        )
            else:
                logger.warning("Could not find attn.attn_drop in last encoder block. Token selection might fail if it relies on attention weights.")
        elif self.pure_baseline_mode:
            logger.info("Pure baseline mode: channel and selection disabled. Skipping hook registration for standard ViT behavior.")
                
    def _get_simplicial_context(self):
        """Helper to extract simplicial internals if available."""
        attn_val = getattr(self.last_encoder_block.attn, 'last_attn', None)
        ctx = {"attn": attn_val}
        
        # Check if last encoder block has SimplicialAttention
        if hasattr(self.last_encoder_block, 'attn') and isinstance(self.last_encoder_block.attn, SimplicialAttention):
            if hasattr(self.last_encoder_block.attn, 'last_m'):
                ctx['m'] = self.last_encoder_block.attn.last_m
            if hasattr(self.last_encoder_block.attn, 'last_u'):
                ctx['u'] = self.last_encoder_block.attn.last_u
            # SimplicialAttention also stores last_attn, which might be cleaner than the hook
            # logic, but we keep the hook for consistency with standard ViT.
            # However, for simplicial strategies, we might prefer the internal one.
            if hasattr(self.last_encoder_block.attn, 'last_attn'):
                ctx['attn'] = self.last_encoder_block.attn.last_attn
                
        return ctx
        
    def forward(self, x, selection_generator=None, step=None):
        """
        Args:
            x: [B, 3, H, W]
            selection_generator: torch.Generator for random sampling.
            step: int, global training step (for scheduling).
        """
        # 1. Patch Embed + Pos Embed
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        
        # 2. Encoder Blocks
        x = self.encoder_blocks(x)
        
        mode = "train" if self.training else "eval"
        
        # Check enabled flag
        if getattr(self, "pure_baseline_mode", False) or not self.selection_enabled:
             # Bypass selection (keep all)
             res = {"tokens_sel": x, "indices_sel": None, "n_alpha": x.shape[1]-1, "strategy": "none"}
        else:
            full_x = x # cache for coverage reg
            current_attn = self._get_simplicial_context().get("attn", None)
            
            res = apply_selection(
                x, 
                current_attn,
                self.selection_config, 
                mode=mode, 
                generator=selection_generator,
                step=step,
                simplicial_context=self._get_simplicial_context()
            )
        
        x = res["tokens_sel"]
        self.last_selection_indices = res["indices_sel"]
        n_alpha_val = res["n_alpha"]
        strategy_used = res["strategy"]
        
        # Regularization logic for Gumbel
        self.last_cov_reg = torch.tensor(0.0, device=x.device)
        self.last_ent_reg = torch.tensor(0.0, device=x.device)
        
        if self.training and self.selection_enabled and strategy_used in ["gumbel_topk", "topk_cls_attention"]:
            gumbel_cfg = self.selection_config.get("train", {}).get("gumbel", {})
            if gumbel_cfg.get("enabled", False) or strategy_used == "gumbel_topk":
                scores = res.get("scores") # [B, N-1]
                if scores is not None:
                    # Soft selection probabilities using current tau or just softmax/sigmoid
                    tau = res.get("gs_tau", 1.0)
                    p = torch.sigmoid(scores / tau)
                    
                    # Entropy Regulation
                    ent_cfg = gumbel_cfg.get("entropy_reg", {})
                    if ent_cfg.get("enabled", False) and ent_cfg.get("weight", 0.0) > 0:
                        p_mean = p.mean(dim=0) # [N-1]
                        eps = 1e-8
                        # We want to maximize entropy, so we return negative entropy to be added to loss as a penalty
                        self.last_ent_reg = -torch.sum(p_mean * torch.log(p_mean + eps))
                        
                    # Coverage Regulation
                    cov_cfg = gumbel_cfg.get("cov_reg", {})
                    if cov_cfg.get("enabled", False) and cov_cfg.get("weight", 0.0) > 0:
                        margin = cov_cfg.get("margin", 0.3)
                        max_tokens = cov_cfg.get("max_tokens", 64)
                        
                        patch_embeds = full_x[:, 1:, :] # [B, N-1, D]
                        B, N_minus_1, D = patch_embeds.shape
                        
                        # Subsample tokens to save memory if N_minus_1 > max_tokens
                        if N_minus_1 > max_tokens:
                            # random subsampling
                            idx = torch.randperm(N_minus_1, device=x.device)[:max_tokens]
                            patch_embeds = patch_embeds[:, idx, :]
                            p_sub = p[:, idx]
                        else:
                            p_sub = p

                        # Normalize embeddings for cosine similarity
                        patch_embeds_norm = torch.nn.functional.normalize(patch_embeds, p=2, dim=-1) # [B, S, D]
                        
                        # Compute similarity matrix [B, S, S]
                        sim_matrix = torch.bmm(patch_embeds_norm, patch_embeds_norm.transpose(1, 2))
                        
                        # Mask out diagonal (self-similarity)
                        eye = torch.eye(sim_matrix.shape[-1], device=sim_matrix.device).unsqueeze(0).expand(B, -1, -1)
                        sim_matrix = sim_matrix * (1.0 - eye)
                        
                        # Apply margin max(0, cos - m)
                        penalty = torch.clamp(sim_matrix - margin, min=0.0)
                        
                        # Weight by soft-selection probs: w_i * w_j * penalty
                        w = p_sub / (p_sub.sum(dim=-1, keepdim=True) + 1e-8) # normalized weights
                        w_matrix = torch.bmm(w.unsqueeze(-1), w.unsqueeze(1)) # [B, S, S]
                        
                        cov_loss_batch = torch.sum(w_matrix * penalty, dim=(1, 2))
                        self.last_cov_reg = cov_loss_batch.mean()

        # Debug Dump: Split Point Info
        self._dump_split_info(x, self.last_selection_indices)

        # 4. Communication
        x, stats = self.comm(
            x,
            selection_indices=self.last_selection_indices,
            selection_scores=res.get("scores", None),
            generator=selection_generator,
        )
        
        # Augment stats
        stats['n_tokens_full'] = res.get("n_tokens_full", self.vit.patch_embed.num_patches + 1)
        stats['n_tokens_effective'] = x.shape[1]
        stats['tokens_total_pre_selection'] = stats['n_tokens_full']
        if 'tokens_selected' not in stats:
            stats['tokens_selected'] = max(0, stats['n_tokens_full'] - 1) if n_alpha_val is None else n_alpha_val
            
        stats['n_alpha'] = n_alpha_val
        stats['selection_strategy'] = strategy_used
        stats['selection_indices'] = self.last_selection_indices
        stats['selection_scores'] = res.get("scores", None)
        stats['gs_tau'] = res.get("gs_tau", None)

        # Assertion to guarantee selection count correctness (pre-channel)
        # We expect 1 + n_alpha (CLS + patches)
        if self.selection_enabled and n_alpha_val is not None:
             expected_tokens = 1 + n_alpha_val
             assert x.shape[1] == expected_tokens, f"Selection Audit Failed: Expected {expected_tokens} tokens (1 CLS + {n_alpha_val} patches), got {x.shape[1]}."

        # Audit Logging
        stats['diversify_lambda'] = self.selection_config.get("diversify", {}).get("lambda", 0.0)
        stats['keep_cls_neighbors'] = self.selection_config.get("keep_cls_neighbors", 0)
        
        self.last_stats = stats
        
        # 5. Decoder Blocks
        x = self.decoder_blocks(x)
        
        # 6. Head
        x = self.vit.norm(x)
        x = self.vit.forward_head(x) 
        
        return x
        
    def get_comm_stats(self):
        stats = getattr(self, 'last_stats', {}).copy()
        
        # Aggregate simplicial diagnostics
        simplicial_blocks = []
        # Use stored indices to find blocks
        if hasattr(self, 'simplicial_indices'):
            for idx in self.simplicial_indices:
                block = self.vit.blocks[idx]
                if hasattr(block, 'attn') and isinstance(block.attn, SimplicialAttention):
                    simplicial_blocks.append(block.attn.get_diagnostics())
        
        count = len(simplicial_blocks)
        stats['simplicial_blocks_count'] = count
        
        if count > 0:
            # Keys to aggregate
            keys = [
                "simplicial_beta_mean", 
                "simplicial_gate_mean", 
                "simplicial_gamma_mean", 
                "simplicial_branch_ratio_mean",
                "simplicial_std_norm_mean",
                "simplicial_tri_norm_mean"
            ]
            
            # Helper for aggregation
            for k in keys:
                values = [d.get(k, 0.0) for d in simplicial_blocks]
                stats[f"{k}_avg"] = sum(values) / count
                
                # Optional: min/max for gate and branch ratio
                if k in ["simplicial_gate_mean", "simplicial_branch_ratio_mean"]:
                    stats[f"{k}_min"] = min(values)
                    stats[f"{k}_max"] = max(values)
                    
            # Also keep the single-scalar version for backward compat if count=1?
            # Or just rely on _avg. The request said:
            # "simplicial_gate_mean_avg" etc.
            # It didn't explicitly ask to remove the old ones, but they were implicitly per-block.
            # If we have multiple blocks, "simplicial_gate_mean" is ambiguous.
            # We will NOT emit the bare keys to avoid confusion, only the aggregated ones.
        
        return stats

    def override_channel(self, config_snippet):
        """
        Overrides the channel configuration in-place.
        Delegates to CommModule.reconfigure.
        """
        if hasattr(self, 'comm'):
            self.comm.reconfigure(config_snippet)

    def _dump_split_info(self, x, indices):
        """
        Dumps debug info about the split point tensor if enabled.
        Run once per process/run usually.
        """
        dump_enabled = False
        dump_path = None
        
        if hasattr(self, 'comm') and hasattr(self.comm, 'channel') and self.comm.channel is not None:
            if getattr(self.comm.channel, 'debug_dump', None):
                dump_enabled = True
                dump_path = self.comm.channel.debug_dump
        
        # Avoid spamming
        if hasattr(self, '_split_dumped') and self._split_dumped:
            return
            
        if dump_enabled:
            import json
            import os
            
            # Compute a lightweight hash/stats
            mean_val = x.mean().item()
            std_val = x.std().item()
            
            info = {
                "type": "SplitPoint",
                "split_block_index": self.split_block,
                "tensor_shape": list(x.shape), # [B, N, D]
                "tensor_stats": {
                    "mean": mean_val, 
                    "std": std_val
                },
                "selection": {
                    "enabled": self.selection_enabled,
                    "strategy": self.selection_method,
                    "indices_count": indices.shape[1] if indices is not None else "All"
                },
                "packing_order": "Token-Major (Batch, Tokens, Dim) -> Flatten",
                "notes": "CLS token is at index 0 if preserved."
            }
            
            print("\\n--- [Split-Point Debug Dump] ---")
            print(json.dumps(info, indent=2))
            print("--------------------------------\\n")
            
            if isinstance(dump_path, (str, os.PathLike)):
                 try:
                    out_file = os.path.join(dump_path, "debug_split.json")
                    with open(out_file, 'w') as f:
                        json.dump(info, f, indent=2)
                    print(f"Split debug dump saved to {out_file}")
                 except Exception as e:
                    print(f"Failed to save split debug dump: {e}")
            
            self._split_dumped = True
