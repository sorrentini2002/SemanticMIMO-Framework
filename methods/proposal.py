# ============================================================
# methods/proposal.py
# ============================================================
# Split-learning proposal: selects tokens
# at the split point, then passes features through the channel.
#
# Supported channel types (passed via Hydra cfg.communication):
#   - Gaussian_Noise_Analogic_Channel  (legacy AWGN scalar)
#   - MyMIMOChannel                    (simple MIMO wrapper)
#   - CommModuleWrapper                (full CommModule pipeline:
#                                       bottleneck + MIMO + PA)
# ============================================================

from timm.models import VisionTransformer
import torch.nn as nn
import torch
import torch.nn.functional as F

# --- Import the advanced MIMO channel wrapper ---
# CommModuleWrapper adapts CommModule to the nn.Sequential API:
# it swallows the (tensor, info_dict) return and yields only tensor.
from comm.comm_module_wrapper import CommModuleWrapper


# Block used by our proposal to select tokens
class Token_Selection_Block_Wrapper(nn.Module):

    def __init__(self, block, token_compression, pooling='attention'):
        super().__init__()

        assert pooling in ['attention', 'average', 'cls'], (f"Pooling must be either in "
                                                            f"{['attention', 'average', 'cls']}"
                                                            f" but {pooling} was given.")

        self.block = block

        # Store compression rates 
        self.token_compression = token_compression

        self.n_new_tokens = 0

        self.pooling = pooling
        self.last_adc_scores = None

        # Reference to parent model to read clean_validation flag
        self._model_ref = None

        # Diagnostic stats for channel integrity monitoring
        self.diagnostic_stats = {
            "payload_x_norm": [], 
            "payload_out_norm": [], 
            "payload_diff_norm": []
        }

    def select_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Get input dimensions 
        n_batches, n_tokens, hidden_dim = x.size()

        # Compute the new number of tokens
        self.n_new_tokens = max(1, int(self.token_compression * n_tokens))
        
        # Store device
        device = x.device

        # Get the token scores based on the pooling strategy
        if self.pooling == 'attention':
            # class_token_attention has shape [n_batches, n_tokens]
            # We take scores for patch tokens (excluding CLS at index 0)
            scores = self.block.attn.class_token_attention[:, 1:]
        elif self.pooling == 'average':
            scores = x[:, 1:].mean(dim=-1)
        else: # 'cls'
            scores = x[:, 1:].abs().mean(dim=-1)

        # Select top-k tokens for each image in the batch
        # scores: [B, N-1]
        top_k = torch.topk(scores, k=self.n_new_tokens - 1, dim=1, largest=True, sorted=False)
        top_k_indices = top_k.indices # [B, n_new_tokens - 1]
        
        # Prepend CLS token index (0) to each selection
        cls_idx = torch.zeros((n_batches, 1), dtype=torch.long, device=device)
        full_indices = torch.cat([cls_idx, top_k_indices + 1], dim=1) # [B, n_new_tokens]

        # Gather selected tokens: [B, n_new_tokens, hidden_dim]
        batch_indices = torch.arange(n_batches, device=device).view(-1, 1)
        output = x[batch_indices, full_indices]

        # Keep the scores of the selected tokens for MIMO stream/power allocation
        # We need scores for all selected tokens including CLS (dummy 1.0)
        cls_scores = torch.ones((n_batches, 1), dtype=top_k.values.dtype, device=device)
        self.last_adc_scores = torch.cat([cls_scores, top_k.values], dim=1) # [B, n_new_tokens]
        
        # Store indices for Server Spatial Reconstruction
        self.last_indices_sel = full_indices

        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Normal block forward 
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        # Always apply token selection
        x = self.select_tokens(x)

        return x

    # Function to handle the conversion of labels to one-hot (no merging)
    def compress_labels(self, labels, num_classes) -> torch.Tensor:
        return F.one_hot(labels, num_classes=num_classes).float()


# An attention class that stores class token attention scores
class Store_Class_Token_Attn_Wrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normal attention behaviour
        B, N, C = x.shape
        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Store class_token attention 
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1).detach()

        # Normal attention behaviour 
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class model(nn.Module):

    def __init__(self,
                 model: VisionTransformer,
                 channel,
                 split_index,
                 token_compression=None,
                 pooling='attention',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if token_compression is not None:
            self.compression_ratio = token_compression
        else:
            raise ValueError('Set token_compression')

        self.compressor_module = None

        # Store compression


        # Build model 
        self.model = self.build_model(model, channel, split_index, self.compression_ratio, pooling)

        # Store channel 
        self.channel = channel

        # Store the split index so we can bypass the channel during eval
        self.split_index = split_index

        # Clean validation flag: when True, skip ADC + channel during eval.
        # Default is False (noisy validation = ADC + channel active).
        self.clean_validation = False

        # Wire the flag reference to the compressor block
        if self.compressor_module is not None:
            self.compressor_module._model_ref = [self]

        # Variable to store communication 
        self.communication = 0

        # Store name 
        self.name = "Proposal"

    # --------------------------------------------------------
    # build_model: assembles the complete split-learning model
    # --------------------------------------------------------
    def build_model(self,
                    model,
                    channel,
                    split_index,
                    token_compression,
                    pooling):

        # --- Wrap the attention module at split_index-1 ---
        # Stores class-token attention scores used for token selection.
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(
            model.blocks[split_index - 1].attn
        )

        # --- Wrap the full block with token selection ---
        model.blocks[split_index - 1] = Token_Selection_Block_Wrapper(
            model.blocks[split_index - 1],
            token_compression,
            pooling=pooling,
        )
        self.compressor_module = model.blocks[split_index - 1]

        # --- Split backbone into client and server sections ---
        blocks_before = model.blocks[:split_index]   # runs on client
        blocks_after  = model.blocks[split_index:]   # runs on server

        # --- Insert the channel between client and server ---
        # CommModuleWrapper is already nn.Sequential-compatible
        # (it returns a single tensor, not a tuple).
        # Gaussian_Noise_Analogic_Channel is also compatible out-of-the-box.
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        # --- Wire attention scores → CommModuleWrapper ---
        # If the channel is a CommModuleWrapper, give it a reference to the
        # compressor block so it can read class_token_attention on every
        # forward pass and pass it to CommModule as selection_scores.
        # This activates power / stream / mode allocation without changing
        # the nn.Sequential API or the model's forward signature.
        if isinstance(channel, CommModuleWrapper):
            channel.set_score_source(self.compressor_module)

        return model

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
            return self.model.forward(x)

        # --- Eval mode ---
        if self.clean_validation:
            # Clean validation: bypass both ADC (handled in the block)
            # and the channel layer entirely.
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
            x = self.model.norm_pre(x)

            blocks = self.model.blocks
            for i in range(self.split_index):
                x = blocks[i](x)
            # Skip blocks[split_index] which is the channel
            for i in range(self.split_index + 1, len(blocks)):
                x = blocks[i](x)

            x = self.model.norm(x)
            x = self.model.forward_head(x)
            return x
        else:
            # Noisy validation: normal forward (ADC + channel active)
            return self.model.forward(x)
