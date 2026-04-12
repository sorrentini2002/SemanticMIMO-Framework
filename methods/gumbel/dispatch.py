
import torch
import logging
from .topk_cls import select_topk_by_cls_attn
from .random import select_random
from .uniform import select_uniform_stride, select_uniform_grid
from .simplicial import select_simplicial

logger = logging.getLogger(__name__)

from .gumbel import sample_gumbel_topk, compute_gumbel_mc_scores
from .schedules import compute_tau
from .core import select_by_scores

_gs_logged = False

def apply_selection(tokens, attn, selection_cfg, *, mode="train", generator=None, step=None, simplicial_context=None):
    global _gs_logged
    """
    Apply token selection based on strategy and budget configuration.
    
    Args:
        tokens: [B, N, D]
        attn: [B, H, N, N] or [B, N, N]
        selection_cfg: dict containing strategy, unit, train/eval params.
        mode: "train" or "eval"
        generator: torch.Generator for random sampling.
        step: int or None, global step for annealing.
        
    Returns:
        dict with:
            tokens_sel: [B, K, D]
            indices_sel: [B, K]
            n_alpha: int (used)
            strategy: str (used)
    """
    # 1. Resolve Mode-Specific Configuration
    mode_cfg = selection_cfg.get(mode, {})
    
    # Strategy Resolution:
    # 1. Check mode-specific override (e.g. selection.eval.strategy)
    # 2. Fallback to global selection.strategy
    # 3. Default to "none"
    strategy = mode_cfg.get("strategy", selection_cfg.get("strategy", "none"))
    
    if strategy == "none":
        B, N, D = tokens.shape
        indices = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        return {
            "tokens_sel": tokens,
            "indices_sel": indices,
            "n_alpha": N - 1,
            "strategy": "none"
        }
        
    # 2. Resolve Budget (n_alpha)
    B, N, D = tokens.shape
    num_patches = N - 1
    
    # Unit: global setting usually, but allowing override could be useful. 
    # Sticking to global for unit to avoid confusion unless override exists.
    unit = selection_cfg.get("unit", "k")
    
    budget_mode = mode_cfg.get("mode", "fixed") # "fixed" or "sampled"
    
    if mode == "train" and budget_mode == "sampled":
        # Sample n_alpha or alpha (Training robustness)
        if unit == "alpha":
            low, high = mode_cfg.get("alpha_range", [0.1, 1.0])
            alpha = low + (high - low) * torch.rand(1, generator=generator).item()
            n_alpha = round(alpha * num_patches)
        else: # unit == "k"
            low, high = mode_cfg.get("k_range", [0, num_patches])
            n_alpha = torch.randint(low, high + 1, (1,), generator=generator).item()
    else:
        # Fixed value (Evaluation or Fixed Training)
        # Check mode-specific k/alpha first, then fallback to global (if any), then default.
        if unit == "alpha":
            alpha = mode_cfg.get("alpha", selection_cfg.get("alpha", 1.0))
            n_alpha = round(alpha * num_patches)
        else:
            k_val = mode_cfg.get("k", selection_cfg.get("k", num_patches))
            n_alpha = k_val
            
    # Clamp
    n_alpha = max(0, min(n_alpha, num_patches))
    
    # 3. Determinism Control (Eval)
    # If mode is eval, we check 'sample' flag.
    # If sample=False (default), we force deterministic behavior where applicable.
    # Note: 'random' strategy is inherently random, but 'topk' with noise or 'gumbel' might need control.
    if mode == "eval":
        eval_sample = mode_cfg.get("sample", False)
        if not eval_sample:
            # Force deterministic behavior if possible
            # For Gumbel, this might mean hard=True and noise=False?
            # Or just rely on the seed (generator) being fixed by the engine?
            # The user request says: "selection should be deterministic by default ... unless eval.sample=true"
            # And "Confirm random selection in eval is controlled by seed".
            # So if sample=False, maybe we shouldn't use Gumbel noise?
            pass

    # 4. Dispatch
    scores = None
    gs_tau = None
    
    # --- Keep CLS Neighbors Logic ---
    # Allow override in mode_cfg
    keep_cls_neighbors = mode_cfg.get("keep_cls_neighbors", selection_cfg.get("keep_cls_neighbors", 0))
    pre_selected_indices = None
    
    if keep_cls_neighbors > 0:
        if attn is not None:
             # Compute s_pair (CLS attn)
            if attn.dim() == 4:
                attn_mean = attn.mean(dim=1)
            else:
                attn_mean = attn
            # CLS scores: [B, N] row 0
            cls_scores = attn_mean[:, 0, :]
            # Exclude CLS from ranking (indices 1..N-1)
            patch_scores = cls_scores[:, 1:]
            
            # Select top m
            m = min(keep_cls_neighbors, patch_scores.shape[1])
            if m > 0:
                _, top_idx = torch.topk(patch_scores, m, dim=1)
                pre_selected_indices = top_idx + 1 # global indices
        else:
            if strategy != "random": # Random might not have attn, but if we want neighbors we need it?
                 logger.warning("keep_cls_neighbors > 0 but attn is None. Ignoring neighbor constraint.")

    # --- Startup Logging for Gumbel Softmax (Train only) ---
    if mode == "train" and not _gs_logged:
        gumbel_cfg = mode_cfg.get("gumbel", selection_cfg.get("gumbel", {}))
        if gumbel_cfg.get("enabled", False):
            logger.info("=== Gumbel-Softmax Selection Enabled (Train) ===")
            logger.info(f"  Strategy: {strategy}")
            logger.info(f"  Tau Max: {gumbel_cfg.get('tau', 1.0)}")
            logger.info(f"  Tau Min: {gumbel_cfg.get('tau_min', 0.3)}")
            logger.info(f"  Anneal: {gumbel_cfg.get('anneal', 'linear')} over {gumbel_cfg.get('steps', 10000)} steps")
            logger.info(f"  Hard: {gumbel_cfg.get('hard', True)}, Straight-Through: {gumbel_cfg.get('straight_through', True)}")
            _gs_logged = True

    if strategy == "random":
        tokens_sel, indices_sel, scores = select_random(
            tokens, n_alpha, generator=generator, 
            selection_cfg=selection_cfg, 
            pre_selected_indices=pre_selected_indices
        )

    elif strategy in ["topk_cls_attention", "gumbel_topk"]:
        if attn is None:
            logger.error("topk_cls_attention requested but attn is None. Returning original tokens.")
            indices = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
            return {"tokens_sel": tokens, "indices_sel": indices, "n_alpha": num_patches, "strategy": "none"}
        
        # Check for Gumbel config
        # We look in mode_cfg first (e.g. selection.train.gumbel), then fallback to selection.gumbel
        gumbel_cfg = mode_cfg.get("gumbel", selection_cfg.get("gumbel", {}))
        
        # "gumbel_topk" strategy implies gumbel_enabled=True, or it checks the flag
        gumbel_enabled = gumbel_cfg.get("enabled", False) or strategy == "gumbel_topk"

        if mode == "eval" and mode_cfg.get("gumbel_mc", {}).get("enabled", False):
            # MC-Averaged Gumbel-Softmax Ranking
            mc_cfg = mode_cfg.get("gumbel_mc", {})
            num_samples = mc_cfg.get("num_samples", 16)
            tau_eval = mc_cfg.get("tau_eval", 0.5)
            aggregate = mc_cfg.get("aggregate", "mean")
            
            # 1. Compute Base Scores (CLS attention on patches)
            # Handle already averaged attention
            if attn.dim() == 4:
                attn_mean = attn.mean(dim=1)
            else:
                attn_mean = attn
            cls_scores = attn_mean[:, 0, :]
            patch_scores = cls_scores[:, 1:] # [B, N-1]
            
            # 2. Get MC Averaged probabilities
            p_mean = compute_gumbel_mc_scores(
                patch_scores, 
                num_samples=num_samples, 
                tau=tau_eval, 
                aggregate=aggregate, 
                generator=generator
            )
            
            # 3. Optional Score Post-processing
            postprocess_cfg = mode_cfg.get("score_postprocess", {})
            if postprocess_cfg.get("enabled", False) and postprocess_cfg.get("method") == "temperature":
                temp = postprocess_cfg.get("temperature", 1.5)
                # p_mean is a probability distribution. Use log to get back to unnormalized logits, then scale.
                p_mean = torch.nn.functional.softmax(torch.log(p_mean + 1e-8) / temp, dim=-1)
                
            # Log the configs used for extracting metrics later
            # (We set gs_tau to tau_eval so stats aggregator picks it up if needed, though we log it separately in evaluator)
            gs_tau = tau_eval
            
            # 4. Deterministic Top-K Selection (supports Diversity)
            debug_ctx = selection_cfg.get("debug_ctx", None)
            tokens_sel, indices_sel, scores = select_by_scores(
                tokens, p_mean, n_alpha, 
                selection_cfg=selection_cfg,
                pre_selected_indices=pre_selected_indices
            )
            
        elif gumbel_enabled and (mode == "train" or mode_cfg.get("sample", False)):
            # Gumbel Sampling (Train or explicitly sampled eval)
            tau_start = gumbel_cfg.get("tau_start", gumbel_cfg.get("tau", 1.0))
            tau_end = gumbel_cfg.get("tau_end", gumbel_cfg.get("tau_min", 0.3))
            anneal_schedule = gumbel_cfg.get("schedule", gumbel_cfg.get("anneal", "linear"))
            anneal_steps = gumbel_cfg.get("steps", 10000)
            
            # Simple schedule fallback if compute_tau handles these, or parse step directly
            curr_step = step if step is not None else 0
            tau = compute_tau(curr_step, tau_start, tau_end, anneal_steps, anneal_schedule)
            
            hard = gumbel_cfg.get("hard", True)
            straight_through = gumbel_cfg.get("straight_through", True)
            
            tokens_sel, indices_sel, scores, gs_tau = sample_gumbel_topk(
                tokens, attn, n_alpha, 
                tau=tau, 
                hard=hard, 
                straight_through=straight_through, 
                generator=generator
            )
        else:
            # Standard Deterministic Top-K
            debug_ctx = selection_cfg.get("debug_ctx", None)
            tokens_sel, indices_sel, scores = select_topk_by_cls_attn(
                tokens, attn, n_alpha, 
                debug_ctx=debug_ctx, 
                selection_cfg=selection_cfg,
                pre_selected_indices=pre_selected_indices
            )
            
    elif strategy == "uniform_stride":
        tokens_sel, indices_sel = select_uniform_stride(tokens, n_alpha)
        # TODO: Support pre_selected_indices for uniform
    elif strategy == "uniform_grid":
        grid_size = selection_cfg.get("grid_size", None)
        tokens_sel, indices_sel = select_uniform_grid(tokens, n_alpha, grid_size=grid_size)
        n_alpha = indices_sel.shape[1] - 1
        # TODO: Support pre_selected_indices for uniform
    elif strategy in ["simplicial", "simplicial_pair", "simplicial_triangle", "simplicial_pair_plus_triangle"]:
        # "simplicial" is the new standard. Others kept for backward compat (handled inside select_simplicial)
        tokens_sel, indices_sel, scores, gs_tau = select_simplicial(
            tokens, 
            simplicial_context, 
            n_alpha, 
            selection_cfg=selection_cfg,
            pre_selected_indices=pre_selected_indices,
            mode=mode,
            step=step,
            generator=generator
        )
    else:
        logger.warning(f"Unknown selection strategy: {strategy}. Defaulting to none.")
        indices = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        return {"tokens_sel": tokens, "indices_sel": indices, "n_alpha": num_patches, "strategy": "none"}

    return {
        "tokens_sel": tokens_sel,
        "indices_sel": indices_sel,
        "n_alpha": n_alpha,
        "strategy": strategy,
        "scores": scores,
        "gs_tau": gs_tau
    }
