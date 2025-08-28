"""HookedTransformer wrapper with 2D RoPE via hooks."""
from __future__ import annotations

from typing import Any, List, Tuple

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ..data.stream import get_default_vocab
from .rope2d import apply_rope_2d


def build_model(cfg: Any) -> Tuple[HookedTransformer, Any]:
    """Instantiate HookedTransformer and return (model, forward_fn).

    forward_fn(tokens, pos2d, loss_mask) -> (loss, logits, cache)
    """
    vocab = get_default_vocab()
    d_vocab = max(vocab.values()) + 1

    mc = cfg.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_tl = HookedTransformerConfig(
        n_layers=mc.n_layers,
        d_model=mc.d_model,
        n_heads=mc.n_heads,
        d_head=mc.d_head,
        d_mlp=mc.d_mlp,
        n_ctx=mc.n_ctx,
        act_fn="gelu",
        d_vocab=d_vocab,
        attn_only=False,
        normalization_type="LNPre",
        device=str(device),
    )
    model = HookedTransformer(cfg_tl)
    # Move to device (dtype kept default; bf16 handled in later steps if needed)
    model = model.to(device)

    rotary_dim = mc.d_head

    def rope_single(x, pos2d_local):
        x2, _ = apply_rope_2d(x, x, pos2d_local, rotary_dim=rotary_dim)
        return x2

    def forward_fn(tokens: torch.Tensor, pos2d: torch.Tensor, loss_mask: torch.Tensor):
        tokens = tokens.to(device)
        pos2d = pos2d.to(device)
        loss_mask_l = loss_mask.to(device)

        # Prepare forward hooks for all layers
        def make_hook(pos2d_captured):
            def _hook_q(q, hook):
                return rope_single(q, pos2d_captured)

            def _hook_k(k, hook):
                return rope_single(k, pos2d_captured)

            return _hook_q, _hook_k

        fwd_hooks: List[Tuple[str, Any]] = []
        HookQ, HookK = make_hook(pos2d)
        for layer in range(model.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", HookQ))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", HookK))

        logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks)
        # Compute masked cross-entropy on t1 segment (mask==1)
        target = tokens
        logits_2d = logits
        if model.cfg.dtype in {torch.float16, torch.bfloat16}:
            logits_2d = logits_2d.float()
        # Flatten masked positions
        mask_flat = loss_mask_l.bool().view(-1)
        logits_flat = logits_2d.view(-1, logits_2d.size(-1))[mask_flat]
        target_flat = target.view(-1)[mask_flat]
        loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
        cache = None
        return loss, logits, cache

    return model, forward_fn
