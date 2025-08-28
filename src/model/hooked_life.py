"""HookedTransformer wrapper with 2D RoPE via hooks."""
from __future__ import annotations

from typing import Any, List, Tuple

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ..data.stream import get_default_vocab
from .rope2d import apply_rope_2d
from .film import RuleFiLM


def build_model(cfg: Any) -> Tuple[HookedTransformer, Any]:
    """Instantiate HookedTransformer and return (model, forward_fn).

    forward_fn(tokens, pos2d, loss_mask) -> (loss, logits, cache)
    """
    vocab = get_default_vocab()
    d_vocab = max(vocab.values()) + 1

    mc = cfg.model
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
        device="cpu",
    )
    model = HookedTransformer(cfg_tl)

    rotary_dim = mc.d_head
    use_film = bool(getattr(mc, "film", False))
    film_module = RuleFiLM(mc.n_layers, mc.d_model).to(model.W_E.device) if use_film else None

    def rope_single(x, pos2d_local):
        x2, _ = apply_rope_2d(x, x, pos2d_local, rotary_dim=rotary_dim)
        return x2

    def forward_fn(tokens: torch.Tensor, pos2d: torch.Tensor, loss_mask: torch.Tensor):
        # Assume caller has moved model and tensors to the same device
        loss_mask_l = loss_mask

        # Prepare forward hooks for all layers
        def make_hook(pos2d_captured, film_params=None):
            def _hook_q(q, hook):
                return rope_single(q, pos2d_captured)

            def _hook_k(k, hook):
                return rope_single(k, pos2d_captured)
            def _hook_resid(x, hook):
                if film_params is None:
                    return x
                gamma, beta = film_params
                layer = hook.layer()
                # x: [B,T,D]; gamma/beta: [B,n_layers,D]
                g = 1.0 + gamma[:, layer, :].unsqueeze(1)
                b = beta[:, layer, :].unsqueeze(1)
                x32 = x.to(torch.float32)
                y = x32 * g + b
                return y.to(x.dtype)

            return _hook_q, _hook_k, _hook_resid

        fwd_hooks: List[Tuple[str, Any]] = []
        film_params = None
        if use_film:
            # Extract rule bits from first 18 rule tokens in prefix
            # tokens at positions 1..18 correspond to <R{i}_*> tokens
            rb = torch.zeros((tokens.size(0), 18), dtype=torch.float32, device=tokens.device)
            for i_b in range(tokens.size(0)):
                for r in range(18):
                    tid = tokens[i_b, 1 + r].item()
                    # <Rr_1> has id vocab[f"<R{r}_1>"]
                    rb[i_b, r] = 1.0 if tid == vocab[f"<R{r}_1>"] else 0.0
            gamma, beta = film_module(rb)
            film_params = (gamma, beta)
        HookQ, HookK, HookR = make_hook(pos2d, film_params)
        for layer in range(model.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", HookQ))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", HookK))
            fwd_hooks.append((f"blocks.{layer}.hook_resid_pre", HookR))

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
