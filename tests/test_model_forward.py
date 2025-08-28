import types

import torch

from src.data.rules import sb_to_bits
from src.data.stream import assemble_sequence, get_default_vocab, make_batch
from src.model.hooked_life import build_model


def make_cfg():
    mc = types.SimpleNamespace(
        n_layers=2,
        d_model=256,
        n_heads=4,
        d_head=64,
        d_mlp=1024,
        n_ctx=600,
        attn_impl="sdpa",
        film=False,
    )
    tc = types.SimpleNamespace(bf16=True)
    cfg = types.SimpleNamespace(model=mc, train=tc)
    return cfg


def test_forward_smoke_cuda():
    if not torch.cuda.is_available():
        return  # skip if no GPU
    device = torch.device("cuda:0")
    cfg = make_cfg()
    model, fwd = build_model(cfg)
    B, H, W = 2, 8, 8
    rb, t, t1 = make_batch(B=B, H=H, W=W, device=device, rules=sb_to_bits({2, 3}, {3}))
    tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=get_default_vocab())
    loss, logits, cache = fwd(tokens, pos2d, mask)
    assert logits.shape[:2] == tokens.shape
    assert torch.isfinite(loss).item() is True

