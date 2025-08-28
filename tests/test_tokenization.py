import torch

from src.data.rules import sb_to_bits
from src.data.stream import assemble_sequence, get_default_vocab, make_batch


def test_lengths_and_mask():
    B, H, W = 3, 5, 5
    vocab = get_default_vocab()
    rb = sb_to_bits({2, 3}, {3}).unsqueeze(0).expand(B, -1)
    # deterministic boards
    t = torch.zeros((B, 1, H, W), dtype=torch.long)
    for b in range(B):
        t[b, 0, b % H, :] = 1
    t1 = torch.flip(t, dims=[-1])
    tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab)
    T = 1 + 18 + 1 + H * W + 1 + H * W
    assert tokens.shape == (B, T)
    assert mask.shape == (B, T)
    assert pos2d.shape == (B, T, 2)
    # mask zeros except last H*W tokens
    assert mask[:, : T - H * W].sum().item() == 0
    assert mask[:, T - H * W :].all()


def test_roundtrip_boards():
    B, H, W = 2, 5, 5
    vocab = get_default_vocab()
    rb, t, t1 = make_batch(B=B, H=H, W=W, device=torch.device("cpu"), rules=sb_to_bits({2, 3}, {3}))
    tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab)
    T = tokens.shape[1]
    # Recover t and t1 from tokens
    start_t = 1 + 18 + 1
    start_t1 = start_t + H * W + 1
    flat_t = tokens[:, start_t : start_t + H * W]
    flat_t1 = tokens[:, start_t1 : start_t1 + H * W]
    rec_t = flat_t.view(B, 1, H, W)
    rec_t1 = flat_t1.view(B, 1, H, W)
    assert torch.equal(rec_t, t)
    assert torch.equal(rec_t1, t1)
    # Positions for board tokens non-zero, prefix zeros
    assert pos2d[:, : start_t, :].abs().sum().item() == 0
    assert pos2d[:, start_t : start_t + H * W, :].max().item() == max(H - 1, W - 1)

