import torch

from src.model.rope2d import apply_rope_2d


def test_zero_pos_identity():
    B, T, H, D = 2, 7, 4, 64
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    pos2d = torch.zeros(B, T, 2, dtype=torch.long)
    q2, k2 = apply_rope_2d(q, k, pos2d, rotary_dim=64)
    assert torch.allclose(q, q2)
    assert torch.allclose(k, k2)


def test_norm_preservation():
    B, T, H, D = 3, 5, 2, 64
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    pos2d = torch.randint(0, 10, (B, T, 2))
    q2, k2 = apply_rope_2d(q, k, pos2d, rotary_dim=64)
    qn, q2n = q.norm(dim=-1), q2.norm(dim=-1)
    kn, k2n = k.norm(dim=-1), k2.norm(dim=-1)
    assert torch.allclose(qn, q2n, atol=1e-5, rtol=1e-5)
    assert torch.allclose(kn, k2n, atol=1e-5, rtol=1e-5)


def test_broadcast_batches():
    B, T, H, D = 1, 6, 3, 64
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    pos2d = torch.randint(0, 10, (B, T, 2))
    q2, k2 = apply_rope_2d(q, k, pos2d, rotary_dim=64)
    q_rep, k_rep = q.repeat(3, 1, 1, 1), k.repeat(3, 1, 1, 1)
    pos_rep = pos2d.repeat(3, 1, 1)
    q2_rep, k2_rep = apply_rope_2d(q_rep, k_rep, pos_rep, rotary_dim=64)
    assert torch.allclose(q2_rep[0], q2)
    assert torch.allclose(k2_rep[0], k2)

