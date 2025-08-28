import torch

from src.data.rules import sb_to_bits
from src.data.stream import ToroidalNeighborhood, _apply_rule, make_batch


def slow_neighbors(board: torch.Tensor) -> torch.Tensor:
    """Slow Python loop neighbor counts with toroidal wrap.

    board: (B,1,H,W) int {0,1}
    returns: (B,1,H,W) int
    """
    B, _, H, W = board.shape
    out = torch.zeros_like(board)
    for b in range(B):
        for i in range(H):
            for j in range(W):
                s = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ii = (i + di) % H
                        jj = (j + dj) % W
                        s += int(board[b, 0, ii, jj].item())
                out[b, 0, i, j] = s
    return out


def test_neighbors_match_slow():
    torch.manual_seed(0)
    B, H, W = 1, 5, 5
    x = (torch.rand(B, 1, H, W) < 0.4).to(torch.int64)
    neigh = ToroidalNeighborhood()
    fast = neigh.neighbors(x.to(torch.float32)).to(torch.float32)
    slow = slow_neighbors(x).to(torch.float32)
    assert torch.allclose(fast, slow, atol=1e-6)


def test_known_transitions_conway():
    # Conway S23/B3
    S, B = {2, 3}, {3}
    bits = sb_to_bits(S, B).unsqueeze(0)  # (1,18)
    # Blinker oscillates: --- becomes | in next step
    H = W = 5
    t = torch.zeros((1, 1, H, W), dtype=torch.int64)
    t[0, 0, 2, 1:4] = 1  # horizontal at row 2, cols 1..3
    neigh = ToroidalNeighborhood()
    counts = neigh.neighbors(t.to(torch.float32))
    t1 = _apply_rule(t, counts, bits)
    exp = torch.zeros_like(t)
    exp[0, 0, 1:4, 2] = 1  # vertical at col 2, rows 1..3
    assert torch.equal(t1, exp)

    # Still life: 2x2 block stays
    t2 = torch.zeros((1, 1, H, W), dtype=torch.int64)
    t2[0, 0, 1:3, 1:3] = 1
    counts2 = neigh.neighbors(t2.to(torch.float32))
    t2_next = _apply_rule(t2, counts2, bits)
    assert torch.equal(t2, t2_next)


def test_make_batch_shapes():
    rb, t, t1 = make_batch(B=4, H=5, W=5, device=torch.device("cpu"), rules=sb_to_bits({2,3}, {3}))
    assert rb.shape == (4, 18)
    assert t.shape == (4, 1, 5, 5)
    assert t1.shape == (4, 1, 5, 5)

