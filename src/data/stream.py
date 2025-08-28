"""On-GPU data stream for cellular automata with toroidal padding."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .rules import bits_to_sb, random_rule_bits, sb_to_bits


class ToroidalNeighborhood:
    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.device = device
        k = torch.ones((1, 1, 3, 3), dtype=dtype)
        k[0, 0, 1, 1] = 0.0
        self.register_kernel(k)

    def register_kernel(self, k: torch.Tensor) -> None:
        self.kernel = k

    @torch.no_grad()
    def neighbors(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 8-neighbor counts with circular padding.

        Args:
            x: (B,1,H,W) binary 0/1 tensor (int/bool/float)
        Returns:
            counts: (B,1,H,W) int tensor in [0,8]
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("x must have shape (B,1,H,W)")
        k = self.kernel.to(device=x.device, dtype=torch.float32)
        x_pad = F.pad(x.to(torch.float32), (1, 1, 1, 1), mode="circular")
        y = F.conv2d(x_pad, k, padding=0)
        return y.round().to(torch.int32)


def _apply_rule(t: torch.Tensor, counts: torch.Tensor, rule_bits: torch.Tensor) -> torch.Tensor:
    """Apply S/B rule to get next board.

    Args:
        t: (B,1,H,W) int/bool current board
        counts: (B,1,H,W) int neighbor counts 0..8
        rule_bits: (B,18) bool tensor [S0..S8, B0..B8]
    Returns:
        t1: (B,1,H,W) int {0,1}
    """
    B, _, H, W = t.shape
    s_mask = rule_bits[:, :9]  # (B,9)
    b_mask = rule_bits[:, 9:]  # (B,9)
    # one-hot along 9 counts
    ar = torch.arange(9, device=t.device).view(1, 9, 1, 1)  # (1,9,1,1)
    c = counts.to(torch.int64)
    oh = (c == ar).squeeze(1)  # (B,9,H,W)
    # Membership per position
    surv_allowed = (oh & s_mask[:, :, None, None]).any(dim=1)  # (B,H,W)
    birth_allowed = (oh & b_mask[:, :, None, None]).any(dim=1)  # (B,H,W)
    alive = (t.squeeze(1) > 0)
    dead = ~alive
    t1 = (alive & surv_allowed) | (dead & birth_allowed)
    return t1.to(torch.int64).unsqueeze(1)


def _inject_pattern(board: torch.Tensor, pattern: torch.Tensor, top: int, left: int) -> None:
    """Write pattern onto board with wrap-around in-place.

    board: (1,H,W) or (H,W)
    pattern: (ph,pw)
    """
    H, W = board.shape[-2], board.shape[-1]
    ph, pw = pattern.shape
    for i in range(ph):
        for j in range(pw):
            r = (top + i) % H
            c = (left + j) % W
            board[..., r, c] = torch.maximum(board[..., r, c], pattern[i, j])


def _make_structured_board(H: int, W: int, device=None) -> torch.Tensor:
    """Make an empty board and randomly stamp a few canonical patterns."""
    board = torch.zeros((1, H, W), dtype=torch.int64, device=device)
    # Blinker (horizontal)
    blinker = torch.tensor([[1, 1, 1]], dtype=torch.int64, device=device)
    # Glider (one of the canonical orientations)
    glider = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=torch.int64,
        device=device,
    )
    # Stamp a couple with random positions
    for pat in (blinker, glider):
        top = torch.randint(0, H, ()).item()
        left = torch.randint(0, W, ()).item()
        _inject_pattern(board, pat, top, left)
    return board  # (1,H,W)


def make_batch(
    B: int,
    H: int,
    W: int,
    density_mix: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = None,
    rules: Optional[torch.Tensor] = None,
    structured_prob: float = 0.1,
):
    """Generate (rule_bits, t, t+1) batch entirely on device.

    - If rules is None: sample per-example random rules.
    - If rules is (18,) or (B,18): use as provided.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if density_mix is None:
        density_mix = (0.05, 0.1, 0.2, 0.3, 0.5)

    # Rule bits
    if rules is None:
        rule_bits = torch.stack([random_rule_bits().to(torch.bool) for _ in range(B)], dim=0).to(device)
    else:
        if rules.dim() == 1:
            rule_bits = rules.to(torch.bool).to(device).expand(B, -1)
        else:
            assert rules.shape == (B, 18)
            rule_bits = rules.to(torch.bool).to(device)

    # Boards t
    ps = torch.tensor(density_mix, device=device)
    idx = torch.randint(0, len(ps), (B,), device=device)
    p_sel = ps[idx].view(B, 1, 1, 1)
    t = (torch.rand((B, 1, H, W), device=device) < p_sel).to(torch.int64)

    # Optionally inject structured patterns into a fraction of batch
    if structured_prob > 0:
        n_struct = int(round(B * structured_prob))
        for b in range(n_struct):
            board = _make_structured_board(H, W, device=device)  # (1,H,W)
            t[b] = torch.maximum(t[b], board)

    # Counts and next state
    neigh = ToroidalNeighborhood(device=device)
    counts = neigh.neighbors(t.to(torch.float32))
    t1 = _apply_rule(t, counts, rule_bits)
    return rule_bits, t, t1


def get_default_vocab():
    """Return a simple vocab mapping for tokens used in assembly.

    Tokens:
      - '0', '1', '<BOS>', '<SEP>', '<SEP2>'
      - '<R{i}_0>' and '<R{i}_1>' for i in 0..17
    """
    vocab = {"0": 0, "1": 1, "<BOS>": 2, "<SEP>": 3, "<SEP2>": 4, "<SEP3>": 5, "<SEP4>": 6}
    idx = 5
    for i in range(18):
        vocab[f"<R{i}_0>"] = idx
        idx += 1
        vocab[f"<R{i}_1>"] = idx
        idx += 1
    return vocab


def assemble_sequence(rule_bits, t, t1, vocab=None, multi_steps: int = 0):
    """Assemble prefix-conditioned sequence and loss mask.

    Layout: <BOS>, 18×<R*_val>, <SEP>, H*W×t, <SEP2>, H*W×t1
    Returns tokens [B,T], loss_mask [B,T] (1 on t1 segment), pos2d [B,T,2].
    """
    B, _, H, W = t.shape
    # Additional segments for multi-steps
    extra_targets = 0
    if multi_steps >= 2:
        extra_targets += 1  # t+2
    if multi_steps >= 3:
        extra_targets += 1  # t+3
    T = 1 + 18 + 1 + H * W + 1 + H * W + extra_targets * (1 + H * W)
    if vocab is None:
        vocab = get_default_vocab()
    # Build token buffer
    tokens = torch.empty((B, T), dtype=torch.long, device=t.device)
    pos2d = torch.zeros((B, T, 2), dtype=torch.long, device=t.device)
    loss_mask = torch.zeros((B, T), dtype=torch.bool, device=t.device)

    # Index helpers
    i = 0
    tokens[:, i] = vocab["<BOS>"]
    i += 1
    # 18 rule tokens
    rb = rule_bits.to(torch.bool)
    for b in range(B):
        for r in range(18):
            tokens[b, i + r] = vocab[f"<R{r}_1>"] if rb[b, r].item() else vocab[f"<R{r}_0>"]
    i += 18
    tokens[:, i] = vocab["<SEP>"]
    i += 1
    # t board tokens (row-major)
    flat_t = t.view(B, -1)
    tokens[:, i : i + H * W] = flat_t
    # positions for these tokens
    rows = torch.arange(H, device=t.device).view(H, 1).expand(H, W)
    cols = torch.arange(W, device=t.device).view(1, W).expand(H, W)
    pos2d[:, i : i + H * W, 0] = rows.flatten()
    pos2d[:, i : i + H * W, 1] = cols.flatten()
    i += H * W
    tokens[:, i] = vocab["<SEP2>"]
    i += 1
    # t1 board tokens
    flat_t1 = t1.view(B, -1)
    tokens[:, i : i + H * W] = flat_t1
    pos2d[:, i : i + H * W, 0] = rows.flatten()
    pos2d[:, i : i + H * W, 1] = cols.flatten()
    # Loss mask on t1 segment only
    loss_mask[:, i : i + H * W] = True
    i += H * W
    # t+2 and t+3 segments (optional)
    if multi_steps >= 2:
        # compute t2
        neigh = ToroidalNeighborhood(device=t.device)
        counts1 = neigh.neighbors(t1.to(torch.float32))
        rb = rule_bits.to(torch.bool)
        rbB = rb if rb.dim() == 2 else rb.unsqueeze(0).expand(B, -1)
        t2 = _apply_rule(t1, counts1, rbB)
        tokens[:, i] = vocab["<SEP3>"]
        i += 1
        tokens[:, i : i + H * W] = t2.view(B, -1)
        pos2d[:, i : i + H * W, 0] = rows.flatten()
        pos2d[:, i : i + H * W, 1] = cols.flatten()
        loss_mask[:, i : i + H * W] = True
        i += H * W
        if multi_steps >= 3:
            counts2 = neigh.neighbors(t2.to(torch.float32))
            t3 = _apply_rule(t2, counts2, rbB)
            tokens[:, i] = vocab["<SEP4>"]
            i += 1
            tokens[:, i : i + H * W] = t3.view(B, -1)
            pos2d[:, i : i + H * W, 0] = rows.flatten()
            pos2d[:, i : i + H * W, 1] = cols.flatten()
            loss_mask[:, i : i + H * W] = True
            i += H * W
    assert i == T
    return tokens, loss_mask, pos2d
