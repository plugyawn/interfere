from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import torch
import os
import shutil
import subprocess
import tempfile

from ..data.stream import assemble_sequence, get_default_vocab
from ..data.stream import ToroidalNeighborhood as Neigh
from ..data.stream import _apply_rule as apply_rule


@torch.no_grad()
def _predict_next(fwd: Callable, rb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Predict t+1 using model forward.

    fwd should return either (loss, logits) or (loss, logits, cache).
    Returns a tensor (B,1,H,W) with {0,1}.
    """
    vocab = get_default_vocab()
    B, _, H, W = t.shape
    tokens, mask, pos2d = assemble_sequence(rb.expand(B, -1), t, torch.zeros_like(t), vocab=vocab)
    out = fwd(tokens, pos2d, mask)
    logits = out[1]  # supports both 2- and 3-tuple
    start = 1 + 18 + 1 + H * W + 1
    pred = logits.argmax(dim=-1)[:, start : start + H * W].view(B, 1, H, W)
    return pred


@torch.no_grad()
def _rollout_sequences(
    fwd: Callable,
    rb: torch.Tensor,
    t0: torch.Tensor,
    steps: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (gt_seq, pred_seq) lists of length steps+1 starting at t0."""
    device = t0.device
    B, _, H, W = t0.shape
    neigh = Neigh(device=device)
    gt_seq = [t0.clone()]
    pred_seq = [t0.clone()]
    t_gt = t0.clone()
    t_pred = t0.clone()
    for _ in range(steps):
        counts = neigh.neighbors(t_gt.to(torch.float32))
        t_gt = apply_rule(t_gt, counts, rb.expand(B, -1))
        t_pred = _predict_next(fwd, rb, t_pred)
        gt_seq.append(t_gt.clone())
        pred_seq.append(t_pred.clone())
    return gt_seq, pred_seq


@torch.no_grad()
def save_rollout_png(
    fwd: Callable,
    rule_bits: torch.Tensor,
    H: int,
    W: int,
    steps: int = 16,
    device: torch.device | None = None,
    savepath: str | None = None,
):
    """Save a PNG comparing ground-truth vs model rollouts from a canonical seed.

    Uses a centered blinker as the seed. Produces a 2 x (steps+1) grid.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rb = rule_bits.to(device).unsqueeze(0) if rule_bits.dim() == 1 else rule_bits.to(device)
    t0 = torch.zeros((1, 1, H, W), dtype=torch.long, device=device)
    t0[0, 0, H // 2, W // 2 - 1 : W // 2 + 2] = 1  # blinker

    gt_seq, pred_seq = _rollout_sequences(fwd, rb, t0, steps=steps)

    cols = steps + 1
    fig, axes = plt.subplots(2, cols, figsize=(max(12, cols * 0.8), 3))
    for i in range(cols):
        axes[0, i].imshow(gt_seq[i][0, 0].cpu(), cmap="Greys", vmin=0, vmax=1)
        axes[0, i].set_title(f"GT t+{i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_seq[i][0, 0].cpu(), cmap="Greys", vmin=0, vmax=1)
        axes[1, i].set_title(f"Pred t+{i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


@torch.no_grad()
def save_rollout_mp4(
    fwd: Callable,
    rule_bits: torch.Tensor,
    H: int,
    W: int,
    steps: int = 64,
    device: torch.device | None = None,
    savepath: str | None = None,
    fps: int = 8,
):
    """Save an MP4 showing GT vs autoregressive model rollouts.

    - Autoregressive: the model's prediction is fed back in as the next input.
    - Each frame is a 2x1 panel (GT on top, Pred on bottom) at time t.
    - Uses ffmpeg to encode PNG frames into MP4 (yuv420p for compatibility).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rb = rule_bits.to(device).unsqueeze(0) if rule_bits.dim() == 1 else rule_bits.to(device)
    t0 = torch.zeros((1, 1, H, W), dtype=torch.long, device=device)
    t0[0, 0, H // 2, W // 2 - 1 : W // 2 + 2] = 1

    gt_seq, pred_seq = _rollout_sequences(fwd, rb, t0, steps=steps)

    # Prepare temporary frames
    tmpdir = tempfile.mkdtemp(prefix="rollout_frames_")
    try:
        for i in range(len(gt_seq)):
            fig, axes = plt.subplots(2, 1, figsize=(4, 4))
            axes[0].imshow(gt_seq[i][0, 0].cpu(), cmap="Greys", vmin=0, vmax=1)
            axes[0].set_title(f"GT t+{i}")
            axes[0].axis("off")
            axes[1].imshow(pred_seq[i][0, 0].cpu(), cmap="Greys", vmin=0, vmax=1)
            axes[1].set_title(f"Pred t+{i}")
            axes[1].axis("off")
            plt.tight_layout()
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        # Ensure ffmpeg exists
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH; please install ffmpeg")

        os.makedirs(os.path.dirname(savepath or "assets/rollouts/out.mp4"), exist_ok=True)
        out_path = savepath or os.path.join("assets", "rollouts", "rollout.mp4")
        # Build ffmpeg command
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    finally:
        # Cleanup frames
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
