"""Single-GPU training loop for Life-GPT (S7)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.rules import sb_to_bits
from ..data.stream import assemble_sequence, get_default_vocab, make_batch
from ..model.hooked_life import build_model


@dataclass
class TrainStats:
    loss: float
    acc: float
    tokens_per_step: Optional[int] = None
    tps: Optional[float] = None
    secs: Optional[float] = None


def _acc_from_logits(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    m = mask.bool()
    correct = (preds[m] == tokens[m]).sum().item()
    total = int(m.sum().item())
    return correct / max(total, 1)


def train_loop(cfg) -> TrainStats:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = cfg.train.batch_per_gpu
    H, W = cfg.board.H, cfg.board.W
    vocab = get_default_vocab()

    # Build model
    model, fwd = build_model(cfg)
    model.train()

    # Optimizer & scheduler
    opt = AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    steps = int(300 if getattr(cfg.train, "fast", False) else (cfg.train.steps or 1000))
    warmup = min(cfg.train.warmup_steps, steps // 10 if steps > 0 else 0)
    sched = CosineAnnealingLR(opt, T_max=max(steps - warmup, 1))

    # Fixed rule for smoke: Conway S23/B3
    rule_bits = sb_to_bits({2, 3}, {3})

    # Logging
    run_log = []
    t0 = time.time()
    tokens_per_step = None

    scaler = None  # BF16: no grad scaling required
    last_loss = 0.0
    last_acc = 0.0

    for step in range(steps):
        rb, t, t1 = make_batch(B=B, H=H, W=W, device=device, rules=rule_bits, structured_prob=0.1)
        tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab)
        if tokens_per_step is None:
            tokens_per_step = int(tokens.numel())

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=getattr(cfg.train, "bf16", True)):
            loss, logits, _ = fwd(tokens, pos2d, mask)
        loss.backward()
        opt.step()
        if step >= warmup:
            sched.step()

        acc = _acc_from_logits(logits.detach(), tokens, mask)
        last_loss = float(loss.detach().item())
        last_acc = float(acc)
        run_log.append({"step": step, "loss": last_loss, "acc": last_acc})

        if (step + 1) % 50 == 0 or step == 0:
            print(f"step {step+1}/{steps} loss={last_loss:.4f} acc={last_acc:.4f}")

    secs = time.time() - t0
    tps = float(tokens_per_step * steps / max(secs, 1e-6)) if tokens_per_step is not None else None

    # Persist minimal logs
    os = __import__("os")
    os.makedirs("runs", exist_ok=True)
    with open("runs/train_smoke.jsonl", "a") as f:
        for r in run_log:
            f.write(json.dumps(r) + "\n")
    with open("runs/calibration.json", "w") as f:
        json.dump({"tokens_per_step": tokens_per_step, "tps": tps, "steps": steps, "secs": secs}, f)

    return TrainStats(loss=last_loss, acc=last_acc, tokens_per_step=tokens_per_step, tps=tps, secs=secs)

