"""Single-GPU training loop for Life-GPT (S7)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ..data.rules import sb_to_bits
from ..data.stream import assemble_sequence, get_default_vocab, make_batch
from ..model.hooked_life import build_model
from ..viz.rollout import save_rollout_png, save_rollout_mp4


def _sched_sampling_prob(step: int, start: int, end: int, p_max: float) -> float:
    if end <= start:
        return float(p_max)
    if step <= start:
        return 0.0
    if step >= end:
        return float(p_max)
    return float(p_max) * float(step - start) / max(float(end - start), 1.0)


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


def _target_segment_bounds(H: int, W: int, multi_steps: int) -> list[tuple[int, int]]:
    """Return [(start,end)] (half-open) indices for each supervised target segment.

    Layout: <BOS>, 18 rule, <SEP>, H*W (t), <SEP2>, H*W (t+1), [<SEP3>, H*W (t+2), ...]
    """
    HxW = H * W
    i = 0
    i += 1              # BOS
    i += 18             # rule
    i += 1              # SEP
    i += HxW            # t
    i += 1              # SEP2
    bounds = []
    # t+1
    start = i
    end = i + HxW
    bounds.append((start, end))
    i = end
    if multi_steps >= 2:
        i += 1          # SEP3
        bounds.append((i, i + HxW))
        i += HxW
        if multi_steps >= 3:
            i += 1      # SEP4
            bounds.append((i, i + HxW))
    return bounds


def train_loop(cfg) -> TrainStats:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = cfg.train.batch_per_gpu
    H, W = cfg.board.H, cfg.board.W
    vocab = get_default_vocab()

    # Build model
    model, fwd = build_model(cfg)
    model = model.to(device)
    model.train()

    # Optimizer & scheduler
    opt = AdamW(model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    steps = int(300 if getattr(cfg.train, "fast", False) else (cfg.train.steps or 1000))
    warmup = min(cfg.train.warmup_steps, steps // 10 if steps > 0 else 0)
    if warmup > 0:
        warm = LinearLR(opt, start_factor=0.1, total_iters=warmup)
        cosine = CosineAnnealingLR(opt, T_max=max(steps - warmup, 1))
        sched = SequentialLR(opt, schedulers=[warm, cosine], milestones=[warmup])
    else:
        sched = CosineAnnealingLR(opt, T_max=max(steps, 1))

    # Fixed rule for smoke: Conway S23/B3
    rule_bits = sb_to_bits({2, 3}, {3})

    # Logging
    run_log = []
    t0 = time.time()
    tokens_per_step = None

    scaler = None  # BF16: no grad scaling required
    last_loss = 0.0
    last_acc = 0.0

    # Tag this run to avoid overwriting rollouts
    run_id = time.strftime("%Y%m%d_%H%M%S")
    roll_dir = os.path.join("assets", "rollouts", run_id)
    os.makedirs(roll_dir, exist_ok=True)
    print(f"Rollouts will be saved under: {roll_dir}")

    pbar = tqdm(range(steps), desc="train", leave=True)
    # EMA setup
    ema_cfg = getattr(cfg.train, "ema", {})
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema_decay = float(getattr(ema_cfg, "decay", 0.999))
    ema: Dict[str, torch.Tensor] = {}
    if ema_enabled:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    ema[n] = p.detach().float().clone()
    for step in pbar:
        aug = getattr(cfg.train, "augment", {})
        d4p = float(getattr(aug, "d4_prob", 0.0) or 0.0)
        shp = float(getattr(aug, "shift_prob", 0.0) or 0.0)
        rb, t, t1 = make_batch(B=B, H=H, W=W, device=device, rules=rule_bits, structured_prob=0.1, d4_prob=d4p, shift_prob=shp)
        tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab, multi_steps=getattr(cfg.train, "multi_steps", 0))
        if tokens.size(1) > model.cfg.n_ctx:
            raise RuntimeError(f"Sequence length {tokens.size(1)} exceeds n_ctx {model.cfg.n_ctx}")
        if tokens_per_step is None:
            tokens_per_step = int(tokens.numel())

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=getattr(cfg.train, "bf16", True)):
            # Train under inference-like inputs: mask out future target segments in the input
            tokens_in = tokens.clone()
            mask_id = int(vocab.get("<MASK>", vocab["0"]))
            for s, e in _target_segment_bounds(H, W, getattr(cfg.train, "multi_steps", 0)):
                assert e <= tokens.size(1), "target segment bounds exceed sequence length"
                tokens_in[:, s:e] = mask_id
            loss, logits, _ = fwd(tokens_in, pos2d, mask, labels_tokens=tokens)
        loss.backward()
        # Gradient clipping
        gc = float(getattr(cfg.train, "grad_clip", 0.0) or 0.0)
        if gc > 0.0:
            clip_grad_norm_(model.parameters(), gc)
        opt.step()
        # EMA update
        if ema_enabled:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        ema[n].mul_(ema_decay).add_(p.detach().float(), alpha=1.0 - ema_decay)
        sched.step()

        acc = _acc_from_logits(logits.detach(), tokens, mask)
        last_loss = float(loss.detach().item())
        last_acc = float(acc)
        run_log.append({"step": step, "loss": last_loss, "acc": last_acc})

        if (step + 1) % 5 == 0 or step == 0:
            pbar.set_postfix({"loss": f"{last_loss:.4f}", "acc": f"{last_acc:.4f}"})
            pbar.write(f"step {step+1}/{steps} loss={last_loss:.4f} acc={last_acc:.4f}")

        # Periodic rollout visualization (MP4, longer autoregressive rollout)
        if (step == 0) or ((step + 1) % 50 == 0):
            try:
                os.makedirs(roll_dir, exist_ok=True)
                save_rollout_mp4(
                    fwd,
                    rule_bits,
                    H,
                    W,
                    steps=64,
                    device=device,
                    savepath=os.path.join(roll_dir, f"sg_rollout_step_{step+1:06d}.mp4"),
                    fps=8,
                )
            except Exception:
                pass

    secs = time.time() - t0
    tps = float(tokens_per_step * steps / max(secs, 1e-6)) if tokens_per_step is not None else None

    # Persist minimal logs
    os.makedirs("runs", exist_ok=True)
    with open("runs/train_smoke.jsonl", "a") as f:
        for r in run_log:
            f.write(json.dumps(r) + "\n")
    with open("runs/calibration.json", "w") as f:
        json.dump({"tokens_per_step": tokens_per_step, "tps": tps, "steps": steps, "secs": secs}, f)

    return TrainStats(loss=last_loss, acc=last_acc, tokens_per_step=tokens_per_step, tps=tps, secs=secs)


def train_loop_ddp(cfg) -> TrainStats:
    """Multi-GPU DDP training with token-budget scheduling.

    Uses calibration tokens_per_step (single-GPU) to compute steps as:
    floor(target_tokens / (tokens_per_step * world_size)).
    """
    import json
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    device = torch.device(f"cuda:{local_rank}")

    B = cfg.train.batch_per_gpu
    H, W = cfg.board.H, cfg.board.W
    vocab = get_default_vocab()

    # Build and wrap model
    model, _ = build_model(cfg)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer & scheduler
    opt = AdamW(ddp_model.parameters(), lr=cfg.train.lr, betas=(0.9, 0.95), weight_decay=cfg.train.weight_decay)
    # EMA setup
    ema_cfg = getattr(cfg.train, "ema", {})
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema_decay = float(getattr(ema_cfg, "decay", 0.999))
    ema: Dict[str, torch.Tensor] = {}

    # Read calibration
    tokens_per_step_single = None
    try:
        with open("runs/calibration.json", "r") as f:
            cal = json.load(f)
            tokens_per_step_single = int(cal.get("tokens_per_step", 0) or 0)
    except Exception:
        pass
    if not tokens_per_step_single:
        # Fallback: estimate from sequence length
        seq_len = 1 + 18 + 1 + H * W + 1 + H * W
        tokens_per_step_single = B * seq_len
    tokens_per_step_global = tokens_per_step_single * world_size
    # Honor explicit step override if provided; else compute from token budget
    explicit_steps = getattr(cfg.train, "steps", None)
    if explicit_steps is not None and int(explicit_steps) > 0:
        steps = int(explicit_steps)
    else:
        target_tokens = int(cfg.train.target_tokens)
        steps = max(target_tokens // max(tokens_per_step_global, 1), 1)
    warmup = min(cfg.train.warmup_steps, steps // 10 if steps > 0 else 0)
    if warmup > 0:
        warm = LinearLR(opt, start_factor=0.1, total_iters=warmup)
        cosine = CosineAnnealingLR(opt, T_max=max(steps - warmup, 1))
        sched = SequentialLR(opt, schedulers=[warm, cosine], milestones=[warmup])
    else:
        sched = CosineAnnealingLR(opt, T_max=max(steps, 1))

    if rank == 0:
        est_minutes = (steps * tokens_per_step_single) / max((cal.get("tps", 1e6) if 'cal' in locals() and cal else 1e6), 1e-6) / 60.0
        print(
            f"DDP world_size={world_size} global_batch={B*world_size} tokens_per_step={tokens_per_step_global} steps={steps} est_min={est_minutes:.1f}"
        )

    rule_bits = sb_to_bits({2, 3}, {3})

    last_loss = 0.0
    last_acc = 0.0

    # Local forward helper using inner model for hooks
    inner = ddp_model.module
    if ema_enabled:
        with torch.no_grad():
            for n, p in inner.named_parameters():
                if p.requires_grad:
                    ema[n] = p.detach().float().clone()

    def fwd(tokens, pos2d, mask, labels_tokens=None):
        # Re-register hooks each call via run_with_hooks
        fwd_hooks = []
        rotary_dim = cfg.model.d_head
        def rope_single(x, pos2d_local):
            from ..model.rope2d import apply_rope_2d as rope
            x2, _ = rope(x, x, pos2d_local, rotary_dim=rotary_dim)
            return x2
        def _hq(q, hook):
            return rope_single(q, pos2d.to(q.device))
        def _hk(k, hook):
            return rope_single(k, pos2d.to(k.device))
        def _hs(scores, hook):
            B2, Hh, Q, K = scores.shape
            HxW = H * W
            start_t1 = 1 + 18 + 1 + HxW + 1
            allowed_k = torch.zeros((K,), dtype=torch.bool, device=scores.device)
            allowed_k[: start_t1 - 1] = True
            q_in_target = torch.zeros((Q,), dtype=torch.bool, device=scores.device)
            q_in_target[start_t1:] = True
            bad = q_in_target[:, None] & (~allowed_k)[None, :]
            scores = scores.masked_fill(bad.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
            return scores
        for layer in range(inner.cfg.n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_q", _hq))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_k", _hk))
            fwd_hooks.append((f"blocks.{layer}.attn.hook_attn_scores", _hs))
        # Segment embedding at input
        def _he(x, hook):
            seg = torch.zeros((x.size(0), x.size(1)), dtype=torch.long, device=x.device)
            HxW = H * W
            i = 0
            i += 1; i += 18; i += 1
            seg[:, i : i + HxW] = 1  # state
            i += HxW
            i += 1
            seg[:, i : i + HxW] = 2  # t+1
            i += HxW
            ms = int(getattr(getattr(cfg, 'train', {}), 'multi_steps', 0))
            if ms >= 2:
                i += 1; seg[:, i : i + HxW] = 2; i += HxW
                if ms >= 3:
                    i += 1; seg[:, i : i + HxW] = 2; i += HxW
            if getattr(inner, 'segment_embed', None) is None:
                return x
            return x + inner.segment_embed(seg)
        fwd_hooks.append(("hook_embed", _he))
        logits = inner.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks)
        # Next-token loss (shift)
        if tokens.size(1) < 2:
            raise ValueError("Sequence too short for next-token training")
        logits_shift = logits[:, :-1, :]
        target_src = tokens if labels_tokens is None else labels_tokens
        target_shift = target_src[:, 1:]
        mask_shift = mask[:, 1:]
        assert not mask[:, 0].any().item(), "loss_mask must be false at position 0"
        mask_flat = mask_shift.reshape(-1).bool()
        loss = F.cross_entropy(
            logits_shift.reshape(-1, logits_shift.size(-1))[mask_flat],
            target_shift.reshape(-1)[mask_flat],
        )
        return loss, logits

    # Tag this run and progress bar on rank 0 only
    run_id = time.strftime("%Y%m%d_%H%M%S")
    roll_dir = os.path.join("assets", "rollouts", run_id)
    if rank == 0:
        os.makedirs(roll_dir, exist_ok=True)
        print(f"Rollouts will be saved under: {roll_dir}")
    # Progress bar on rank 0 only
    if rank == 0:
        iterator = tqdm(range(steps), desc="train(ddp)", leave=True)
    else:
        iterator = range(steps)

    for step in iterator:
        aug = getattr(cfg.train, "augment", {})
        d4p = float(getattr(aug, "d4_prob", 0.0) or 0.0)
        shp = float(getattr(aug, "shift_prob", 0.0) or 0.0)
        rb, t, t1 = make_batch(B=B, H=H, W=W, device=device, rules=rule_bits, structured_prob=0.1, d4_prob=d4p, shift_prob=shp)
        tokens, mask, pos2d = assemble_sequence(rb, t, t1, vocab=vocab, multi_steps=getattr(cfg.train, "multi_steps", 0))
        # Ensure context length does not exceed model capacity
        if tokens.size(1) > inner.cfg.n_ctx:
            raise RuntimeError(f"Sequence length {tokens.size(1)} exceeds n_ctx {inner.cfg.n_ctx}")
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=getattr(cfg.train, "bf16", True)):
            tokens_in = tokens.clone()
            mask_id = int(vocab.get("<MASK>", vocab["0"]))
            for s, e in _target_segment_bounds(H, W, getattr(cfg.train, "multi_steps", 0)):
                assert e <= tokens.size(1), "target segment bounds exceed sequence length"
                tokens_in[:, s:e] = mask_id
            loss, logits = fwd(tokens_in, pos2d, mask, labels_tokens=tokens)
        loss.backward()
        # Gradient clipping
        gc = float(getattr(cfg.train, "grad_clip", 0.0) or 0.0)
        if gc > 0.0:
            clip_grad_norm_(ddp_model.parameters(), gc)
        opt.step()
        # EMA update
        if ema_enabled:
            with torch.no_grad():
                for n, p in inner.named_parameters():
                    if p.requires_grad:
                        ema[n].mul_(ema_decay).add_(p.detach().float(), alpha=1.0 - ema_decay)
        sched.step()

        if rank == 0 and ((step + 1) % 5 == 0 or step == 0):
            acc = _acc_from_logits(logits.detach(), tokens, mask)
            last_loss = float(loss.detach().item())
            last_acc = float(acc)
            # tqdm.write to avoid breaking the bar
            if hasattr(iterator, "write"):
                iterator.set_postfix({"loss": f"{last_loss:.4f}", "acc": f"{last_acc:.4f}"})
                iterator.write(f"[rank0] step {step+1}/{steps} loss={last_loss:.4f} acc={last_acc:.4f}")
            else:
                print(f"[rank0] step {step+1}/{steps} loss={last_loss:.4f} acc={last_acc:.4f}")

        # Periodic rollout visualization on rank 0 only (MP4)
        if rank == 0 and ((step == 0) or ((step + 1) % 50 == 0)):
            try:
                os.makedirs(roll_dir, exist_ok=True)
                # Wrap local fwd to match (tokens,pos2d,mask)->(loss,logits,cache)
                def fwd_wrap(tokens, pos2d, mask):
                    l, logits = fwd(tokens, pos2d, mask)
                    return l, logits, None
                save_rollout_mp4(
                    fwd_wrap,
                    rule_bits,
                    H,
                    W,
                    steps=64,
                    device=device,
                    savepath=os.path.join(roll_dir, f"ddp_rollout_step_{step+1:06d}.mp4"),
                    fps=8,
                )
            except Exception:
                pass

    # Save checkpoint (rank 0)
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        path = "checkpoints/latest.pt"
        state = {
            "model": inner.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg.__dict__,
        }
        if ema_enabled:
            # store ema separately
            state["ema"] = {k: v.cpu() for k, v in ema.items()}
        torch.save(state, path)

    dist.barrier()
    dist.destroy_process_group()

    return TrainStats(loss=last_loss, acc=last_acc, tokens_per_step=tokens_per_step_global, tps=None, secs=None)
