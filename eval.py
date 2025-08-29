from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig

from src.data.rules import PREDEFINED, sb_to_bits
from src.data.stream import assemble_sequence, get_default_vocab
from src.data.stream import ToroidalNeighborhood as Neigh
from src.model.hooked_life import build_model


def step_model(model_fwd, rule_bits, t, vocab):
    B, _, H, W = t.shape
    rb = rule_bits.expand(B, -1)
    tokens, mask, pos2d = assemble_sequence(rb, t, torch.zeros_like(t), vocab=vocab)
    # Replace target segment inputs with [MASK] for inference-like context
    H, W = t.shape[-2], t.shape[-1]
    start = 1 + 18 + 1 + H * W + 1
    end = start + H * W
    if "<MASK>" in vocab:
        tokens[:, start:end] = vocab["<MASK>"]
    # We only need logits on t1 segment
    _, logits, _ = model_fwd(tokens, pos2d, mask)
    # Predict t+1 tokens from logits at the same positions (inputs are masked)
    start = 1 + 18 + 1 + H * W + 1  # index of first t1 token in the input
    logits_for_t1 = logits[:, start : start + H * W, :]
    pred_flat = logits_for_t1.argmax(dim=-1)
    pred = pred_flat.view(B, 1, H, W)
    return pred


def rollout_divergence(model_fwd, rule_bits, t0, steps: int = 100) -> int:
    device = t0.device
    B, _, H, W = t0.shape
    neigh = Neigh(device=device)
    t = t0.clone()
    div_step = steps
    for s in range(1, steps + 1):
        # Ground truth next state
        counts = neigh.neighbors(t.to(torch.float32))
        from src.data.stream import _apply_rule  # local import

        gt = _apply_rule(t, counts, rule_bits.expand(B, -1))
        # Model prediction
        pred = step_model(model_fwd, rule_bits, t, get_default_vocab())
        if (pred != gt).any().item():
            div_step = s
            break
        t = pred
    return div_step


@hydra.main(config_path="cfg", config_name="exp/life32", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fwd = build_model(cfg)
    # Load checkpoint if available
    ckpt_path = os.path.join("checkpoints", "latest.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state["model"], strict=False)
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint: {e}")

    vocab = get_default_vocab()

    results: Dict[str, Dict[str, float]] = {}
    for name in cfg.interp.rule_battery:
        S, B = PREDEFINED.get(name, ({2, 3}, {3}))
        rb = sb_to_bits(S, B).to(device).unsqueeze(0)
        # Accuracy over random boards
        accs = []
        for _ in range(10):
            H, W = cfg.board.H, cfg.board.W
            t = (torch.rand(8, 1, H, W, device=device) < 0.5).to(torch.int64)
            # Build targets via ground truth
            neigh = Neigh(device=device)
            counts = neigh.neighbors(t.to(torch.float32))
            from src.data.stream import _apply_rule

            t1 = _apply_rule(t, counts, rb.expand(t.size(0), -1))
            tokens, mask, pos2d = assemble_sequence(rb.expand(t.size(0), -1), t, t1, vocab=vocab)
            loss, logits, _ = fwd(tokens, pos2d, mask)
            acc = (logits.argmax(dim=-1)[mask] == tokens[mask]).float().mean().item()
            accs.append(acc)
        acc_avg = float(sum(accs) / len(accs))
        # Rollouts from canonical seeds
        # Blinker
        t0 = torch.zeros((1, 1, cfg.board.H, cfg.board.W), dtype=torch.long, device=device)
        t0[0, 0, cfg.board.H // 2, cfg.board.W // 2 - 1 : cfg.board.W // 2 + 2] = 1
        div = rollout_divergence(fwd, rb, t0, steps=100)
        results[name] = {"acc": acc_avg, "rollout_divergence": div}

    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(results, f)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
