Explorations with TransformerLens (interfere)

This folder contains small, reproducible scripts to analyze trained checkpoints with TransformerLens + HookedTransformer. The goal is to produce clear, beautiful plots/animations while testing concrete hypotheses inspired by:

- A Toy Model of Universality: Reverse Engineering how Networks Learn Group Operations (arXiv:2302.03025)
- Actually, Othello-GPT Has A Linear Emergent World Representation (AF post)
- TransformerLens docs (Main Demo)

Strong Hypotheses

- H1 Locality via attention: Target tokens learn to aggregate a 3×3 neighborhood from the t grid (i.e., attention implements a convolutional stencil). Expected: attention heatmaps over keys in t concentrate on a 3×3 around the corresponding spatial position.
- H2 Linear world variables: Residual features linearly encode per-cell quantities (neighbor count 0..8 and prior state). Expected: A linear probe on resid_pre recovers neighbor count with high R^2 and predicts next-alive with high AUC.
- H3 Physics tokens usage: Specific heads read “physics” rule bits (the 18 S/B prefix tokens). Expected: measurable attention from target queries to rule tokens, and causal responsibility (patching rule tokens changes predictions appropriately).
- H4 Layer roles: Early layers gather local evidence; later layers threshold/classify. Expected: probes improve across layers; attention becomes more localized.

Scripts

- utils.py: Helpers (compose cfg, load ckpt, build tokens/pos2d, figure dirs).
- attn_maps.py: Visualize attention heatmaps (per head) from a chosen target query to t-keys, projected to 2D. Saves PNGs/optional MP4s.
- probes.py: Train linear probes to decode neighbor count and next-alive from resid_pre across layers; saves curves and summary.
- physics_bits.py: Quantify attention to rule tokens and run small activation-patching tests (toggle rule bits and measure effect).
- embed_scatter.py: PCA scatter of resid_pre (layer L) at t positions across samples; color by neighbor count or next-alive.
- attn_contrib_vector.py: Contribution heatmap (weights × OV→alive) for a head with gradient quiver overlay — a “vector plot” of contribution flow.
- rule_vectors.py: Plot rule-bit vectors from token embeddings. Cosine similarity heatmap across bits, PCA scatter with S/B labels, and arrows for <Rr_0>→<Rr_1>.

Quick Start

1) Ensure a checkpoint exists at checkpoints/latest.pt (rank-0 writes it at the end of training). For DDP, you can copy it immediately or run a short single-GPU save.

2) Attention maps (life32 example):
   python explorations/attn_maps.py --config-name exp/life32 --head 0 --layer 0 --center --run-id <RUN_ID>

3) Linear probes:
   python explorations/probes.py --config-name exp/life32 --samples 512 --run-id <RUN_ID>

4) Physics bits analysis:
   python explorations/physics_bits.py --config-name exp/life32 --run-id <RUN_ID>

Outputs are written under assets/explorations/<timestamp>/.

Checkpoint discovery
- Training now saves to `checkpoints/<RUN_ID>/latest.pt` and updates `checkpoints/latest.pt` as a symlink.
- Explorations accept `--run-id` or `--ckpt` to choose a checkpoint; if omitted, the newest `checkpoints/*/latest.pt` is used (fallback to `checkpoints/latest.pt`).

Notes

- All analyses run with the same hooks used at train time: 2D RoPE (Q/K) and segment embeddings at hook_embed.
- For rollouts in videos we mask target inputs with <MASK> to match train/infer; analyses respect this convention.
- For reproducibility, you can add a fixed seed per run; the scripts are light and intended as interactive building blocks.
