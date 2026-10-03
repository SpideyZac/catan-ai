# AI & training guide

This document explains how the learning agent works and how to train it on your own
hardware. **Full training is meant to run on a GPU machine**; the development machine
only runs smoke tests.

## Overview

```
                 ┌───────────────── Rust (rayon, GIL released) ─────────────────┐
 Python PPO  ──► │ VecEnv: N games · scripted bot seats · reward shaping · reset │
  trainer   ◄──  │ observe() -> obs[N,1788], mask[N,419], actor[N]               │
                 │ step(actions) -> rewards[N,4], done, winner, turns            │
                 └───────────────────────────────────────────────────────────────┘
       │
       ▼
 EntityTransformer (PyTorch) ──► masked categorical policy + value
```

* **Agent**: one shared policy plays every seat; observations are seat-relative so it
  always sees itself as seat 0.
* **Algorithm**: PPO with GAE, clipped value loss, entropy bonus (cosine decayed), cosine
  LR schedule, AdamW, bf16 autocast on CUDA, KL early stopping.
* **Opponents**: pure self-play by default, plus a *league* of frozen past snapshots
  (`pool_prob` of games put one learner seat against snapshot opponents), plus optional
  scripted Rust bots (`bot_kind`, `bot_seats`) for curriculum warm-up.
* **Trading**: fully inside the action space. The agent proposes from 120 trade templates,
  answers offers (accept / reject / counter with a template), and picks a partner when
  several accept. Because the observation contains the live offer and every response,
  it also handles arbitrary offers made by humans in the web app.

## The multi-seat credit-assignment problem (read this before changing `ppo.py`)

In Catan only one seat acts per step, but one step can change the outcome for all seats
(the winning build ends the game for everyone; a steal hurts the victim). The trainer
therefore keeps one trajectory **chain per (env, seat)**:

1. Each stored transition records its env and seat, and is linked to the seat's previous
   transition (`next_index`).
2. `VecEnv.step` returns rewards for **all four seats**; each reward is added to the latest
   transition of that seat's chain. Terminal rewards thus reach every player, including
   those who did not make the final move.
3. GAE runs backwards along chains (`RolloutBuffer.compute_gae`), not along time.
4. Chains still open at the end of a rollout bootstrap from `V(observation of that seat
   right now)` via `VecEnv.observe_all_seats()`, which is exact rather than the usual
   "reuse the last value" approximation.

`python/tests/test_training.py::test_gae_follows_seat_chains` pins this behaviour down.

## Rewards

* Terminal: winner `+1`, others `-1/(n-1)` (`zero_sum=True`) or `0`.
* Truncated games (`max_turns`, default 300 turns) give no terminal reward.
* Optional shaping: `vp_reward_scale × Δ(total VP)` for every seat at every step (default
  0.02; the warm-up preset uses 0.05). Shaping is potential-like (it sums to
  `scale × final VP`), so it does not change the optimal policy much; lower it or set it
  to 0 late in training.

## Model

`catan_ai/model.py` — `EntityTransformer` (default) or `MLPNet` (baseline).

* Tokens: CLS + 19 hex + 54 vertex + 4 player + trade + global = **80 tokens**.
* Roads are not tokens: each edge's features are embedded and added to both endpoint
  vertex tokens. (Model v1 had 72 edge tokens, 152 in total; dropping them cut the cost per
  sample ~2.75× on CPU and attention ~3.6×.)
* Per-type linear embeddings + learned positional embeddings (positions are fixed board ids).
* Pre-norm transformer blocks using `scaled_dot_product_attention` with a learned
  **relation bias** per head for the board graph (hex-vertex, vertex-vertex, hex-hex,
  self). This gives the network the board's adjacency structure.
* Heads: vertex tokens → settlement/city logits; road logits from an MLP over the two
  endpoint vertex outputs (`a+b`, `a*b`, symmetric) plus the edge embedding; hex tokens →
  4 robber logits (no victim / relative victim 1-3); an MLP over CLS + me + trade tokens
  → all other actions (163); value MLP over CLS + me.
* Sizes: smoke ≈ 0.15M params; warmup/full presets (d=160, 6 layers, 5 heads of width 32) ≈ 1.62M. Keep
  `d_model / n_heads` a multiple of 8: narrower heads rule out fused attention kernels
  (the model warns).
* Checkpoints record `model_version` (currently 2) and refuse to load across versions.

## Setting up a training machine

```bash
git clone <repo> && cd catan-ai
# Linux + NVIDIA: PyPI torch already ships CUDA
uv sync --extra train
# Windows + NVIDIA (PyPI torch is CPU-only on Windows); also installs triton-windows
# so `--compile true` works:
uv sync --extra train --extra cu130
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Requirements: Rust (stable) to build the engine extension, `uv`, a CUDA GPU with ≥8 GB
VRAM for the full preset. CPU cores matter too: the env steps in parallel with rayon.

## Running

```bash
# 0. Sanity check (minutes, any machine)
uv run catan-train --preset smoke --run-dir runs/smoke

# 1. Warm-up against heuristic bots (fast early signal; ~1-3 h on a modern GPU)
uv run catan-train --preset warmup --run-dir runs/warmup

# 2. Main self-play run initialised from the warm-up weights
uv run catan-train --preset full --run-dir runs/main --resume runs/warmup/latest.pt \
    --bot-kind "" --bot-seats 0

# Resume an interrupted run (same config)
uv run catan-train --config runs/main/config.json --resume runs/main/latest.pt

# Override anything
uv run catan-train --preset full --num-envs 1024 --minibatch-size 16384 --model.d-model 192
```

Resuming restores the model, optimizer, update counter and league pool. The `warmup` and
`full` presets share the same architecture so step 2 can start from the warm-up weights
(the update counter carries over, so the LR schedule continues from where warm-up ended).
`--bot-kind ""` clears the preset's scripted opponents.

Outputs in `--run-dir`:

| File | Content |
|---|---|
| `config.json` | Full resolved config (reusable with `--config`) |
| `metrics.jsonl` | One JSON line per update + eval lines |
| `events.out.tfevents.*` | TensorBoard (`uv run tensorboard --logdir runs`) |
| `latest.pt` | Rolling checkpoint (model, optimizer, update, league pool) |
| `update_XXXXXXX.pt` | Milestones every `checkpoint_every × 20` updates |
| `final.pt` | End of training |

## Tuning throughput for your GPU

```bash
uv run catan-bench            # ~1 minute
```

It measures env throughput and learner forward+backward samples/s for every
`scaled_dot_product_attention` backend (`auto`, `efficient`, `cudnn`, `flash`, `math`)
and with `torch.compile`, reports peak memory, and prints the fastest flags, e.g.
`--attention efficient --compile true`. Pass them to `catan-train`. Unsupported options
are reported as "unavailable" (e.g. flash attention can't take the relation bias;
`torch.compile` needs Triton: bundled with Linux torch, `triton-windows` via the `cu130`
extra on Windows). With `--compile true` the trainer compiles only the learner's fixed-size
micro-batches, and falls back to eager mode with a warning if compilation fails.

Reference (RTX 5070, model v2, d=160, 6 layers, 5 heads, micro-batch 1024): learner
fwd+bwd 9,503 samples/s with the math attention kernel, 15,609 with the memory-efficient
kernel (what `auto` picks), **21,584 with `torch.compile`** (peak 2.5 GiB); env ~271k
decisions/s. End to end the warmup preset trains at **~5,700 sps** (warm-up ≈ 2.4 h,
full preset ≈ 2.7 days). Eager training ran at ~2,300 sps (`roll 2.0s / learn 12.5s`), so the
learner dominates; compile is therefore on by default (`--compile false` to disable).
With 8 heads (width 20) the memory-efficient attention kernel was rejected; the presets
now use 5 heads (width 32). If you still need more speed, `--epochs 2` cuts learner time
by a third.

## GPU memory

Attention with the relation bias can't use the flash kernel, so each layer may
materialize a `batch × heads × 80 × 80` score matrix. The learner therefore splits
every minibatch into micro-batches of `micro_batch_size` (gradient accumulation; the
update is mathematically identical). If a micro-batch still runs out of memory it is
halved and the minibatch retried, printing `CUDA out of memory: retrying with
micro_batch_size=N`. Pass that `N` next time (`--micro-batch-size N`) to skip the retries.
On a 12 GB card start with the default 1024. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation.

## What to watch

* `eval/heuristic_win_rate`: win rate of the greedy policy in 1-vs-3 games against the
  heuristic bot. Random play scores ~0%; 25% = parity; a strong agent should pass 50%+.
* `league_win_rate`: learner vs past snapshots (≈25% means no progress, rising = improving).
* `game_turns`, `truncated_frac`: games should get shorter and truncation should vanish.
* `approx_kl` (≈0.01-0.03), `clip_frac` (≈0.1-0.2), `entropy` (slow decline),
  `explained_var` (should rise toward 0.5+).
* `sps`: decisions per second (rollout + learn); the console line also shows the
  `roll`/`learn` seconds per update so you can see which side is the bottleneck. The env alone does >100k decisions/s,
  so the GPU forward/backward is usually the bottleneck.

## Evaluating & deploying

```bash
uv run catan-eval neural:runs/main/latest.pt heuristic heuristic heuristic --games 400
uv run catan-eval neural:runs/main/latest.pt neural:runs/main/update_0010000.pt --games 400
cp runs/main/latest.pt models/main.pt   # the server now offers "Neural (main)"
```

`neural:<path>@<temperature>` samples instead of playing greedily (more varied opponents).
The server loads neural models lazily, runs them on `CATAN_DEVICE` (default `cpu`; a CPU
forward pass takes a few ms), and falls back to the heuristic bot if a model errors.

## Hyperparameter notes

| Knob | Default | Notes |
|---|---|---|
| `micro_batch_size` | 1024 | Samples per forward/backward; gradients are accumulated over the minibatch, so this only affects memory. Halved automatically on CUDA OOM |
| `num_envs × rollout_steps` | 512 × 128 (full) | ~65k decisions per update; larger batches stabilise multi-agent PPO |
| `gamma` | 0.997 | Per *decision*; a seat makes ~100-200 decisions per game |
| `gae_lambda` | 0.95 | |
| `lr` | 3e-4 → 3e-5 cosine | |
| `entropy_coef` | 0.01 → 0.002 | The action space is large; keep some exploration for trading |
| `max_trade_offers_per_turn` | 3 | Caps proposal spam; the web app allows up to 10 |
| `pool_prob` | 0.3 | Fraction of games vs frozen snapshots |
| `max_turns` | 300 | Truncation for degenerate early policies |

## Ideas for stronger agents

See `docs/ROADMAP.md` (search at inference time with determinized MCTS over the public
belief, opponent modelling for trade acceptance, population-based training, etc.).
