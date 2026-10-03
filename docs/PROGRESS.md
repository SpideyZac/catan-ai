# Progress log

Living status document. Update it at the end of every work session (newest first).

## 2026-10-03 — Training throughput (model v2)

* Owner measured 931 sps on an RTX 5070 with the warmup preset (1.55M-param model):
  ~15 h for warmup, ~16 days for the full preset. Too slow.
* Model v2: removed the 72 edge tokens (152 → 80 tokens). Edge features are folded into
  endpoint vertex tokens and road logits come from endpoint pairs. Cost per sample on CPU
  12.7 → 4.6 ms (fwd+bwd, batch 256); attention ~3.6× cheaper. **Old checkpoints are
  incompatible** (`model_version` check gives a clear error) - restart training.
* Learner: on-device loss statistics (one host sync per minibatch instead of five per
  micro-batch), fused AdamW and TF32 on CUDA, default micro-batch 1024. Console shows
  rollout vs learn seconds.
* Measured by the owner on the RTX 5070: **931 → ~2,300 sps** (roll 2.0 s, learn 12.5 s per
  32,768-decision update). The learner is the bottleneck (~8 TFLOPS effective).
* Added `catan-bench` (attention backend / `torch.compile` comparison on the real GPU) and
  trainer flags `--attention` and `--compile`.
* `--attention math` gave the same 2,300 sps as `auto` (auto already used it), so the
  attention backend is not the lever. Added `triton-windows~=3.8` (matches torch 2.14's
  Triton) to the `cu130` extra so `torch.compile` can fuse the many small ops on Windows;
  compile now falls back to eager instead of crashing. Awaiting the owner's
  `catan-bench` numbers with compile.

## 2026-10-03 — First GPU run feedback

* The owner's first `warmup` run (12 GB GPU) crashed with CUDA OOM in `learn()`: a
  4096-sample minibatch through 6 layers materialized a 4096×8×152×152 fp32 attention
  matrix (exactly the 3,028,287,488-byte allocation that failed).
* Fix: micro-batched gradient accumulation (`micro_batch_size`, default 512) with
  automatic halving on OOM; the rollout value bootstrap is chunked too. Tests prove
  micro-batched gradients equal single-pass gradients and that OOM backoff works.

## 2026-10-01 — Initial build (session 1)

### Delivered

| Area | State | Verified by |
|---|---|---|
| Rust engine (`catan-core`) | Complete base game + trading, encodings, bots, views | 32 Rust tests (invariants checked after every action across hundreds of games), clippy `-D warnings` clean |
| Python bindings (`catan-py`) | `Game`, `VecEnv`, `arena` | `python/tests/test_engine.py` |
| Training stack | Entity transformer, PPO with seat chains + league, CLI presets, eval CLI | `test_training.py` (GAE hand-checked, trainer smoke + resume, neural agent full game); `catan-train --preset smoke` end to end |
| Server | Rooms, pass & play, online, spectators, bots, persistence | `test_server.py` (7 tests incl. full bot game over WebSocket, restart persistence) |
| Web client | Home, lobby, game, trading, modals, mobile | Typecheck + build; `npm run ui-smoke` (headless Chrome: solo game to a live trade offer, pass & play lobby → curtain) passing repeatedly with zero page errors; screenshots in `docs/images/` |
| Docs | README, CLAUDE.md, docs/* | – |
| Deployment | Dockerfile, CI workflow | **Not executed**: Docker daemon was not running on the dev machine and the repo has no remote yet |

### Benchmarks (dev machine, single thread unless noted)

* Random-bot games: ~1,400 games/s (~3.0M actions/s)
* Heuristic-bot games: ~4,200 games/s (~2.3M actions/s)
* Heuristic + observation + mask per decision: ~0.76M decisions/s
* `VecEnv` (256 envs, rayon) from Python with random legal actions: ~117k decisions/s
* Heuristic vs 3 random bots: wins ~99%

### Known limitations / open questions

* No trained neural model is shipped; "Neural" AI levels appear only after a checkpoint is
  placed in `models/`. Hyperparameters are reasoned defaults, not tuned (only smoke runs
  were executed here by design).
* The heuristic bot is decent but beatable by experienced players.
* Server is single-node (in-memory rooms + JSON snapshots).
* Human players have no turn timer; a disconnected human stalls the game until they
  return or the host clicks "Replace with AI" on their panel.
* The UI smoke test is not wired into CI (needs a running server + Chrome).

### Bugs found and fixed during verification

* Bot loop could die silently on an exception and stall a table → now logged, retried and
  resurrected on the next client message.
* Rate limit (40 msgs/10 s) could drop legitimate bursts → 120/10 s.
* Lobby seat/settings edits sent full lists built from stale client state, so quick
  successive edits clobbered each other → atomic `set_seat` + partial settings merge.

### Next steps

See [ROADMAP.md](ROADMAP.md) Milestone 2: run the warm-up and full training presets on the
GPU machine, then drop the best checkpoint into `models/`.
