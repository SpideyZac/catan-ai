# Progress log

Living status document. Update it at the end of every work session (newest first).

## 2026-10-03 — Background music and quick-start rules

* `web/src/lib/music.ts`: procedural folk music (no asset files) - a slow D Dorian harbour
  air for home/lobby and a G Mixolydian 6/8 jig with bodhrán for games, crossfading between
  screens. Separate 🎵 toggle (home, lobby, game top bar), persisted as `catan.music`.
  Levels measured in headless Chrome: music peaks ~0.09-0.12 vs effects ~0.24.
* Play vs AI / Watch the AI now open a setup dialog (players 2-4, points to win, trade
  offers, AI speed, beginner board, AI level), remembered in `localStorage`. The lobby's
  rules form is now the shared `RulesFields` component.
* `npm run ui-smoke` drives the dialog (3-player game) and checks music starts/stops; a
  one-off check confirmed the chosen settings and bot level reach the server for both modes.

## 2026-10-03 — UI sound effects

* `web/src/lib/sound.ts`: procedural Web Audio effects (no asset files) for dice, 7s,
  your production, builds, dev cards, robber, steals, trade offers/deals/no-deal, Longest
  Road / Largest Army, win/lose, your-turn chime and incoming chat. Per-update batches are
  deduplicated and capped; mute toggle in the top bar persists in `localStorage`.
* Verified with `npm run ui-smoke` and a headless check that effects fire during setup and
  stop when muted. See `docs/WEB.md` → Sound.

## 2026-10-03 — Warm-up → self-play handoff fix

* Diagnosed the regression after resuming the warm-up into the full preset (heuristic win
  rate 0.29 → ~0.08, entropy 0.58 → 1.5): the LR/entropy cosines ran over the absolute
  update counter, so update 1501 of a 20000-update schedule jumped LR 3e-5 → 2.96e-4 and
  the entropy bonus 0.002 → 0.0099. Removing all heuristic seats at the same time also
  removed the only fixed reference.
* `schedule_start_update`: the cosine now runs from it to `total_updates`. Automatic on
  resume: kept for the same run, set to the checkpoint's update when the schedule settings
  change (logged).
* `bot_seats` is a float: the fractional part is the per-game probability of one more
  scripted seat (`0.5` = a heuristic bot in half the games).
* `player_counts` (`--player-counts 2,3,4`) mixes 2/3/4-player games in one run; extra
  sizes get their own eval (`eval/heuristic_win_rate_2p`, …). The encoding already
  supported 2-4 players, but presets still train 4-player only, so the model is
  out of distribution in 2/3-player rooms until trained with this flag.
* Presets: warm-up 1500 → 3000 updates; new `selfplay` preset (= `full` with LR 1e-4,
  entropy 0.003 → 0.001, `bot_kind heuristic`, `bot_seats 0.5`). Handoff is now
  `--preset selfplay --resume runs/warmup/final.pt`. The `runs/main` checkpoints from the
  bad handoff should be discarded.

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
  compile now falls back to eager instead of crashing.
* `catan-bench` on the RTX 5070: eager 7,843 vs compiled 15,137 learner samples/s
  (1.93×). Efficient attention was rejected because the head width was 20 (needs a
  multiple of 8). Presets now use 5 heads (width 32) and compile defaults to on. Next:
  owner re-runs `catan-bench` (efficient attention should now be eligible) and restarts
  warm-up (head count changed, so the old warm-up checkpoint doesn't fit the new presets).
* Re-bench with 5 heads: eager 9,545, compiled **18,063** samples/s. Efficient attention was
  still refused because the relation-bias mask was a permuted view (last-dim stride 5);
  it is now made contiguous, and `catan-bench` also tries `efficient + compile`.
* Final RTX 5070 bench: math 9,503 · efficient (= auto) 15,609 · **compiled 21,584**
  samples/s, peak 2.5 GiB. `auto` already selects the efficient kernel, so the defaults
  (auto + compile) are optimal. Learner went 7,843 → 21,584 samples/s overall (2.75×);
  expected training throughput ~5,000 sps vs 931 at the start of the day.
* **Measured: 5,700 sps** in the warmup preset (6.1× the original 931). Warm-up ≈ 2.4 h,
  full preset ≈ 2.7 days on the RTX 5070.

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
