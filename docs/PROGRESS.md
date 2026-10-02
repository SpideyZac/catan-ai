# Progress log

Living status document. Update it at the end of every work session (newest first).

## 2026-10-01 — Initial build (session 1)

### Delivered

| Area | State | Verified by |
|---|---|---|
| Rust engine (`catan-core`) | Complete base game + trading, encodings, bots, views | 32 Rust tests (invariants checked after every action across hundreds of games), clippy `-D warnings` clean |
| Python bindings (`catan-py`) | `Game`, `VecEnv`, `arena` | `python/tests/test_engine.py` |
| Training stack | Entity transformer, PPO with seat chains + league, CLI presets, eval CLI | `test_training.py` (GAE hand-checked, trainer smoke + resume, neural agent full game); `catan-train --preset smoke` end to end |
| Server | Rooms, pass & play, online, spectators, bots, persistence | `test_server.py` (7 tests incl. full bot game over WebSocket, restart persistence) |
| Web client | Home, lobby, game, trading, modals, mobile | Typecheck + build; Playwright-driven headless Chrome walkthroughs with zero console errors (screenshots in `docs/images/`) |
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
* No automated browser tests in the repo.

### Next steps

See [ROADMAP.md](ROADMAP.md) Milestone 2: run the warm-up and full training presets on the
GPU machine, then drop the best checkpoint into `models/`.
