# Roadmap

Status legend: ✅ done · 🔜 next · 💡 idea

## Milestone 1 — Foundations ✅

- ✅ Rust rules engine, base game 2-4 players, full trading protocol (offer / accept / reject / counter / confirm / cancel)
- ✅ Encodings (419 actions, 1788-float observation), public hand belief, event log, views
- ✅ Scripted random + heuristic bots (heuristic trades and counter-offers)
- ✅ pyo3 bindings, parallel `VecEnv`, Rust arena
- ✅ Entity-transformer policy, PPO self-play with seat chains and snapshot league, train/eval CLIs
- ✅ FastAPI server: rooms, pass & play, online, spectators, AI seats, persistence, reconnects
- ✅ Polished React client (desktop + mobile), lobby, trading UI
- ✅ Docs, CLAUDE.md, CI workflow, Dockerfile

## Milestone 2 — A strong trained agent 🔜 (owner: runs on the GPU machine)

- 🔜 Run `warmup` then `full` presets (see TRAINING.md); track `eval/heuristic_win_rate`
- 🔜 Copy the best checkpoint to `models/` and play it in the web app
- 🔜 Tune: batch size, entropy schedule, `vp_reward_scale` annealing, `pool_prob`
- 💡 Elo ladder: `catan-eval` round-robins between milestone checkpoints + heuristic, plotted over time
- 💡 Distil/quantize the final model for faster CPU inference on the server

## Milestone 3 — Smarter play

- 💡 Inference-time search: determinized / information-set MCTS in Rust that samples hidden
  hands from the public belief and uses the policy as prior and value net as leaf evaluator
- 💡 Opponent-aware trading: model each opponent's acceptance probability; directed offers
  (engine change: optional `to` seat on `OfferTrade`)
- 💡 Richer trade templates (3-card shapes) gated by an `ENCODING_VERSION` bump
- 💡 Auxiliary heads (predict final VP, opponents' hands) to speed up representation learning
- 💡 Population-based training with diverse reward shaping to avoid exploitable styles

## Milestone 4 — Product polish

- 💡 Run `npm run ui-smoke` in CI (start server + headless Chromium); extend it through game over
- 💡 Turn timers / AFK handling (host can already replace an offline player with the AI)
- ✅ Sound effects (procedural Web Audio, mute toggle)
- 💡 Resource-flow animations from tiles to player panels, card hover zoom
- 💡 Accessibility: keyboard navigation of board targets, colour-blind piece patterns, ARIA live log
- 💡 Game replays from the event log (scrubbable), shareable links, end-of-game stats charts
- 💡 Hints for beginners (spot ratings from the heuristic / value network)
- 💡 i18n

## Milestone 5 — Scale & variants

- 💡 Horizontal scaling: room ownership per node + sticky routing, or Redis-backed state/pub-sub
- 💡 Accounts, friends lists, ranked matchmaking vs AI levels
- 💡 5-6 player extension (bigger board topology, special building phase)
- 💡 Seafarers / Cities & Knights (major engine work)
