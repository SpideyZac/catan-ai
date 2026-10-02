# Architecture

```
┌────────────────────────────── browser (web/) ───────────────────────────────┐
│ React + TypeScript + SVG board  ── WebSocket JSON ──┐                        │
└─────────────────────────────────────────────────────┼────────────────────────┘
                                                      ▼
┌──────────────────────── python/catan_ai/server (FastAPI) ───────────────────┐
│ rooms · seats · lobby · chat · per-client redaction · bot loop · snapshots  │
│        │ CatanGame (engine.py)              │ BotRegistry (ai.py)            │
└────────┼────────────────────────────────────┼────────────────────────────────┘
         ▼                                    ▼
┌──────── catan_ai._engine (crates/catan-py, pyo3) ────────┐   ┌─ catan_ai (PyTorch) ─┐
│ Game (JSON API + numpy encodings) · VecEnv (rayon) · arena│◄──│ model.py · ppo.py    │
└──────────────────────────┬────────────────────────────────┘   │ agents.py · train.py │
                           ▼                                    │ evaluate.py          │
┌──────────────── crates/catan-core (pure Rust) ───────────────┐└──────────────────────┘
│ rules · topology bitboards · trading · encodings · bots      │
└───────────────────────────────────────────────────────────────┘
```

## Layers

1. **`catan-core`** (Rust, no Python): the rules engine and everything that must be fast
   or must be identical everywhere: legal moves, transitions, encodings for agents,
   scripted bots, information-filtered views. See [ENGINE.md](ENGINE.md).
2. **`catan-py`** (Rust, pyo3 + numpy + rayon): exposes `Game`, `VecEnv` and `arena` as
   `catan_ai._engine`. JSON strings cross the boundary for the game API (stable and easy to
   version); numpy arrays cross it for encodings. `VecEnv` releases the GIL and steps
   environments in parallel.
3. **`catan_ai`** (Python): `engine.py` (dict-based wrapper), `agents.py`, `model.py`,
   `ppo.py`, `train.py`, `evaluate.py`. See [TRAINING.md](TRAINING.md).
4. **`catan_ai.server`** (FastAPI): multiplayer rooms and AI seats. See [SERVER.md](SERVER.md).
5. **`web/`** (React/Vite/TS): the client. See [WEB.md](WEB.md).

## Key decisions

| Decision | Why |
|---|---|
| Rust engine, Python AI | Rules in one fast, testable place; ML ecosystem in Python. Throughput: millions of actions/s single-threaded. |
| Engine owns encodings (`encode.rs`) | Observation/mask computation is on the hot path and must match the rules exactly; Python only reads offsets. Versioned with `ENCODING_VERSION`. |
| Trades as first-class phases (`TradeResponse`, `TradeConfirm`) with multiple simultaneous actors | Mirrors the real table: anyone can answer or counter; the proposer chooses. The `actors()` bitmask lets both a server (concurrent humans) and a sequential RL driver (`next_actor()`) use the same engine. |
| Discrete trade templates for agents, arbitrary trades for humans | Keeps the action space small (120 templates) while humans keep full freedom; agents still respond to any offer because the offer is in the observation. |
| One card per discard action | Avoids a combinatorial discard action space; the same mechanism works for humans (UI batches them). |
| Public hand belief tracked by the engine | Agents and bots get the information a careful human would track (exact except robber steals) without peeking at hidden state. |
| Per-(env, seat) trajectory chains in PPO | Correct credit assignment in a turn-based multi-agent game where one step affects all seats. |
| Topology-biased entity transformer | The board is a fixed graph; relation biases give spatial structure cheaply and the per-entity heads map naturally onto vertex/edge/hex actions. |
| Server-side authoritative state + per-client views | Cheating-resistant online play; pass-and-play is simply one client holding several seats. |
| Python server (FastAPI) rather than Rust | AI agents (PyTorch) run in-process; the engine is already native. Simplicity beats a second service. |
| JSON-file room snapshots | Zero-dependency persistence that survives restarts; adequate for a single-node deployment. |
| SVG board with procedural art | Crisp at any resolution, no binary assets, small bundle (~86 KB gzipped JS). |

## Data flow of one human move

1. Browser sends `{"type":"action","seat":0,"action":{…}}`.
2. `RoomManager.handle` validates the message (Pydantic) and seat ownership, then
   `Room.apply_action` → `CatanGame.apply` → `GameState::apply` (`check` + transition).
3. New engine events are drained into the room log; the snapshot is written.
4. `broadcast` sends each socket its own `state` (views + legal moves + redacted log).
5. The room's bot loop wakes; if a bot seat is now an actor it sleeps for the configured
   delay, computes a move in a worker thread and applies it the same way.

## Repository layout

```
Cargo.toml                 Rust workspace
crates/catan-core/         engine (src/, tests/rules.rs, examples/bench.rs)
crates/catan-py/           pyo3 bindings
pyproject.toml             Python package (maturin backend, uv-managed)
python/catan_ai/           AI + server package
python/tests/              pytest suites
web/                       React client (Vite)
models/                    drop trained checkpoints here (served as neural bots)
docs/                      this documentation
Dockerfile, .github/       deployment and CI
```
