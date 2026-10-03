# CLAUDE.md

Guidance for AI coding sessions in this repository. Read this first, then
`docs/PROGRESS.md` (current state) and `docs/ROADMAP.md` (what's next).

## What this is

A Settlers of Catan AI **with player-to-player trading**: a Rust rules engine
(`crates/catan-core`), pyo3 bindings (`crates/catan-py` → `catan_ai._engine`), a PyTorch
PPO self-play trainer and a FastAPI multiplayer server (`python/catan_ai`), and a React
web client (`web/`) supporting vs-AI, pass & play and online play.

## Owner preferences (must follow)

- **Commits**: Conventional Commits (`feat(core): …`, `fix(server): …`, `docs: …`,
  `build: …`, `test: …`, `refactor(web): …`). Commit in small logical steps as you go.
  **Do not add `Co-Authored-By` or any AI attribution lines** to commits or PRs.
- **Python tooling is uv** (`uv sync`, `uv run …`). Never `pip install` into the project.
- **No full training runs on the dev machine.** Smoke runs only
  (`uv run catan-train --preset smoke`). The owner trains on their own GPU machine.
- Production-quality code: tests for new behaviour, docs updated in the same change.
- Update `docs/PROGRESS.md` at the end of each session (newest entry first).

## Commands

```bash
# Rust engine
cargo test -p catan-core --release
cargo clippy -p catan-core --all-targets --release -- -D warnings
cargo fmt --all
cargo run --release -p catan-core --example bench -- 3000 heuristic --encode

# Python (rebuilds the Rust extension automatically when crates/ change)
uv sync                       # server + engine
uv sync --extra train         # + torch/tensorboard (needed for training tests & neural bots)
uv run pytest -q              # python/tests (training tests skip without torch)
uv run ruff check python && uv run ruff format python

# Web
cd web && npm ci && npm run build      # typecheck + bundle into web/dist
cd web && npm run dev                  # :5173, proxies to the server on :8000

# Run
uv run catan-server --port 8000        # serves web/dist
cd web && npm run ui-smoke             # headless-Chrome E2E walkthrough against :8000 (add -- --out shots/)
uv run catan-train --preset smoke --run-dir runs/smoke
uv run catan-eval heuristic random random random --games 500
uv run catan-bench                     # fastest attention/compile flags for this GPU
```

## Where things live

| Change | Files |
|---|---|
| Game rules / legality / transitions | `crates/catan-core/src/rules.rs` (+ `state.rs`) |
| Board geometry | `topology.rs` (static), `board.rs` (layouts) |
| Trade templates for agents | `trade.rs` |
| Action space / observation | `encode.rs` → bump `ENCODING_VERSION`, update `docs/ENGINE.md` |
| Scripted bots | `bots.rs` |
| UI-facing JSON | `view.rs`, `event.rs`, mirrored in `web/src/lib/types.ts` |
| Python bindings | `crates/catan-py/src/lib.rs`, wrapper `python/catan_ai/engine.py` |
| Model / PPO / CLI | `python/catan_ai/model.py`, `ppo.py`, `train.py`, `evaluate.py`, `agents.py` |
| Server | `python/catan_ai/server/{app,rooms,protocol,ai,store}.py`, protocol in `docs/SERVER.md` |
| Client | `web/src/components/*`, theme in `web/src/styles.css` |

## Invariants & contracts

- The engine is the single source of truth for rules. Never re-implement legality in
  Python or TypeScript; use `legal_actions` / `check` / the server's `legal` lists.
- Engine JSON shapes (`Action`, `Phase`, views, events) are used by the server and the web
  client. When changing them, update `web/src/lib/types.ts` and `docs/ENGINE.md`.
- Observations/actions are **seat-relative**. Any layout change requires bumping
  `ENCODING_VERSION` (old checkpoints then refuse to load, by design).
- Hidden information: opponents' hands appear only as the public `belief`. Views/events
  must stay redacted per viewer (`view.rs`, `Event::redacted_for`, `Room._redact`).
- `tests/rules.rs::check_invariants` must keep passing for random and heuristic games;
  add targeted tests for every rules change.
- PPO credit assignment uses per-(env, seat) chains; see `docs/TRAINING.md` before touching
  `ppo.py`, and keep `test_gae_follows_seat_chains` green.

## Gotchas

- **Windows DLL lock**: a running `catan-server`/Python process locks `_engine.pyd`; stop it
  before `uv sync` or the rebuild fails with "os error 32".
- The extension is built in **release with fat LTO**; a rebuild takes 1-2 minutes. `uv run`
  triggers it automatically when Rust sources change (`tool.uv.cache-keys`).
- pyo3 converts `Vec<u8>` to Python `bytes`; return `Vec<u32>` (or similar) for lists of ints.
- `cargo build -p catan-py` alone fails (needs a Python interpreter); build through `uv sync`.
- On this Windows dev box use `py` (not `python`) for ad-hoc scripts outside the venv, and
  prefer the Write tool over Bash heredocs containing quotes.
- Torch flavours: `--extra cpu` and `--extra cu130` are mutually exclusive uv extras;
  plain `--extra train` uses PyPI torch.
- Server tests use zero bot delay and lift the rate limit (see the `fast_bots` fixture);
  scripted test clients must act only on fresh states (log `seq`) to avoid duplicate sends.
- Headless UI checks: `npm run ui-smoke` (playwright-core + locally installed Chrome; set
  `CHROME_PATH` off Windows). `?name=Foo` on `/room/CODE` skips the name prompt.
- Client → server edits must be **atomic or partial** (`set_seat`, partial `configure.settings`):
  never send a full list rebuilt from possibly stale client state.
