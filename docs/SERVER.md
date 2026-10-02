# Game server & protocol (`python/catan_ai/server`)

A FastAPI app that hosts tables ("rooms"), runs the authoritative engine, drives AI
seats and serves the web client.

| File | Role |
|---|---|
| `app.py` | `create_app(settings)`: REST endpoints, `/ws/{code}` WebSocket, static SPA, rate/size limits |
| `rooms.py` | `Room` (seats, clients, game, log, chat), `RoomManager` (message handling, broadcast, bot loop, persistence, cleanup) |
| `protocol.py` | Pydantic schemas for client → server messages |
| `ai.py` | `BotRegistry`: AI levels (`random`, `heuristic`, `neural:<model>` from `models/`) |
| `store.py` | `RoomStore`: one JSON snapshot per room, written atomically |
| `__main__.py` | `catan-server` CLI (uvicorn) |

## Running

```bash
cd web && npm ci && npm run build && cd ..        # build the client into web/dist
uv run catan-server --port 8000                   # http://localhost:8000
```

Development with hot reload on both sides:

```bash
uv run catan-server --port 8000 --reload          # terminal 1
cd web && npm run dev                             # terminal 2 -> http://localhost:5173 (proxies /api, /ws)
```

Configuration (CLI flag or environment variable):

| Env var | Flag | Default | Meaning |
|---|---|---|---|
| `CATAN_HOST` / `CATAN_PORT` | `--host` / `--port` | `0.0.0.0` / `8000` | Bind address |
| `CATAN_STATIC_DIR` | `--static-dir` | bundled `static/` or `web/dist` | Built web client |
| `CATAN_MODELS_DIR` | `--models-dir` | `models` | `*.pt` checkpoints offered as neural bots |
| `CATAN_DATA_DIR` | `--data-dir` | `data/rooms` | Room snapshots (`""` disables persistence) |
| `CATAN_DEVICE` | `--device` | `cpu` | Torch device for neural bots |
| `CATAN_TRADE_TIMEOUT` | – | `45` | Seconds before a bot withdraws an offer humans haven't answered |
| `CATAN_CORS_ORIGINS` | – | empty | Comma-separated origins if the client is hosted elsewhere |

### Docker

```bash
docker build -t catan-ai .                         # add --build-arg WITH_TORCH=1 for neural bots
docker run -p 8000:8000 -v catan-data:/data -v $PWD/models:/models catan-ai
```

The server is a single process holding rooms in memory (snapshotted to disk after every
change). Run one replica behind a TLS-terminating reverse proxy that supports WebSockets
(nginx, Caddy, Traefik). Horizontal scaling would need sticky routing by room code; see
the roadmap.

## Concepts

* **Room**: a table identified by a 5-character code. Has up to 4 seats, settings, a chat,
  an event log and (once started) a game.
* **Client**: a browser identity in a room: public `client_id` + secret `token` (kept in
  `localStorage` as `catan.token.<CODE>`). Reconnecting with the token restores the identity
  and its seats.
* **Seat**: `kind = "bot"` (with an AI level) or `"human"` (claimed by a client).
  A client holding **several** seats is pass-and-play on one device; seats held by
  different clients make an online game. Clients with no seat are spectators.
* **Host**: the room creator; if they disconnect, the longest-connected client becomes host.
  Only the host configures seats/settings, starts games and returns to the lobby.

## REST

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/api/health` | `{ok, rooms}` |
| `GET` | `/api/bots` | `[{id, name, description}]` |
| `POST` | `/api/rooms` | `{name}` → `{code, client_id, token}` (creator is host and sits in seat 0) |
| `GET` | `/api/rooms/{code}` | Lobby view without chat, 404 if unknown |

## WebSocket `/ws/{code}`

The first client message must be `hello`; the server answers `welcome` and then pushes a
`state` message to every socket in the room after each change.

### Client → server

```jsonc
{"type": "hello", "name": "Ada", "token": "…optional…"}
{"type": "claim_seat", "seat": 1, "name": "Kid"}   // name: optional per-seat name (pass & play)
{"type": "leave_seat", "seat": 1}                  // lobby only
{"type": "configure", "settings": {"num_players": 4, "vp_to_win": 10, "max_trade_offers_per_turn": 5,
                                   "beginner_board": false, "bot_speed": "normal"}}   // host, lobby only
{"type": "configure", "seats": [{"kind": "human"}, {"kind": "bot", "bot": "heuristic"}, …]}  // host; allowed mid-game (e.g. hand a seat to a bot)
{"type": "start"}                                  // host
{"type": "action", "seat": 0, "action": {"type": "build_road", "edge": 30}}   // engine action JSON, see ENGINE.md
{"type": "chat", "text": "anyone have wood?"}
{"type": "back_to_lobby"}                          // host
{"type": "ping"}
```

### Server → client

```jsonc
{"type": "welcome", "client_id": "…", "token": "…"}
{"type": "error", "message": "…", "fatal": false}
{"type": "pong", "ts": 0}
{"type": "state",
 "room": {"code", "status": "lobby|playing|finished", "host_id", "settings", "game_id",
          "seats": [{"index", "kind", "bot", "client_id", "name", "color", "connected"}],
          "clients": [{"id", "name", "connected"}], "chat": [{"from", "client_id", "text", "ts"}]},
 "you": {"client_id", "seats": [0, 1], "is_host": true},
 "game": null | {
    "id": 3,
    "board": {…board_view…},
    "views": {"0": {…game_view for seat 0…}, "1": {…}} | {"spectator": {…}},
    "legal": {"0": [actions…]},          // only for your seats that must act; offer templates removed
    "can_offer": {"0": true},
    "log": [{"seq", "ts", "event"}],     // last 120 events, redacted for your seats
    "trade_started_at": 1700000000.0 | null
 }}
```

## Security & robustness

* The server is authoritative; every action is validated by the engine (`GameState::check`).
* A client can only act for seats it holds; views and logs are filtered per client
  (other hands, dev cards and private steal details are never sent).
* Message size limit 8 KB, 120 messages / 10 s per socket, 15 s handshake timeout,
  Pydantic validation of every message, room code format checked before touching disk.
* Bot moves run in a thread pool against a copy of the game; an illegal bot move falls
  back to the heuristic bot; neural model failures fall back to the heuristic bot.
* Rooms idle for 24 h (3 h if not playing) with no sockets are deleted; snapshots are
  restored on startup and bot loops resume.

## Tests

`python/tests/test_server.py` covers room creation/join, a full all-bot game over the
socket, a human seat vs bots with hidden-information checks, spectator restrictions,
pass-and-play seat claiming and token reconnect, host-only operations, invalid messages
and persistence across a restart.
