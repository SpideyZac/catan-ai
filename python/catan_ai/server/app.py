"""FastAPI application: REST endpoints, the game WebSocket and the static web client."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from catan_ai.server.ai import BotRegistry
from catan_ai.server.protocol import NAME_MAX, Hello
from catan_ai.server.rooms import RoomError, RoomManager
from catan_ai.server.store import RoomStore

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 8_192
RATE_LIMIT = (40, 10.0)  # messages per window (seconds)


def _default_static_dir() -> str:
    here = Path(__file__).resolve().parent
    for candidate in (here / "static", here.parents[2] / "web" / "dist"):
        if (candidate / "index.html").exists():
            return str(candidate)
    return str(here / "static")


@dataclass
class ServerSettings:
    static_dir: str = field(
        default_factory=lambda: os.environ.get("CATAN_STATIC_DIR") or _default_static_dir()
    )
    models_dir: str | None = field(default_factory=lambda: os.environ.get("CATAN_MODELS_DIR", "models"))
    data_dir: str | None = field(default_factory=lambda: os.environ.get("CATAN_DATA_DIR", "data/rooms"))
    device: str = field(default_factory=lambda: os.environ.get("CATAN_DEVICE", "cpu"))
    cors_origins: list[str] = field(
        default_factory=lambda: [o for o in os.environ.get("CATAN_CORS_ORIGINS", "").split(",") if o]
    )
    trade_timeout: float = field(default_factory=lambda: float(os.environ.get("CATAN_TRADE_TIMEOUT", "45")))


class CreateRoom(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_MAX)


class RateLimiter:
    def __init__(self, limit: int, window: float) -> None:
        self.limit, self.window = limit, window
        self.stamps: list[float] = []

    def allow(self) -> bool:
        now = time.monotonic()
        self.stamps = [t for t in self.stamps if now - t < self.window]
        if len(self.stamps) >= self.limit:
            return False
        self.stamps.append(now)
        return True


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    registry = BotRegistry(settings.models_dir, device=settings.device)
    store = RoomStore(settings.data_dir) if settings.data_dir else None
    manager = RoomManager(registry, store, trade_timeout=settings.trade_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.startup()
        yield
        await manager.shutdown()

    app = FastAPI(title="Catan AI", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
        )

    # ------------------------------------------------------------------ REST
    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "rooms": len(manager.rooms)}

    @app.get("/api/bots")
    async def bots() -> list[dict]:
        return [lvl.__dict__ for lvl in registry.levels()]

    @app.post("/api/rooms")
    async def create_room(body: CreateRoom) -> dict:
        try:
            room, host = manager.create_room(body.name.strip())
        except RoomError as e:
            raise HTTPException(503, str(e)) from e
        return {"code": room.code, "client_id": host.id, "token": host.token}

    @app.get("/api/rooms/{code}")
    async def room_info(code: str) -> dict:
        room = manager.get(code)
        if room is None:
            raise HTTPException(404, "room not found")
        view = room.lobby_view()
        view.pop("chat", None)
        return view

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws/{code}")
    async def game_socket(ws: WebSocket, code: str) -> None:
        room = manager.get(code)
        await ws.accept()
        if room is None:
            await ws.send_json({"type": "error", "message": "room not found", "fatal": True})
            await ws.close(code=4404)
            return
        try:
            first = await asyncio.wait_for(ws.receive_text(), timeout=15)
            hello = Hello.model_validate_json(first)
            async with room.lock:
                client = room.add_client(hello.name.strip(), hello.token)
        except (TimeoutError, ValueError, RoomError) as e:
            with contextlib.suppress(Exception):
                await ws.send_json({"type": "error", "message": f"handshake failed: {e}", "fatal": True})
                await ws.close(code=4400)
            return
        except WebSocketDisconnect:
            return

        sock_id = manager.attach(room, client, ws)
        limiter = RateLimiter(*RATE_LIMIT)
        try:
            await ws.send_json({"type": "welcome", "client_id": client.id, "token": client.token})
            await manager.broadcast(room)
            while True:
                text = await ws.receive_text()
                if len(text) > MAX_MESSAGE_BYTES:
                    await ws.send_json({"type": "error", "message": "message too large"})
                    continue
                if not limiter.allow():
                    await ws.send_json({"type": "error", "message": "slow down"})
                    continue
                try:
                    reply = await manager.handle(room, client, json.loads(text))
                    if reply:
                        await ws.send_json(reply)
                except (RoomError, json.JSONDecodeError) as e:
                    await ws.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("socket error in room %s", room.code)
        finally:
            manager.detach(room, sock_id)
            with contextlib.suppress(Exception):
                await manager.broadcast(room)

    # ------------------------------------------------------------------ static client
    static_dir = Path(settings.static_dir)
    if (static_dir / "index.html").exists():
        assets = static_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith(("api/", "ws/")):
                raise HTTPException(404)
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        async def no_client():
            return JSONResponse(
                {"message": "web client not built; run `npm run build` in web/ (see docs/SERVER.md)"}, 200
            )

    return app
