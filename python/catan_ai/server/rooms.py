"""Room/lobby management and the authoritative game loop.

A *room* has up to four seats. Each seat is either a bot (with an AI level) or a
human seat claimed by a *client* (a browser identified by a secret token stored in
localStorage). A client may claim several seats - that is pass-and-play on one
device - and several clients on different devices make an online game. Everyone
else in the room is a spectator.

All game mutations happen under the room lock; after each one every connected
socket receives a personalized ``state`` message containing only what its seats may
see (hands, dev cards and private event details of other seats are hidden).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from catan_ai.engine import CatanGame
from catan_ai.server.ai import BotRegistry
from catan_ai.server.protocol import (
    ActionMsg,
    BackToLobby,
    Chat,
    ClaimSeat,
    Configure,
    Hello,
    LeaveSeat,
    Ping,
    RoomSettings,
    Start,
    client_message,
)
from catan_ai.server.store import RoomStore

log = logging.getLogger(__name__)

SEAT_COLORS = ("red", "blue", "white", "orange")
BOT_NAMES = ("Ada", "Blaise", "Grace", "Alan")
BOT_DELAYS = {"fast": 0.2, "normal": 0.75, "slow": 1.5}
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
LOG_LIMIT = 300
CHAT_LIMIT = 100


class RoomError(Exception):
    """A client error to report back over the socket."""


@dataclass
class Seat:
    kind: str = "human"  # "human" | "bot"
    bot: str | None = None
    client_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "bot": self.bot, "client_id": self.client_id, "name": self.name}


@dataclass
class Client:
    id: str
    token: str
    name: str
    joined_at: float = field(default_factory=time.time)
    connections: int = 0


@dataclass(eq=False)
class Room:
    code: str
    settings: RoomSettings = field(default_factory=RoomSettings)
    seats: list[Seat] = field(default_factory=list)
    clients: dict[str, Client] = field(default_factory=dict)
    host_id: str | None = None
    status: str = "lobby"  # lobby | playing | finished
    game: CatanGame | None = None
    game_id: int = 0
    log: list[dict] = field(default_factory=list)
    log_seq: int = 0
    chat: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    trade_started_at: float | None = None
    # Runtime only.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    wake: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    sockets: dict[int, tuple[str, Any]] = field(default_factory=dict, repr=False)
    bot_task: asyncio.Task | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ clients
    def add_client(self, name: str, token: str | None = None) -> Client:
        if token:
            for c in self.clients.values():
                if secrets.compare_digest(c.token, token):
                    c.name = name or c.name
                    return c
        if len(self.clients) >= 32:
            raise RoomError("room is full")
        c = Client(id=secrets.token_hex(4), token=secrets.token_urlsafe(24), name=name)
        self.clients[c.id] = c
        if self.host_id is None:
            self.host_id = c.id
        return c

    def ensure_host(self) -> None:
        """Hand the host role to a connected client if the host has left."""
        host = self.clients.get(self.host_id or "")
        if host and host.connections > 0:
            return
        connected = sorted((c for c in self.clients.values() if c.connections > 0), key=lambda c: c.joined_at)
        if connected:
            self.host_id = connected[0].id

    def controlled_seats(self, client_id: str) -> list[int]:
        return [i for i, s in enumerate(self.seats) if s.kind == "human" and s.client_id == client_id]

    def seat_name(self, i: int) -> str:
        s = self.seats[i]
        if s.kind == "bot":
            return s.name or BOT_NAMES[i % len(BOT_NAMES)]
        if s.name:
            return s.name
        c = self.clients.get(s.client_id or "")
        return c.name if c else f"Seat {i + 1}"

    # ------------------------------------------------------------------ lobby
    def reset_seats(self) -> None:
        n = self.settings.num_players
        while len(self.seats) < n:
            self.seats.append(Seat(kind="bot", bot="heuristic"))
        del self.seats[n:]

    def touch(self) -> None:
        self.updated_at = time.time()

    # ------------------------------------------------------------------ game
    def start_game(self) -> None:
        unclaimed = [i for i, s in enumerate(self.seats) if s.kind == "human" and not s.client_id]
        if unclaimed:
            raise RoomError(f"seat {unclaimed[0] + 1} has no player yet")
        st = self.settings
        self.game = CatanGame(
            seed=secrets.randbits(63),
            num_players=st.num_players,
            vp_to_win=st.vp_to_win,
            max_trade_offers_per_turn=st.max_trade_offers_per_turn,
            beginner_board=st.beginner_board,
            record_events=True,
        )
        self.game_id += 1
        self.status = "playing"
        self.log = []
        self.trade_started_at = None
        self.ingest_events()

    def ingest_events(self) -> None:
        assert self.game is not None
        now = time.time()
        for ev in self.game.drain_events():
            self.log_seq += 1
            self.log.append({"seq": self.log_seq, "ts": now, "event": ev})
        del self.log[:-LOG_LIMIT]
        phase = self.game.phase
        if phase == "trade_response":
            if self.trade_started_at is None:
                self.trade_started_at = now
        else:
            self.trade_started_at = None
        if self.game.is_over and self.status == "playing":
            self.status = "finished"

    def apply_action(self, seat: int, action: dict) -> None:
        if self.game is None or self.status != "playing":
            raise RoomError("no game in progress")
        try:
            self.game.apply(seat, action)
        except ValueError as e:
            raise RoomError(str(e)) from e
        self.ingest_events()
        self.touch()

    # ------------------------------------------------------------------ views
    @staticmethod
    def _redact(ev: dict, seats: set[int]) -> dict:
        t = ev.get("type")
        if t == "dev_card_bought" and ev["player"] not in seats:
            return {**ev, "card": None}
        if t == "stolen" and ev["thief"] not in seats and ev["victim"] not in seats:
            return {**ev, "resource": None}
        return ev

    def lobby_view(self) -> dict:
        return {
            "code": self.code,
            "status": self.status,
            "host_id": self.host_id,
            "settings": self.settings.model_dump(),
            "seats": [
                {
                    "index": i,
                    "kind": s.kind,
                    "bot": s.bot,
                    "client_id": s.client_id,
                    "name": self.seat_name(i),
                    "color": SEAT_COLORS[i],
                    "connected": s.kind == "bot"
                    or (s.client_id in self.clients and self.clients[s.client_id].connections > 0),
                }
                for i, s in enumerate(self.seats)
            ],
            "clients": [
                {"id": c.id, "name": c.name, "connected": c.connections > 0}
                for c in sorted(self.clients.values(), key=lambda c: c.joined_at)
            ],
            "chat": self.chat[-50:],
            "game_id": self.game_id,
        }

    def state_for(self, client_id: str) -> dict:
        seats = self.controlled_seats(client_id)
        payload: dict[str, Any] = {
            "type": "state",
            "room": self.lobby_view(),
            "you": {"client_id": client_id, "seats": seats, "is_host": client_id == self.host_id},
            "game": None,
        }
        g = self.game
        if g is None:
            return payload
        seat_set = set(seats)
        actors = set(g.actors)
        views = {str(s): g.view(s) for s in seats} if seats else {"spectator": g.view(None)}
        legal: dict[str, list] = {}
        can_offer: dict[str, bool] = {}
        for s in seats:
            if s in actors or (g.phase == "trade_response" and g.view(s)["trade"]["proposer"] == s):
                acts = g.legal_actions(s)
                can_offer[str(s)] = any(a["type"] == "offer_trade" for a in acts)
                legal[str(s)] = [a for a in acts if a["type"] != "offer_trade"]
        payload["game"] = {
            "id": self.game_id,
            "board": g.board(),
            "views": views,
            "legal": legal,
            "can_offer": can_offer,
            "log": [{**e, "event": self._redact(e["event"], seat_set)} for e in self.log[-120:]],
            "trade_started_at": self.trade_started_at,
        }
        return payload

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "settings": self.settings.model_dump(),
            "seats": [s.to_dict() for s in self.seats],
            "clients": {
                c.id: {"token": c.token, "name": c.name, "joined_at": c.joined_at}
                for c in self.clients.values()
            },
            "host_id": self.host_id,
            "status": self.status,
            "game": self.game.to_json() if self.game else None,
            "game_id": self.game_id,
            "log": self.log,
            "log_seq": self.log_seq,
            "chat": self.chat[-CHAT_LIMIT:],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Room:
        room = cls(code=d["code"], settings=RoomSettings(**d["settings"]))
        room.seats = [Seat(**s) for s in d["seats"]]
        room.clients = {
            cid: Client(id=cid, token=c["token"], name=c["name"], joined_at=c.get("joined_at", 0.0))
            for cid, c in d["clients"].items()
        }
        room.host_id = d.get("host_id")
        room.status = d.get("status", "lobby")
        room.game = CatanGame.from_json(d["game"]) if d.get("game") else None
        room.game_id = d.get("game_id", 0)
        room.log = d.get("log", [])
        room.log_seq = d.get("log_seq", 0)
        room.chat = d.get("chat", [])
        room.created_at = d.get("created_at", time.time())
        room.updated_at = d.get("updated_at", time.time())
        return room


class RoomManager:
    def __init__(
        self,
        registry: BotRegistry,
        store: RoomStore | None = None,
        room_ttl: float = 24 * 3600,
        trade_timeout: float = 45.0,
        max_rooms: int = 1000,
    ) -> None:
        self.registry = registry
        self.store = store
        self.room_ttl = room_ttl
        self.trade_timeout = trade_timeout
        self.max_rooms = max_rooms
        self.rooms: dict[str, Room] = {}
        self._sock_seq = 0
        self._janitor: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    async def startup(self) -> None:
        if self.store:
            for data in self.store.load_all():
                try:
                    room = Room.from_dict(data)
                except Exception:
                    log.exception("could not restore room %s", data.get("code"))
                    continue
                self.rooms[room.code] = room
                if room.status == "playing":
                    self._ensure_bot_task(room)
            log.info("restored %d rooms", len(self.rooms))
        self._janitor = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        tasks = [r.bot_task for r in self.rooms.values() if r.bot_task] + (
            [self._janitor] if self._janitor else []
        )
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for room in self.rooms.values():
            self._persist(room)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            now = time.time()
            for code, room in list(self.rooms.items()):
                idle = now - room.updated_at
                if not room.sockets and (
                    idle > self.room_ttl or (room.status != "playing" and idle > 3 * 3600)
                ):
                    self.delete_room(code)

    def delete_room(self, code: str) -> None:
        room = self.rooms.pop(code, None)
        if room and room.bot_task:
            room.bot_task.cancel()
        if self.store:
            self.store.delete(code)

    def _persist(self, room: Room) -> None:
        if self.store:
            try:
                self.store.save(room.code, room.to_dict())
            except Exception:
                log.exception("failed to persist room %s", room.code)

    # ------------------------------------------------------------------ rooms
    def create_room(self, host_name: str) -> tuple[Room, Client]:
        if len(self.rooms) >= self.max_rooms:
            raise RoomError("server is at capacity, try again later")
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(5))
            if code not in self.rooms:
                break
        room = Room(code=code)
        room.reset_seats()
        host = room.add_client(host_name)
        room.seats[0] = Seat(kind="human", client_id=host.id)
        self.rooms[code] = room
        self._persist(room)
        return room, host

    def get(self, code: str) -> Room | None:
        return self.rooms.get(code.upper())

    # ------------------------------------------------------------------ sockets
    def attach(self, room: Room, client: Client, ws: Any) -> int:
        self._sock_seq += 1
        room.sockets[self._sock_seq] = (client.id, ws)
        client.connections += 1
        room.ensure_host()
        return self._sock_seq

    def detach(self, room: Room, sock_id: int) -> None:
        entry = room.sockets.pop(sock_id, None)
        if entry:
            c = room.clients.get(entry[0])
            if c:
                c.connections = max(0, c.connections - 1)
        room.ensure_host()

    async def broadcast(self, room: Room) -> None:
        dead = []
        for sid, (cid, ws) in list(room.sockets.items()):
            try:
                await ws.send_json(room.state_for(cid))
            except Exception:
                dead.append(sid)
        for sid in dead:
            self.detach(room, sid)

    # ------------------------------------------------------------------ messages
    async def handle(self, room: Room, client: Client, raw: Any) -> dict | None:
        """Process one client message. Returns a direct reply, if any."""
        try:
            msg = client_message.validate_python(raw)
        except ValidationError as e:
            raise RoomError(f"invalid message: {e.errors()[0].get('msg', 'bad format')}") from e

        if isinstance(msg, Ping):
            return {"type": "pong", "ts": time.time()}
        if isinstance(msg, Hello):
            raise RoomError("already joined")

        async with room.lock:
            is_host = client.id == room.host_id
            if isinstance(msg, ClaimSeat):
                self._claim(room, client, msg)
            elif isinstance(msg, LeaveSeat):
                if msg.seat >= len(room.seats) or room.seats[msg.seat].client_id != client.id:
                    raise RoomError("you do not hold that seat")
                if room.status == "playing":
                    raise RoomError("cannot leave a seat during a game; ask the host to hand it to a bot")
                room.seats[msg.seat].client_id = None
                room.seats[msg.seat].name = None
            elif isinstance(msg, Configure):
                if not is_host:
                    raise RoomError("only the host can change the table")
                self._configure(room, msg)
            elif isinstance(msg, Start):
                if not is_host:
                    raise RoomError("only the host can start the game")
                if room.status == "playing":
                    raise RoomError("game already running")
                room.start_game()
                self._ensure_bot_task(room)
            elif isinstance(msg, ActionMsg):
                if (
                    msg.seat >= len(room.seats)
                    or room.seats[msg.seat].kind != "human"
                    or (room.seats[msg.seat].client_id != client.id)
                ):
                    raise RoomError("you do not control that seat")
                room.apply_action(msg.seat, msg.action)
            elif isinstance(msg, Chat):
                room.chat.append(
                    {"from": client.name, "client_id": client.id, "text": msg.text, "ts": time.time()}
                )
                del room.chat[:-CHAT_LIMIT]
            elif isinstance(msg, BackToLobby):
                if not is_host:
                    raise RoomError("only the host can do that")
                room.status = "lobby"
                room.game = None
                room.log = []
            room.touch()
            self._persist(room)
        if room.status == "playing":
            self._ensure_bot_task(room)
        await self.broadcast(room)
        return None

    def _claim(self, room: Room, client: Client, msg: ClaimSeat) -> None:
        if msg.seat >= len(room.seats):
            raise RoomError("no such seat")
        seat = room.seats[msg.seat]
        if seat.kind != "human":
            raise RoomError("that seat is played by a bot")
        if seat.client_id and seat.client_id != client.id:
            owner = room.clients.get(seat.client_id)
            # Seats of disconnected players can be taken over only in the lobby.
            if room.status == "playing" or (owner and owner.connections > 0):
                raise RoomError("seat already taken")
        seat.client_id = client.id
        seat.name = msg.name.strip() if msg.name and msg.name.strip() else None

    def _configure(self, room: Room, msg: Configure) -> None:
        if msg.settings is not None:
            if room.status == "playing":
                raise RoomError("settings are locked while a game is running")
            room.settings = msg.settings
            room.reset_seats()
        if msg.seats is not None:
            if len(msg.seats) != len(room.seats):
                raise RoomError(f"expected {len(room.seats)} seats")
            for i, cfg in enumerate(msg.seats):
                seat = room.seats[i]
                if cfg.kind == "bot":
                    level = cfg.bot or "heuristic"
                    if not self.registry.is_valid(level):
                        raise RoomError(f"unknown bot level {level!r}")
                    room.seats[i] = Seat(kind="bot", bot=level)
                elif seat.kind == "bot":
                    room.seats[i] = Seat(kind="human")
        if room.status == "playing":
            self._ensure_bot_task(room)

    # ------------------------------------------------------------------ bots
    def _ensure_bot_task(self, room: Room) -> None:
        if room.bot_task is None or room.bot_task.done():
            room.bot_task = asyncio.create_task(self._bot_loop(room), name=f"bots-{room.code}")
        room.wake.set()

    def _bot_actor(self, room: Room) -> int | None:
        g = room.game
        if g is None or room.status != "playing" or g.is_over:
            return None
        for p in g.actors:
            if room.seats[p].kind == "bot":
                return p
        # A bot waiting on humans for too long withdraws its trade offer.
        if (
            g.phase == "trade_response"
            and room.trade_started_at is not None
            and time.time() - room.trade_started_at > self.trade_timeout
        ):
            proposer = g.current
            if room.seats[proposer].kind == "bot":
                return proposer
        return None

    async def _bot_loop(self, room: Room) -> None:
        while room.status == "playing":
            try:
                await self._bot_step(room)
            except asyncio.CancelledError:
                raise
            except BaseException:  # includes pyo3 PanicException; never let a table stall
                log.exception("bot loop error in room %s", room.code)
                await asyncio.sleep(1.0)

    async def _bot_step(self, room: Room) -> None:
        """Wait for a bot decision (or a wake-up), then play one bot move."""
        room.wake.clear()
        p = self._bot_actor(room)
        if p is None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(room.wake.wait(), timeout=2.0)
            return
        await asyncio.sleep(BOT_DELAYS.get(room.settings.bot_speed, 0.75))
        async with room.lock:
            if self._bot_actor(room) != p or room.game is None:
                return
            g = room.game
            level = room.seats[p].bot or "heuristic"
            if g.phase == "trade_response" and p not in g.actors:
                action = {"type": "cancel_trade"}
            else:
                snapshot = g.copy()
                loop = asyncio.get_running_loop()
                action = await loop.run_in_executor(None, self.registry.choose, level, snapshot, p)
            try:
                room.apply_action(p, action)
            except RoomError:
                log.exception("bot %s produced an illegal action %s", level, action)
                room.apply_action(p, g.bot_action(p, "heuristic", secrets.randbits(32)))
            self._persist(room)
        await self.broadcast(room)
