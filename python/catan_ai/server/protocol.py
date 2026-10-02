"""WebSocket message schemas (client -> server).

Server -> client messages are plain dicts built in :mod:`catan_ai.server.rooms`;
the full protocol is documented in ``docs/SERVER.md``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

NAME_MAX = 24


class Hello(BaseModel):
    type: Literal["hello"]
    name: str = Field(min_length=1, max_length=NAME_MAX)
    token: str | None = Field(default=None, max_length=64)


class ClaimSeat(BaseModel):
    type: Literal["claim_seat"]
    seat: int = Field(ge=0, le=3)
    # Optional per-seat display name (pass-and-play players sharing a device).
    name: str | None = Field(default=None, max_length=NAME_MAX)


class LeaveSeat(BaseModel):
    type: Literal["leave_seat"]
    seat: int = Field(ge=0, le=3)


class SeatConfig(BaseModel):
    kind: Literal["human", "bot"]
    bot: str | None = Field(default=None, max_length=64)


class RoomSettings(BaseModel):
    num_players: int = Field(default=4, ge=2, le=4)
    vp_to_win: int = Field(default=10, ge=5, le=15)
    max_trade_offers_per_turn: int = Field(default=5, ge=0, le=10)
    beginner_board: bool = False
    bot_speed: Literal["fast", "normal", "slow"] = "normal"


class Configure(BaseModel):
    type: Literal["configure"]
    settings: RoomSettings | None = None
    seats: list[SeatConfig] | None = Field(default=None, max_length=4)


class Start(BaseModel):
    type: Literal["start"]


class ActionMsg(BaseModel):
    type: Literal["action"]
    seat: int = Field(ge=0, le=3)
    action: dict[str, Any]


class Chat(BaseModel):
    type: Literal["chat"]
    text: str = Field(min_length=1, max_length=300)


class BackToLobby(BaseModel):
    type: Literal["back_to_lobby"]


class Ping(BaseModel):
    type: Literal["ping"]


ClientMessage = Annotated[
    Hello | ClaimSeat | LeaveSeat | Configure | Start | ActionMsg | Chat | BackToLobby | Ping,
    Field(discriminator="type"),
]
client_message = TypeAdapter(ClientMessage)
