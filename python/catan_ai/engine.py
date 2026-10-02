"""Pythonic wrapper around the Rust engine (``catan_ai._engine``).

The native ``Game`` speaks JSON strings so the Rust/Python boundary stays simple and
versionable. This module converts to and from plain Python dicts.

Action dicts look like ``{"type": "build_road", "edge": 12}``; see ``docs/ENGINE.md``
for every action type.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from catan_ai import _engine

ACTION_SIZE: int = _engine.ACTION_SIZE
OBS_SIZE: int = _engine.OBS_SIZE
ENCODING_VERSION: int = _engine.ENCODING_VERSION
ACTION_OFFSETS: dict[str, int] = dict(_engine.ACTION_OFFSETS)
OBS_OFFSETS: dict[str, tuple[int, int]] = {name: (off, width) for name, off, width in _engine.OBS_OFFSETS}

RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
DEV_CARDS = ("knight", "victory_point", "road_building", "year_of_plenty", "monopoly")

Action = dict[str, Any]


class CatanGame:
    """A single game. Thin, allocation-cheap wrapper over the native engine."""

    __slots__ = ("_g",)

    def __init__(
        self,
        seed: int | None = None,
        num_players: int = 4,
        vp_to_win: int = 10,
        max_trade_offers_per_turn: int = 3,
        max_turns: int = 0,
        beginner_board: bool = False,
        record_events: bool = True,
        discard_limit: int = 7,
        max_trade_cards: int = 6,
        _native: _engine.Game | None = None,
    ) -> None:
        self._g = _native or _engine.Game(
            seed=seed,
            num_players=num_players,
            vp_to_win=vp_to_win,
            max_trade_offers_per_turn=max_trade_offers_per_turn,
            max_turns=max_turns,
            beginner_board=beginner_board,
            record_events=record_events,
            discard_limit=discard_limit,
            max_trade_cards=max_trade_cards,
        )

    # -- persistence -------------------------------------------------------------
    @classmethod
    def from_json(cls, s: str) -> CatanGame:
        return cls(_native=_engine.Game.from_json(s))

    def to_json(self) -> str:
        return self._g.to_json()

    def copy(self) -> CatanGame:
        return CatanGame(_native=self._g.copy())

    @property
    def native(self) -> _engine.Game:
        return self._g

    # -- state -------------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self._g.phase

    @property
    def current(self) -> int:
        return self._g.current

    @property
    def turn(self) -> int:
        return self._g.turn

    @property
    def num_players(self) -> int:
        return self._g.num_players

    @property
    def winner(self) -> int | None:
        return self._g.winner

    @property
    def is_over(self) -> bool:
        return self._g.is_over

    @property
    def actors(self) -> list[int]:
        return self._g.actors

    @property
    def next_actor(self) -> int | None:
        return self._g.next_actor

    def public_vp(self, player: int) -> int:
        return self._g.public_vp(player)

    def total_vp(self, player: int) -> int:
        return self._g.total_vp(player)

    # -- actions -----------------------------------------------------------------
    def legal_actions(self, player: int) -> list[Action]:
        return json.loads(self._g.legal_actions_json(player))

    def check(self, player: int, action: Action) -> str | None:
        """Return an error message if ``action`` is illegal, else ``None``."""
        return self._g.check_json(player, json.dumps(action))

    def apply(self, player: int, action: Action) -> None:
        """Apply an action; raises ``ValueError`` if it is illegal."""
        self._g.apply_json(player, json.dumps(action))

    def apply_index(self, player: int, index: int) -> None:
        self._g.apply_index(player, int(index))

    def action_to_index(self, player: int, action: Action) -> int | None:
        return self._g.action_to_index(player, json.dumps(action))

    def index_to_action(self, player: int, index: int) -> Action | None:
        s = self._g.index_to_action_json(player, int(index))
        return None if s is None else json.loads(s)

    def bot_action(self, player: int, kind: str = "heuristic", seed: int = 0) -> Action:
        return json.loads(self._g.bot_action_json(player, kind, seed))

    def force_roll(self, d1: int, d2: int) -> None:
        self._g.force_roll(d1, d2)

    # -- encodings ---------------------------------------------------------------
    def observe(self, player: int) -> np.ndarray:
        return self._g.observe(player)

    def legal_mask(self, player: int) -> np.ndarray:
        return self._g.legal_mask(player)

    # -- views -------------------------------------------------------------------
    def view(self, viewer: int | None = None) -> dict[str, Any]:
        return json.loads(self._g.view_json(viewer))

    def board(self) -> dict[str, Any]:
        return json.loads(self._g.board_json())

    def drain_events(self) -> list[dict[str, Any]]:
        return json.loads(self._g.drain_events_json())

    @staticmethod
    def redact_events(events: list[dict[str, Any]], viewer: int | None) -> list[dict[str, Any]]:
        return json.loads(_engine.Game.redact_events_json(json.dumps(events), viewer))

    def __repr__(self) -> str:
        return repr(self._g)
