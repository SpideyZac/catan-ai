"""AI opponents available to the game server."""

from __future__ import annotations

import logging
import os
import random
import threading
from dataclasses import dataclass

from catan_ai.agents import Agent, NeuralAgent, ScriptedAgent
from catan_ai.engine import Action, CatanGame

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BotLevel:
    id: str
    name: str
    description: str


BUILTIN_LEVELS = [
    BotLevel("random", "Beginner", "Plays random legal moves. Good for learning the interface."),
    BotLevel("heuristic", "Standard", "Rule-based strategist that builds, trades and robs sensibly."),
]


class BotRegistry:
    """Resolves bot level ids to agents. Neural models are discovered in ``models_dir``
    (``*.pt`` checkpoints) and loaded lazily on first use."""

    def __init__(self, models_dir: str | None = None, device: str = "cpu", temperature: float = 0.0) -> None:
        self.models_dir = models_dir
        self.device = device
        self.temperature = temperature
        self._neural: dict[str, NeuralAgent] = {}
        self._lock = threading.Lock()
        self._torch_ok: bool | None = None

    def _torch_available(self) -> bool:
        if self._torch_ok is None:
            try:
                import torch  # noqa: F401

                self._torch_ok = True
            except ImportError:
                self._torch_ok = False
        return self._torch_ok

    def neural_models(self) -> list[str]:
        if not self.models_dir or not os.path.isdir(self.models_dir) or not self._torch_available():
            return []
        return sorted(f[:-3] for f in os.listdir(self.models_dir) if f.endswith(".pt"))

    def levels(self) -> list[BotLevel]:
        out = list(BUILTIN_LEVELS)
        for name in self.neural_models():
            out.append(BotLevel(f"neural:{name}", f"Neural ({name})", "Self-play trained policy network."))
        return out

    def is_valid(self, level: str) -> bool:
        return any(lvl.id == level for lvl in self.levels())

    def _agent(self, level: str) -> Agent:
        if level in ("random", "heuristic"):
            return ScriptedAgent(level, random.getrandbits(32))
        if level.startswith("neural:"):
            name = level.split(":", 1)[1]
            with self._lock:
                if name not in self._neural:
                    path = os.path.join(self.models_dir or "", f"{name}.pt")
                    log.info("loading neural model %s", path)
                    self._neural[name] = NeuralAgent(
                        path, device=self.device, temperature=self.temperature, name=level
                    )
                return self._neural[name]
        raise ValueError(f"unknown bot level {level!r}")

    def choose(self, level: str, game: CatanGame, player: int) -> Action:
        """Pick a move. Falls back to the heuristic bot if a neural model fails."""
        try:
            return self._agent(level).act(game, player)
        except Exception:
            if level == "heuristic":
                raise
            log.exception("bot level %s failed; falling back to heuristic", level)
            return game.bot_action(player, "heuristic", random.getrandbits(32))
