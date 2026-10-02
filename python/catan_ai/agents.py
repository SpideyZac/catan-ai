"""Agents that pick moves for a seat in a :class:`~catan_ai.engine.CatanGame`.

Every agent implements ``act(game, player) -> action dict``. Agents only use
information visible to their seat (the engine's observation encoder enforces this
for neural agents; scripted bots read the public belief for opponents' hands).
"""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np

from catan_ai.engine import Action, CatanGame


class Agent(Protocol):
    name: str

    def act(self, game: CatanGame, player: int) -> Action: ...


class ScriptedAgent:
    """Rust-implemented scripted bot: ``"random"`` or ``"heuristic"``."""

    def __init__(self, kind: str = "heuristic", seed: int | None = None) -> None:
        if kind not in ("random", "heuristic"):
            raise ValueError(f"unknown scripted bot {kind!r}")
        self.kind = kind
        self.name = kind
        self._rng = random.Random(seed)

    def act(self, game: CatanGame, player: int) -> Action:
        return game.bot_action(player, self.kind, self._rng.getrandbits(63))


class NeuralAgent:
    """Policy network agent loaded from a training checkpoint.

    ``temperature=0`` plays greedily; higher values sample more diversely.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        temperature: float = 0.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        import torch

        from catan_ai.model import load_checkpoint, masked_logits

        self._torch = torch
        self._masked_logits = masked_logits
        self.model, payload = load_checkpoint(checkpoint, device)
        self.device = device
        self.temperature = temperature
        self.meta = {k: v for k, v in payload.items() if k not in ("state_dict",)}
        self.name = name or f"neural:{checkpoint}"
        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(seed)

    def policy(self, game: CatanGame, player: int) -> tuple[np.ndarray, float]:
        """Return the masked action distribution and value estimate for ``player``."""
        torch = self._torch
        obs = torch.from_numpy(game.observe(player)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(game.legal_mask(player)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, value = self.model(obs)
            logits = self._masked_logits(logits.float(), mask)
            t = max(self.temperature, 1e-6)
            probs = torch.softmax(logits / t, dim=-1)[0].cpu().numpy()
        return probs, float(value[0])

    def act(self, game: CatanGame, player: int) -> Action:
        torch = self._torch
        obs = torch.from_numpy(game.observe(player)).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(game.legal_mask(player)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, _ = self.model(obs)
            logits = self._masked_logits(logits.float(), mask)[0].cpu()
        if self.temperature <= 0:
            index = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            index = int(torch.multinomial(probs, 1, generator=self._gen))
        action = game.index_to_action(player, index)
        if action is None:  # pragma: no cover - mask guarantees a decodable index
            raise RuntimeError(f"policy chose undecodable action {index}")
        return action


def make_agent(spec: str, seed: int | None = None, device: str = "cpu") -> Agent:
    """Build an agent from a spec string.

    * ``random`` / ``heuristic`` - scripted Rust bots
    * ``neural:<path>[@temperature]`` - policy checkpoint, e.g. ``neural:runs/x/latest.pt@0.5``
    """
    if spec in ("random", "heuristic"):
        return ScriptedAgent(spec, seed)
    if spec.startswith("neural:"):
        body = spec[len("neural:") :]
        temperature = 0.0
        if "@" in body:
            body, t = body.rsplit("@", 1)
            temperature = float(t)
        return NeuralAgent(body, device=device, temperature=temperature, seed=seed, name=spec)
    raise ValueError(f"unknown agent spec {spec!r}")


def play_game(agents: list[Agent], seed: int | None = None, max_turns: int = 500, **game_kwargs) -> CatanGame:
    """Play one complete game, ``agents[i]`` controlling seat ``i``."""
    game = CatanGame(
        seed=seed, num_players=len(agents), max_turns=max_turns, record_events=False, **game_kwargs
    )
    while (p := game.next_actor) is not None:
        game.apply(p, agents[p].act(game, p))
    return game
