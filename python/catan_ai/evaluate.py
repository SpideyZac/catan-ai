"""Head-to-head evaluation of agents.

Examples::

    uv run catan-eval heuristic random random random --games 1000
    uv run catan-eval neural:runs/main/latest.pt heuristic heuristic heuristic --games 200

Seats are rotated every game so no agent benefits from turn order. Reports win rate
with a 95% Wilson confidence interval for each agent spec.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter

from catan_ai import _engine
from catan_ai.agents import make_agent, play_game


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def evaluate(specs: list[str], games: int, seed: int = 0, max_turns: int = 500, device: str = "cpu") -> dict:
    n = len(specs)
    start = time.perf_counter()
    if all(s in ("random", "heuristic") for s in specs):
        counts = _engine.arena(specs, games, seed=seed, max_turns=max_turns)
        wins_by_index = {i: counts[i] for i in range(n)}
        truncated = counts[n]
    else:
        agents = [make_agent(s, seed=seed + i, device=device) for i, s in enumerate(specs)]
        wins_by_index = Counter()
        truncated = 0
        for g in range(games):
            shift = g % n
            seated = [agents[(seat + shift) % n] for seat in range(n)]
            game = play_game(seated, seed=seed * 1_000_003 + g, max_turns=max_turns)
            if game.winner is None:
                truncated += 1
            else:
                wins_by_index[(game.winner + shift) % n] += 1
    elapsed = time.perf_counter() - start
    results = {}
    for i, s in enumerate(specs):
        w = wins_by_index.get(i, 0)
        lo, hi = wilson(w, games)
        results[f"{i}:{s}"] = {"wins": w, "win_rate": w / games, "ci95": (lo, hi)}
    return {"games": games, "truncated": truncated, "seconds": elapsed, "agents": results}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pit Catan agents against each other.")
    p.add_argument("agents", nargs="+", help="2-4 agent specs: random | heuristic | neural:<ckpt>[@temp]")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=500)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if not 2 <= len(args.agents) <= 4:
        p.error("need 2-4 agents")
    out = evaluate(args.agents, args.games, args.seed, args.max_turns, args.device)
    print(f"{out['games']} games in {out['seconds']:.1f}s ({out['truncated']} truncated)")
    for name, r in out["agents"].items():
        lo, hi = r["ci95"]
        print(
            f"  {name:40s} {r['wins']:5d} wins  {100 * r['win_rate']:5.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]"
        )


if __name__ == "__main__":
    main()
