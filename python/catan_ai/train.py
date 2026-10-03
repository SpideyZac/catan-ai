"""Command-line entry point for PPO self-play training.

Examples::

    # Quick smoke test (CPU friendly, a few minutes)
    uv run catan-train --preset smoke --run-dir runs/smoke

    # Full training on a GPU: warm-up against heuristic bots, then self-play
    uv run catan-train --preset warmup --run-dir runs/warmup
    uv run catan-train --preset selfplay --run-dir runs/main --resume runs/warmup/final.pt

    # Override anything, including model fields
    uv run catan-train --preset full --num-envs 512 --model.d-model 192 --model.n-layers 6

    # Resume
    uv run catan-train --config runs/main/config.json --resume runs/main/latest.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing

from catan_ai.model import ModelConfig
from catan_ai.ppo import TrainConfig, Trainer

PRESETS: dict[str, dict] = {
    # Tiny run to verify the pipeline end to end.
    "smoke": {
        "num_envs": 32,
        "rollout_steps": 64,
        "total_updates": 5,
        "minibatch_size": 512,
        "epochs": 1,
        "eval_every": 5,
        "eval_games": 16,
        "checkpoint_every": 5,
        "snapshot_every": 2,
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 4},
        "compile": False,
    },
    # Bootstrap against heuristic bots first (fast early signal), then switch to "full".
    "warmup": {
        "num_envs": 256,
        "rollout_steps": 128,
        "total_updates": 3000,
        "bot_kind": "heuristic",
        "bot_seats": 2,
        "pool_prob": 0.0,
        "vp_reward_scale": 0.05,
        # Same architecture as "full" so the warm-up checkpoint can be resumed there.
        "model": {"d_model": 160, "n_layers": 6, "n_heads": 5},
    },
    # Main self-play run with a league of past snapshots.
    "full": {
        "num_envs": 512,
        "rollout_steps": 128,
        "total_updates": 20000,
        "minibatch_size": 8192,
        "pool_prob": 0.3,
        "model": {"d_model": 160, "n_layers": 6, "n_heads": 5},
    },
}
# Continuation of a warm-up checkpoint: "full" with a gentler start (the warm-up ends at
# LR 3e-5 / entropy 0.002; starting at 3e-4 / 0.01 undoes it) and a heuristic bot in half
# the games as a fixed anchor, so self-play cannot drift into forgetting how to beat it.
PRESETS["selfplay"] = {
    **PRESETS["full"],
    "lr": 1e-4,
    "entropy_coef": 0.003,
    "entropy_final_coef": 0.001,
    "bot_kind": "heuristic",
    "bot_seats": 0.5,
}


def _field_type(f: dataclasses.Field) -> typing.Any:
    t = f.type
    if isinstance(t, str):
        t = {"int": int, "float": float, "str": str, "bool": bool, "str | None": str, "int | None": int}.get(
            t, str
        )
    return t


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _str2bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Catan agent with PPO self-play.")
    p.add_argument("--preset", choices=sorted(PRESETS), help="start from a named preset")
    p.add_argument("--config", help="JSON config file (e.g. a previous run's config.json)")
    p.add_argument("--resume", help="checkpoint to resume from")
    for f in dataclasses.fields(TrainConfig):
        if f.name == "model":
            continue
        if f.name == "player_counts":
            p.add_argument("--player-counts", dest=f.name, type=_int_list, help="e.g. 2,3,4")
            continue
        t = _field_type(f)
        p.add_argument(
            f"--{f.name.replace('_', '-')}", dest=f.name, type=_str2bool if t is bool else t, default=None
        )
    for f in dataclasses.fields(ModelConfig):
        if f.name == "hidden":
            p.add_argument(
                "--model.hidden", dest="model.hidden", type=lambda s: [int(x) for x in s.split(",")]
            )
            continue
        t = _field_type(f)
        p.add_argument(f"--model.{f.name.replace('_', '-')}", dest=f"model.{f.name}", type=t, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    data: dict = {}
    if args.preset:
        data = json.loads(json.dumps(PRESETS[args.preset]))
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            file_data = json.load(f)
        model = {**data.get("model", {}), **file_data.pop("model", {})}
        data.update(file_data)
        data["model"] = model
    model = data.setdefault("model", {})
    for k, v in vars(args).items():
        if v is None or k in ("preset", "config", "resume"):
            continue
        if k.startswith("model."):
            model[k[len("model.") :]] = v
        else:
            data[k] = v
    return TrainConfig.from_dict(data)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    Trainer(cfg, resume=args.resume).train()


if __name__ == "__main__":
    main()
