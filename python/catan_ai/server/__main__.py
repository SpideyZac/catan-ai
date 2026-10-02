"""Run the game server: ``uv run catan-server --port 8000``."""

from __future__ import annotations

import argparse
import logging
import os


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Catan AI multiplayer server")
    p.add_argument("--host", default=os.environ.get("CATAN_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("CATAN_PORT", "8000")))
    p.add_argument("--static-dir", help="built web client (default: bundled static/ or web/dist)")
    p.add_argument("--models-dir", help="directory of *.pt checkpoints offered as neural bots")
    p.add_argument("--data-dir", help="room snapshot directory ('' disables persistence)")
    p.add_argument("--device", help="torch device for neural bots (cpu, cuda)")
    p.add_argument("--reload", action="store_true", help="auto-reload (development)")
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)

    for flag, env in (
        ("static_dir", "CATAN_STATIC_DIR"),
        ("models_dir", "CATAN_MODELS_DIR"),
        ("data_dir", "CATAN_DATA_DIR"),
        ("device", "CATAN_DEVICE"),
    ):
        value = getattr(args, flag)
        if value is not None:
            os.environ[env] = value

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import uvicorn

    uvicorn.run(
        "catan_ai.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        ws_max_size=16_384,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
