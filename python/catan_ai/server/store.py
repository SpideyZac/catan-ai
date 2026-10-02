"""Durable room snapshots so games survive server restarts."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re

log = logging.getLogger(__name__)
_CODE_RE = re.compile(r"^[A-Z0-9]{3,12}$")


class RoomStore:
    """One JSON file per room, written atomically."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, code: str) -> str:
        if not _CODE_RE.match(code):
            raise ValueError(f"invalid room code {code!r}")
        return os.path.join(self.directory, f"{code}.json")

    def save(self, code: str, data: dict) -> None:
        path = self._path(code)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)

    def delete(self, code: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._path(code))

    def load_all(self) -> list[dict]:
        out = []
        for name in os.listdir(self.directory):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.directory, name), encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                log.exception("skipping unreadable room file %s", name)
        return out
