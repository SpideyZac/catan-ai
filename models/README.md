# Models

Drop trained checkpoints (`*.pt`, produced by `catan-train`) here. The game server
lists every checkpoint as an AI level named `Neural (<file name>)`.

```bash
cp runs/main/latest.pt models/main.pt
uv run catan-server            # now offers "Neural (main)" in the lobby
```

Checkpoints are ignored by git. Neural bots need the `train` extra (`uv sync --extra train`)
or, in Docker, `--build-arg WITH_TORCH=1`.
