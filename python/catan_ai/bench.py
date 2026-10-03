"""Hardware benchmark for training: which attention kernel / compile mode is fastest here?

    uv run catan-bench                       # warmup/full preset model, micro-batch 1024
    uv run catan-bench --micro-batch 2048 --model.d-model 160

Measures, on this machine:

* environment throughput (Rust ``VecEnv``, random legal actions),
* learner forward+backward throughput for each attention backend and with
  ``torch.compile``, with peak GPU memory,

and prints the ``catan-train`` flags that were fastest. Takes about a minute.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import torch

from catan_ai import _engine
from catan_ai.model import ModelConfig, build_model, masked_logits

BACKENDS = {
    "auto": None,
    "efficient": "EFFICIENT_ATTENTION",
    "cudnn": "CUDNN_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "math": "MATH",
}


def attention_context(name: str):
    """Context manager restricting scaled_dot_product_attention to one backend."""
    if BACKENDS.get(name) is None:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(getattr(SDPBackend, BACKENDS[name]))


def bench_env(num_envs: int = 256, steps: int = 200) -> float:
    env = _engine.VecEnv(num_envs, seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps):
        _, mask, _ = env.observe()
        env.step((rng.random(mask.shape) * mask).argmax(1))
    return num_envs * steps / (time.perf_counter() - t0)


def _batch(n: int, device: torch.device):
    env = _engine.VecEnv(n, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(30):  # advance past setup so observations look like mid-game
        _, mask, _ = env.observe()
        env.step((rng.random(mask.shape) * mask).argmax(1))
    obs, mask, _ = env.observe()
    act = (rng.random(mask.shape) * mask).argmax(1)
    return (
        torch.from_numpy(obs).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(act).to(device),
    )


def bench_learner(
    cfg: ModelConfig, device: torch.device, micro: int, attention: str, compiled: bool, iters: int = 10
) -> tuple[float, float]:
    """Return (samples/s for forward+backward, peak memory GiB)."""
    model = build_model(cfg).to(device)
    fwd = torch.compile(model) if compiled else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")
    obs, mask, act = _batch(micro, device)
    amp = device.type == "cuda"

    def step() -> None:
        with attention_context(attention), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, value = fwd(obs)
        logp = torch.log_softmax(masked_logits(logits.float(), mask), -1)
        loss = -logp.gather(1, act[:, None]).mean() + value.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):  # warm-up (and compilation)
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    return micro * iters / dt, peak


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark training throughput options on this machine.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--micro-batch", type=int, default=1024)
    p.add_argument("--model.d-model", dest="d_model", type=int, default=160)
    p.add_argument("--model.n-layers", dest="n_layers", type=int, default=6)
    p.add_argument("--model.n-heads", dest="n_heads", type=int, default=8)
    p.add_argument("--skip-compile", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)

    print(f"torch {torch.__version__} | device {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(device)})", end="")
    print(
        f" | model d={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} | micro-batch {args.micro_batch}"
    )
    print(f"env: {bench_env():,.0f} decisions/s (VecEnv 256, random legal actions)")

    variants = [(a, False) for a in BACKENDS] + ([] if args.skip_compile else [("auto", True)])
    results = []
    for attention, compiled in variants:
        label = f"attention={attention}" + (" + compile" if compiled else "")
        try:
            sps, peak = bench_learner(cfg, device, args.micro_batch, attention, compiled)
        except Exception as e:  # unsupported backend / no triton / OOM
            print(f"  {label:32s} unavailable: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        results.append((sps, attention, compiled))
        print(f"  {label:32s} {sps:9,.0f} samples/s fwd+bwd   peak {peak:5.2f} GiB")

    if results:
        sps, attention, compiled = max(results)
        flags = f"--attention {attention}" + (" --compile true" if compiled else "")
        print(f"\nfastest: {flags}  ({sps:,.0f} samples/s)")
        print(f"e.g. uv run catan-train --preset warmup {flags} --micro-batch-size {args.micro_batch}")


if __name__ == "__main__":
    main()
