import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_ai.agents import NeuralAgent, ScriptedAgent, play_game  # noqa: E402
from catan_ai.engine import ACTION_SIZE, OBS_SIZE, CatanGame  # noqa: E402
from catan_ai.model import ModelConfig, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from catan_ai.ppo import RolloutBuffer, TrainConfig, Trainer  # noqa: E402


@pytest.mark.parametrize("arch", ["entity_transformer", "mlp"])
def test_model_shapes(arch):
    model = build_model(ModelConfig(arch=arch, d_model=32, n_layers=1, n_heads=2, hidden=[64]))
    g = CatanGame(seed=0)
    obs = torch.from_numpy(np.stack([g.observe(0), g.observe(1)]))
    logits, value = model(obs)
    assert logits.shape == (2, ACTION_SIZE)
    assert value.shape == (2,)


def test_gae_follows_seat_chains():
    buf = RolloutBuffer(16, torch.device("cpu"))
    # Two chains interleaved: A = 0 -> 2 -> 4 (terminal), B = 1 -> 3 (bootstrapped).
    for t in range(5):
        buf.add(np.zeros((1, OBS_SIZE)), np.ones((1, ACTION_SIZE), bool), [0], [0.0], [0.5], t)
    buf.next_index[[0, 2, 1]] = [2, 4, 3]
    buf.done[4] = True
    buf.reward[4] = 1.0
    buf.bootstrap[3] = 0.25
    gamma, lam = 0.9, 0.8
    adv, ret = buf.compute_gae(gamma, lam)
    d4 = 1.0 - 0.5
    d2 = 0 + gamma * 0.5 - 0.5
    d0 = d2
    assert adv[4] == pytest.approx(d4)
    assert adv[2] == pytest.approx(d2 + gamma * lam * d4)
    assert adv[0] == pytest.approx(d0 + gamma * lam * adv[2])
    d3 = gamma * 0.25 - 0.5
    assert adv[3] == pytest.approx(d3)
    assert adv[1] == pytest.approx((gamma * 0.5 - 0.5) + gamma * lam * d3)
    assert ret == pytest.approx(adv + 0.5)


def test_trainer_smoke_and_checkpoint(tmp_path):
    cfg = TrainConfig(
        run_dir=str(tmp_path),
        num_envs=8,
        rollout_steps=16,
        total_updates=2,
        minibatch_size=64,
        epochs=1,
        eval_every=2,
        eval_games=4,
        checkpoint_every=1,
        snapshot_every=1,
        pool_prob=0.5,
        device="cpu",
        model=ModelConfig(d_model=32, n_layers=1, n_heads=2),
        compile=False,
    )
    trainer = Trainer(cfg)
    trainer.train()
    assert (tmp_path / "final.pt").exists()
    model, payload = load_checkpoint(str(tmp_path / "final.pt"))
    assert payload["update"] == 2
    # Resume picks up the update counter.
    cfg.total_updates = 3
    t2 = Trainer(cfg, resume=str(tmp_path / "latest.pt"))
    assert t2.update == 2


def test_neural_agent_plays_full_game(tmp_path):
    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2)
    path = tmp_path / "m.pt"
    save_checkpoint(str(path), build_model(cfg), cfg)
    agent = NeuralAgent(str(path), temperature=1.0, seed=0)
    g = play_game(
        [agent, ScriptedAgent("heuristic", 1), ScriptedAgent("heuristic", 2)], seed=3, max_turns=150
    )
    assert g.is_over


def _tiny_trainer(tmp_path, **overrides):
    params = {
        "run_dir": str(tmp_path),
        "num_envs": 8,
        "rollout_steps": 16,
        "total_updates": 1,
        "minibatch_size": 64,
        "epochs": 1,
        "eval_every": 0,
        "pool_prob": 0.0,
        "device": "cpu",
        "model": ModelConfig(d_model=32, n_layers=1, n_heads=2),
        "compile": False,
    }
    return Trainer(TrainConfig(**{**params, **overrides}))


def _minibatch(trainer, size=48):
    trainer.collect()
    buf = trainer.buffer
    adv, ret = buf.compute_gae(0.99, 0.95)
    idx = slice(0, size)
    a = torch.from_numpy(adv[idx])
    return (
        torch.from_numpy(buf.obs[idx]),
        torch.from_numpy(buf.mask[idx]),
        torch.from_numpy(buf.action[idx]),
        torch.from_numpy(buf.logp[idx]),
        torch.from_numpy(buf.value[idx]),
        (a - a.mean()) / (a.std() + 1e-8),
        torch.from_numpy(ret[idx]),
    )


def test_micro_batching_matches_single_pass_gradients(tmp_path):
    trainer = _tiny_trainer(tmp_path)
    tensors = _minibatch(trainer)
    trainer.micro_batch_size = 48
    full = trainer._accumulate_gradients(tensors, 0.01)
    g_full = [p.grad.clone() for p in trainer.model.parameters() if p.grad is not None]
    trainer.micro_batch_size = 7  # uneven split on purpose
    split = trainer._accumulate_gradients(tensors, 0.01)
    g_split = [p.grad.clone() for p in trainer.model.parameters() if p.grad is not None]
    for a, b in zip(g_full, g_split, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    for k in full:
        assert full[k] == pytest.approx(split[k], abs=1e-5)


def test_out_of_memory_halves_micro_batch(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, micro_batch_size=64)
    tensors = _minibatch(trainer)
    forward = trainer.model.forward

    def limited(obs):
        if obs.shape[0] > 12:
            raise torch.OutOfMemoryError("simulated")
        return forward(obs)

    monkeypatch.setattr(trainer.model, "forward", limited)
    trainer._accumulate_gradients(tensors, 0.01)
    assert trainer.micro_batch_size == 8


def test_road_logits_are_local_to_their_edge():
    from catan_ai.engine import ACTION_OFFSETS, OBS_OFFSETS
    from catan_ai.model import EntityTransformer, edge_endpoints

    torch.manual_seed(0)
    model = EntityTransformer(ModelConfig(d_model=32, n_layers=0, n_heads=2)).eval()
    obs = torch.from_numpy(CatanGame(seed=1).observe(0)).unsqueeze(0)
    base, _ = model(obs)
    # With no attention layers, changing edge 10's features may only move road logits of
    # edges sharing an endpoint with it (through the vertex aggregation) - never far ones.
    off, width = OBS_OFFSETS["edge"]
    bumped = obs.clone()
    bumped[0, off + 10 * width : off + 11 * width] += 1.0
    out, _ = model(bumped)
    road = slice(ACTION_OFFSETS["road"], ACTION_OFFSETS["road"] + 72)
    changed = set(torch.nonzero((out[0, road] - base[0, road]).abs() > 1e-6).flatten().tolist())
    ends = edge_endpoints()
    near = {e for e in range(72) if set(ends[e]) & set(ends[10])}
    assert 10 in changed
    assert changed <= near


def test_old_model_checkpoints_are_rejected(tmp_path):
    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=2)
    path = tmp_path / "old.pt"
    save_checkpoint(str(path), build_model(cfg), cfg)
    payload = torch.load(path, weights_only=False)
    payload["model_version"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model v1"):
        load_checkpoint(str(path))


def test_attention_backend_option(tmp_path):
    trainer = _tiny_trainer(tmp_path, attention="math")
    tensors = _minibatch(trainer)
    stats = trainer._accumulate_gradients(tensors, 0.01)
    assert all(np.isfinite(v) for v in stats.values())
    with pytest.raises(ValueError, match="attention"):
        _tiny_trainer(tmp_path, attention="bogus")


def test_bench_cli_runs(capsys):
    from catan_ai.bench import main

    main(
        [
            "--device",
            "cpu",
            "--micro-batch",
            "16",
            "--model.d-model",
            "32",
            "--model.n-layers",
            "1",
            "--model.n-heads",
            "2",
            "--skip-compile",
        ]
    )
    out = capsys.readouterr().out
    assert "decisions/s" in out and "fastest:" in out


def test_compile_falls_back_to_eager_when_unavailable(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no triton here")

    monkeypatch.setattr(torch, "compile", broken)
    trainer = _tiny_trainer(tmp_path, compile=True)
    assert trainer.fwd is trainer.model
    stats = trainer._accumulate_gradients(_minibatch(trainer), 0.01)
    assert all(np.isfinite(v) for v in stats.values())


def test_narrow_heads_warn_and_presets_use_fused_friendly_width():
    from catan_ai.train import PRESETS

    with pytest.warns(UserWarning, match="multiple of 8"):
        build_model(ModelConfig(d_model=160, n_layers=1, n_heads=8))
    for name in ("warmup", "full"):
        m = PRESETS[name]["model"]
        assert (m["d_model"] // m["n_heads"]) % 8 == 0, name


def test_attention_bias_is_kernel_friendly(monkeypatch):
    import torch.nn.functional as F

    seen = {}
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kw):
        seen["stride"] = attn_mask.stride()
        return real(q, k, v, attn_mask=attn_mask, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    model = build_model(ModelConfig(d_model=32, n_layers=1, n_heads=2))
    model(torch.from_numpy(CatanGame(seed=0).observe(0)).unsqueeze(0).repeat(3, 1))
    # Last dim stride 1 (required by fused CUDA kernels), batch broadcast with stride 0.
    assert seen["stride"][-1] == 1
    assert seen["stride"][0] == 0
