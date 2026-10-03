"""PPO self-play training.

Catan is a turn-based, multi-seat game: in any env only one seat acts per step, and a
single step can change the outcome for several seats (e.g. the winning move). We
therefore keep one trajectory *chain* per ``(env, seat)``:

* each stored transition records its env and seat;
* rewards returned by ``VecEnv.step`` (shape ``[N, 4]``) are added to the *latest*
  transition of each seat's chain, so terminal rewards reach every player;
* GAE runs along each chain (``next_index`` links), bootstrapping unfinished chains
  with the value of the seat's current observation (``VecEnv.observe_all_seats``).

Opponents: by default every Python-controlled seat is played by the learner
(pure self-play). With ``pool_prob > 0`` a fraction of games instead pit one learner
seat against frozen past snapshots, which stabilizes training and avoids
strategy cycling. Scripted Rust bots can also fill seats (``bot_kind``/``bot_seats``).

See ``docs/TRAINING.md`` for guidance on hyperparameters and hardware.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from catan_ai import _engine
from catan_ai.bench import BACKENDS, attention_context
from catan_ai.engine import ACTION_SIZE, OBS_SIZE
from catan_ai.model import ModelConfig, build_model, check_compatible, masked_logits, save_checkpoint


@dataclass
class TrainConfig:
    run_dir: str = "runs/default"
    seed: int = 0
    device: str = "auto"
    # Environment
    num_envs: int = 256
    rollout_steps: int = 128
    # Player count of training games (unless player_counts is set) and of the main eval.
    num_players: int = 4
    # Optional per-game mix of player counts, e.g. [2, 3, 4] (uniform; repeat to weight).
    # Each extra count also gets its own eval (eval/heuristic_win_rate_<n>p).
    player_counts: list[int] = field(default_factory=list)
    max_turns: int = 300
    max_trade_offers_per_turn: int = 3
    vp_reward_scale: float = 0.02
    zero_sum: bool = True
    bot_kind: str | None = None
    # Scripted seats per game; fractional = probability of one more (0.5 = a bot in half the
    # games), so the learner cannot spend its whole budget exploiting a fixed bot.
    bot_seats: float = 0
    # Optimization
    total_updates: int = 20_000
    # Update at which the LR/entropy cosine starts (it runs from here to total_updates).
    # None = automatic: 0 for a fresh run, kept when resuming the same run, and the
    # checkpoint's update when resuming with a different schedule (e.g. warm-up -> full).
    schedule_start_update: int | None = None
    lr: float = 3e-4
    lr_final_frac: float = 0.1
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final_coef: float = 0.002
    max_grad_norm: float = 0.5
    epochs: int = 3
    minibatch_size: int = 4096
    # Samples per forward/backward pass; halved automatically on CUDA OOM.
    micro_batch_size: int = 1024
    target_kl: float = 0.03
    amp: bool = True
    # scaled_dot_product_attention backend: auto | efficient | cudnn | flash | math.
    # Run `catan-bench` to find the fastest one for your GPU.
    attention: str = "auto"
    # torch.compile the learner model (needs a working Triton install).
    compile: bool = True
    # League
    pool_prob: float = 0.25
    snapshot_every: int = 50
    pool_size: int = 30
    active_opponents: int = 4
    # Bookkeeping
    eval_every: int = 50
    eval_games: int = 200
    checkpoint_every: int = 25
    log_every: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        model = ModelConfig.from_dict(d.pop("model", {}))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(model=model, **known)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class RolloutBuffer:
    """Flat storage for one rollout; transitions are appended in time order."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, OBS_SIZE), dtype=np.float32)
        self.mask = np.zeros((capacity, ACTION_SIZE), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.logp = np.zeros(capacity, dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.next_index = np.full(capacity, -1, dtype=np.int64)
        self.bootstrap = np.zeros(capacity, dtype=np.float32)
        self.step = np.zeros(capacity, dtype=np.int64)
        self.size = 0

    def reset(self) -> None:
        self.reward[:] = 0
        self.done[:] = False
        self.next_index[:] = -1
        self.bootstrap[:] = 0
        self.size = 0

    def add(self, obs, mask, action, logp, value, step) -> np.ndarray:
        n = len(action)
        s = self.size
        if s + n > self.capacity:
            raise RuntimeError("rollout buffer overflow")
        sl = slice(s, s + n)
        self.obs[sl] = obs
        self.mask[sl] = mask
        self.action[sl] = action
        self.logp[sl] = logp
        self.value[sl] = value
        self.step[sl] = step
        self.size += n
        return np.arange(s, s + n)

    def compute_gae(self, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.size
        adv = np.zeros(n, dtype=np.float32)
        values = self.value[:n]
        nxt = self.next_index[:n]
        done = self.done[:n]
        steps = self.step[:n]
        # Successors always live at a later step, so sweep steps in reverse.
        order = np.argsort(-steps, kind="stable")
        boundaries = np.flatnonzero(np.diff(steps[order])) + 1
        for group in np.split(order, boundaries):
            nx = nxt[group]
            has_next = nx >= 0
            next_v = np.where(has_next, values[np.maximum(nx, 0)], self.bootstrap[group])
            next_adv = np.where(has_next, adv[np.maximum(nx, 0)], 0.0)
            terminal = done[group]
            next_v = np.where(terminal, 0.0, next_v)
            next_adv = np.where(terminal, 0.0, next_adv)
            delta = self.reward[group] + gamma * next_v - values[group]
            adv[group] = delta + gamma * lam * next_adv
        returns = adv + values
        return adv, returns


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: str | None = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        os.makedirs(cfg.run_dir, exist_ok=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.model = build_model(cfg.model).to(self.device)
        cuda = self.device.type == "cuda"
        if cuda:
            # TF32 for any fp32 matmuls left outside bf16 autocast.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=1e-4, fused=cuda
        )
        self.update = 0
        self.total_steps = 0
        self.pool: list[dict] = []
        if resume:
            self._load(resume)
        if cfg.schedule_start_update is None:
            cfg.schedule_start_update = 0

        self.env = _engine.VecEnv(
            cfg.num_envs,
            seed=cfg.seed + 7919 * self.update,
            num_players=cfg.num_players,
            max_trade_offers_per_turn=cfg.max_trade_offers_per_turn,
            max_turns=cfg.max_turns,
            bot_kind=cfg.bot_kind or None,
            num_bot_seats=float(cfg.bot_seats) if cfg.bot_kind else 0.0,
            vp_reward_scale=cfg.vp_reward_scale,
            zero_sum=cfg.zero_sum,
            player_counts=list(cfg.player_counts) or None,
        )
        self.n = cfg.num_envs
        # Widest game in the mix; smaller games just leave the extra seats unused.
        self.p = self.env.num_players
        self.buffer = RolloutBuffer(cfg.num_envs * cfg.rollout_steps, self.device)
        # Latest transition per (env, seat) chain, -1 if none in this rollout.
        self.last_idx = np.full((self.n, 4), -1, dtype=np.int64)
        # Opponent assignment per env: -1 = full self-play, k = active opponent k.
        self.opp_of_env = np.full(self.n, -1, dtype=np.int64)
        self.learner_seat = np.zeros(self.n, dtype=np.int64)
        self.opponents: list[torch.nn.Module] = []
        self._refresh_opponents()
        for i in range(self.n):
            self._assign_env(i)

        self.use_amp = cfg.amp and self.device.type == "cuda"
        if cfg.attention not in BACKENDS:
            raise ValueError(f"attention must be one of {sorted(BACKENDS)}")
        # Compiled view of the model for the learner's fixed-size micro-batches (shares
        # parameters). Rollout/bootstrap batches vary in size and stay eager.
        self.micro_batch_size = max(1, min(cfg.micro_batch_size, cfg.minibatch_size))
        self.fwd = self._maybe_compile() if cfg.compile else self.model
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(cfg.run_dir)
        except Exception:  # tensorboard is optional
            self.writer = None
        self.metrics_file = open(os.path.join(cfg.run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        with open(os.path.join(cfg.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        # (winner, turns, learner seat or -1 for self-play games)
        self.episode_stats: list[tuple[int, int, int]] = []

    # ------------------------------------------------------------------ league
    def _snapshot(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _refresh_opponents(self) -> None:
        cfg = self.cfg
        if cfg.pool_prob <= 0 or not self.pool:
            self.opponents = []
            return
        chosen = random.sample(self.pool, k=min(cfg.active_opponents, len(self.pool)))
        self.opponents = []
        for sd in chosen:
            m = build_model(cfg.model).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            self.opponents.append(m)

    def _assign_env(self, i: int) -> None:
        if self.opponents and random.random() < self.cfg.pool_prob:
            self.opp_of_env[i] = random.randrange(len(self.opponents))
        else:
            self.opp_of_env[i] = -1
        self.learner_seat[i] = -1  # chosen lazily among Python seats

    # ------------------------------------------------------------------ rollout
    def _maybe_compile(self) -> torch.nn.Module:
        """torch.compile the model, falling back to eager mode if compilation fails."""
        try:
            compiled = torch.compile(self.model)
            probe = torch.zeros(self.micro_batch_size, OBS_SIZE, device=self.device)
            with self._ctx():
                logits, value = compiled(probe)
            (logits.float().sum() + value.float().sum()).backward()
            self.model.zero_grad(set_to_none=True)
            print("torch.compile: enabled for the learner")
            return compiled
        except Exception as e:  # missing Triton/compiler, unsupported GPU, ...
            self.model.zero_grad(set_to_none=True)
            first = str(e).strip().splitlines()[0][:160] if str(e).strip() else ""
            print(f"torch.compile unavailable ({type(e).__name__}: {first}); continuing without it")
            return self.model

    def _ctx(self) -> contextlib.ExitStack:
        """Autocast + the configured attention backend, for every model call."""
        stack = contextlib.ExitStack()
        stack.enter_context(torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp))
        stack.enter_context(attention_context(self.cfg.attention))
        return stack

    @torch.no_grad()
    def _policy(self, model, obs: np.ndarray, mask: np.ndarray, greedy: bool = False):
        o = torch.from_numpy(obs).to(self.device, non_blocking=True)
        m = torch.from_numpy(mask).to(self.device, non_blocking=True)
        with self._ctx():
            logits, value = model(o)
        logits = masked_logits(logits.float(), m)
        if greedy:
            action = logits.argmax(-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, action[:, None]).squeeze(1)
        return action.cpu().numpy(), logp.cpu().numpy(), value.float().cpu().numpy()

    def collect(self) -> dict:
        cfg = self.cfg
        buf = self.buffer
        buf.reset()
        self.last_idx[:] = -1
        self.model.eval()
        policy_seats = self.env.policy_seats()
        for t in range(cfg.rollout_steps):
            obs, mask, actor = self.env.observe()
            # Learner rows: self-play envs, or the learner seat in pool envs.
            for i in np.flatnonzero((self.learner_seat < 0) & (self.opp_of_env >= 0)):
                self.learner_seat[i] = int(np.random.choice(np.flatnonzero(policy_seats[i])))
            is_learner = (self.opp_of_env < 0) | (self.learner_seat == actor)
            actions = np.zeros(self.n, dtype=np.int64)

            rows = np.flatnonzero(is_learner)
            if len(rows):
                a, logp, v = self._policy(self.model, obs[rows], mask[rows])
                actions[rows] = a
                idx = buf.add(obs[rows], mask[rows], a, logp, v, t)
                seats = actor[rows]
                prev = self.last_idx[rows, seats]
                linked = prev >= 0
                buf.next_index[prev[linked]] = idx[linked]
                self.last_idx[rows, seats] = idx
            for k, opp in enumerate(self.opponents):
                orows = np.flatnonzero(~is_learner & (self.opp_of_env == k))
                if len(orows):
                    a, _, _ = self._policy(opp, obs[orows], mask[orows])
                    actions[orows] = a

            rewards, done, winner, turns = self.env.step(actions)
            self.total_steps += self.n

            # Credit rewards to the latest transition of every seat's chain.
            has = self.last_idx >= 0
            ei, si = np.nonzero(has[:, : self.p])
            np.add.at(buf.reward, self.last_idx[ei, si], rewards[ei, si])

            if done.any():
                for i in np.flatnonzero(done):
                    chain = self.last_idx[i]
                    buf.done[chain[chain >= 0]] = True
                    self.last_idx[i] = -1
                    if self.opp_of_env[i] < 0 or winner[i] < 0:
                        self.episode_stats.append((int(winner[i]), int(turns[i]), -1))
                    else:
                        self.episode_stats.append((int(winner[i]), int(turns[i]), int(self.learner_seat[i])))
                    self._assign_env(i)
                policy_seats = self.env.policy_seats()

        # Bootstrap unfinished chains with V(current observation of that seat).
        open_env, open_seat = np.nonzero(self.last_idx[:, : self.p] >= 0)
        if len(open_env):
            all_obs = self.env.observe_all_seats()
            o = torch.from_numpy(all_obs[open_env, open_seat]).to(self.device)
            values = []
            with (
                torch.no_grad(),
                torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp),
            ):
                # Up to num_envs x 4 rows: chunk to keep attention memory bounded.
                for chunk in o.split(max(self.micro_batch_size, self.n)):
                    with attention_context(self.cfg.attention):
                        values.append(self.model(chunk)[1].float())
            buf.bootstrap[self.last_idx[open_env, open_seat]] = torch.cat(values).cpu().numpy()
        return {"transitions": buf.size}

    # ------------------------------------------------------------------ update
    def _schedule(self) -> tuple[float, float]:
        start = self.cfg.schedule_start_update or 0
        frac = min(1.0, max(0, self.update - start) / max(1, self.cfg.total_updates - start))
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        lr = self.cfg.lr * (self.cfg.lr_final_frac + (1 - self.cfg.lr_final_frac) * cos)
        ent = self.cfg.entropy_final_coef + (self.cfg.entropy_coef - self.cfg.entropy_final_coef) * cos
        return lr, ent

    def _accumulate_gradients(self, tensors: tuple[torch.Tensor, ...], ent_coef: float) -> dict:
        """Forward/backward one minibatch in micro-batches (gradient accumulation).

        Attention with a relation bias materializes ``B x heads x L x L`` scores per layer, so
        large minibatches don't fit in GPU memory in one pass. Gradients are identical to a
        single pass because each micro-batch loss is weighted by its share of the minibatch.
        On CUDA OOM the micro-batch size is halved and the minibatch is retried.
        """
        cfg = self.cfg
        total = tensors[0].shape[0]
        while True:
            self.opt.zero_grad(set_to_none=True)
            # Accumulated on-device; a single host sync at the end of the minibatch.
            sums = torch.zeros(5, device=self.device)
            try:
                for lo in range(0, total, self.micro_batch_size):
                    o, m, act, old_logp, old_v, a, ret = (t[lo : lo + self.micro_batch_size] for t in tensors)
                    with self._ctx():
                        logits, value = self.fwd(o)
                    logits = masked_logits(logits.float(), m)
                    value = value.float()
                    logp_all = torch.log_softmax(logits, -1)
                    logp = logp_all.gather(1, act[:, None]).squeeze(1)
                    entropy = -(logp_all.exp() * logp_all.masked_fill(~m, 0.0)).sum(-1).mean()
                    ratio = (logp - old_logp).exp()
                    pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    v_clipped = old_v + (value - old_v).clamp(-cfg.clip, cfg.clip)
                    v_loss = (
                        0.5
                        * torch.max(
                            F.mse_loss(value, ret, reduction="none"),
                            F.mse_loss(v_clipped, ret, reduction="none"),
                        ).mean()
                    )
                    loss = pg + cfg.value_coef * v_loss - ent_coef * entropy
                    w = o.shape[0] / total
                    (loss * w).backward()
                    with torch.no_grad():
                        sums += w * torch.stack(
                            [
                                pg.detach(),
                                v_loss.detach(),
                                entropy.detach(),
                                ((ratio - 1) - (logp - old_logp)).mean(),
                                ((ratio - 1).abs() > cfg.clip).float().mean(),
                            ]
                        )
                keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
                return dict(zip(keys, sums.tolist(), strict=True))
            except torch.OutOfMemoryError:
                if self.micro_batch_size <= 1:
                    raise
                self.opt.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.micro_batch_size //= 2
                print(f"  CUDA out of memory: retrying with micro_batch_size={self.micro_batch_size}")

    def learn(self) -> dict:
        cfg = self.cfg
        buf = self.buffer
        n = buf.size
        adv, ret = buf.compute_gae(cfg.gamma, cfg.gae_lambda)
        lr, ent_coef = self._schedule()
        for g in self.opt.param_groups:
            g["lr"] = lr

        dev = self.device
        obs = torch.from_numpy(buf.obs[:n]).to(dev)
        mask = torch.from_numpy(buf.mask[:n]).to(dev)
        act = torch.from_numpy(buf.action[:n]).to(dev)
        old_logp = torch.from_numpy(buf.logp[:n]).to(dev)
        old_v = torch.from_numpy(buf.value[:n]).to(dev)
        adv_t = torch.from_numpy(adv).to(dev)
        ret_t = torch.from_numpy(ret).to(dev)

        self.model.train()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        batches = 0
        stop = False
        for _epoch in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, cfg.minibatch_size):
                b = perm[start : start + cfg.minibatch_size]
                # Normalize advantages over the whole minibatch, not per micro-batch.
                a = adv_t[b]
                a = (a - a.mean()) / (a.std() + 1e-8)
                tensors = (obs[b], mask[b], act[b], old_logp[b], old_v[b], a, ret_t[b])
                mb_stats = self._accumulate_gradients(tensors, ent_coef)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                for k, v in mb_stats.items():
                    stats[k] += v
                batches += 1
                if cfg.target_kl and mb_stats["approx_kl"] > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: v / max(1, batches) for k, v in stats.items()}
        var_y = float(np.var(ret))
        out["explained_var"] = float("nan") if var_y == 0 else 1 - float(np.var(ret - buf.value[:n])) / var_y
        out["lr"] = lr
        out["entropy_coef"] = ent_coef
        out["early_stop"] = float(stop)
        return out

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(
        self, games: int, bot_kind: str = "heuristic", greedy: bool = True, num_players: int | None = None
    ) -> dict:
        """Win rate of the current policy (one seat) against scripted bots."""
        players = num_players or self.cfg.num_players
        env = _engine.VecEnv(
            min(games, 128),
            seed=10_000 + self.update,
            num_players=players,
            max_turns=self.cfg.max_turns,
            bot_kind=bot_kind,
            num_bot_seats=players - 1,
        )
        self.model.eval()
        wins = finished = truncated = 0
        lengths = []
        seats = env.policy_seats()
        while finished < games:
            obs, mask, _ = env.observe()
            a, _, _ = self._policy(self.model, obs, mask, greedy=greedy)
            _, done, winner, turns = env.step(a)
            for i in np.flatnonzero(done):
                if finished >= games:
                    break
                finished += 1
                lengths.append(int(turns[i]))
                if winner[i] < 0:
                    truncated += 1
                elif seats[i, winner[i]]:
                    wins += 1
            if done.any():
                seats = env.policy_seats()
        return {
            f"eval/{bot_kind}_win_rate": wins / games,
            f"eval/{bot_kind}_truncated": truncated / games,
            f"eval/{bot_kind}_game_turns": float(np.mean(lengths)) if lengths else 0.0,
        }

    # ------------------------------------------------------------------ io
    def save(self, name: str) -> str:
        path = os.path.join(self.cfg.run_dir, name)
        tmp = path + ".tmp"
        save_checkpoint(
            tmp,
            self.model,
            self.cfg.model,
            extra={
                "optimizer": self.opt.state_dict(),
                "update": self.update,
                "total_steps": self.total_steps,
                "train_config": self.cfg.to_dict(),
                "pool": self.pool[-self.cfg.pool_size :],
            },
        )
        os.replace(tmp, path)
        return path

    def _load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        check_compatible(payload)
        self.model.load_state_dict(payload["state_dict"])
        if "optimizer" in payload:
            self.opt.load_state_dict(payload["optimizer"])
        self.update = int(payload.get("update", 0))
        self.total_steps = int(payload.get("total_steps", 0))
        self.pool = list(payload.get("pool", []))
        if self.cfg.schedule_start_update is None:
            self.cfg.schedule_start_update = resumed_schedule_start(
                payload.get("train_config", {}), self.cfg, self.update
            )
            if self.cfg.schedule_start_update == self.update and self.update > 0:
                print(f"schedule settings changed: LR/entropy cosine restarts at update {self.update}")

    def log(self, metrics: dict) -> None:
        metrics = {"update": self.update, "steps": self.total_steps, **metrics}
        self.metrics_file.write(json.dumps(metrics) + "\n")
        self.metrics_file.flush()
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("update",):
                    self.writer.add_scalar(k if "/" in k else f"train/{k}", v, self.total_steps)

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        cfg = self.cfg
        print(
            f"training on {self.device} | {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params | "
            f"{cfg.num_envs} envs x {cfg.rollout_steps} steps"
        )
        while self.update < cfg.total_updates:
            t0 = time.perf_counter()
            roll = self.collect()
            t1 = time.perf_counter()
            stats = self.learn()
            t2 = time.perf_counter()
            self.update += 1

            if self.update % cfg.snapshot_every == 0 and cfg.pool_prob > 0:
                self.pool.append(self._snapshot())
                self.pool = self.pool[-cfg.pool_size :]
                self._refresh_opponents()

            if self.update % cfg.log_every == 0:
                eps = self.episode_stats
                self.episode_stats = []
                ep = {}
                if eps:
                    ep["episodes"] = len(eps)
                    ep["game_turns"] = float(np.mean([e[1] for e in eps]))
                    ep["truncated_frac"] = float(np.mean([e[0] < 0 for e in eps]))
                    league = [e for e in eps if e[2] >= 0]
                    if league:
                        ep["league_win_rate"] = float(np.mean([e[0] == e[2] for e in league]))
                metrics = {
                    **stats,
                    **ep,
                    "transitions": roll["transitions"],
                    "sps": roll["transitions"] / max(1e-9, t2 - t0),
                    "rollout_s": t1 - t0,
                    "learn_s": t2 - t1,
                }
                self.log(metrics)
                print(
                    f"upd {self.update:6d} | steps {self.total_steps:11,d} | sps {metrics['sps']:7.0f} "
                    f"(roll {metrics['rollout_s']:.1f}s learn {metrics['learn_s']:.1f}s) | "
                    f"pl {stats['policy_loss']:+.3f} vl {stats['value_loss']:.3f} ent {stats['entropy']:.2f} "
                    f"kl {stats['approx_kl']:.4f} ev {stats['explained_var']:+.2f} | "
                    f"turns {ep.get('game_turns', float('nan')):.0f} trunc {ep.get('truncated_frac', float('nan')):.2f}"
                )

            if cfg.eval_every and self.update % cfg.eval_every == 0:
                ev = self.evaluate(cfg.eval_games)
                for n in sorted(set(cfg.player_counts) - {cfg.num_players}):
                    extra = self.evaluate(cfg.eval_games, num_players=n)
                    ev.update({f"{k}_{n}p": v for k, v in extra.items()})
                self.log(ev)
                print("  eval:", {k: round(v, 3) for k, v in ev.items()})
            if self.update % cfg.checkpoint_every == 0 or self.update == cfg.total_updates:
                self.save("latest.pt")
                if self.update % (cfg.checkpoint_every * 20) == 0:
                    self.save(f"update_{self.update:07d}.pt")
        self.save("final.pt")
        if self.writer:
            self.writer.close()
        self.metrics_file.close()


SCHEDULE_KEYS = ("total_updates", "lr", "lr_final_frac", "entropy_coef", "entropy_final_coef")


def resumed_schedule_start(prev: dict, cfg: TrainConfig, update: int) -> int:
    """Where the cosine schedules start when resuming from a checkpoint at ``update``.

    Resuming the same run keeps its schedule. Resuming with different schedule settings
    (typically warm-up -> full) restarts the cosine at the checkpoint's update, so the new
    run's ``lr``/``entropy_coef`` are its starting values instead of whatever point of the
    new curve the old update counter happens to land on.
    """
    if prev and all(prev.get(k) == getattr(cfg, k) for k in SCHEDULE_KEYS):
        return int(prev.get("schedule_start_update") or 0)
    return update


def clone_config(cfg: TrainConfig, **overrides) -> TrainConfig:
    c = copy.deepcopy(cfg)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c
