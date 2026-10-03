"""Policy/value networks.

The default architecture, :class:`EntityTransformer`, splits the flat observation into
entity tokens (19 hexes, 54 vertices, 4 players, 1 trade token, 1 global token, plus a
learned CLS token: 80 tokens) and runs a pre-norm transformer over them. Each attention
layer receives a learned bias per (relation type, head) derived from the fixed board
topology (hex-vertex, vertex-vertex, hex-hex adjacency), which gives the model a strong
spatial prior.

Roads are *not* tokens (that would almost double the sequence length and the attention
cost). Instead each edge's features are embedded and summed into its two endpoint vertex
tokens, and road logits are computed from the two endpoint vertex outputs plus the edge's
own embedding.

Action logits come from the entities they are about:

* settlement/city logits from vertex tokens,
* road logits from pairs of endpoint vertex tokens,
* robber (hex x victim) logits from hex tokens,
* everything else (dice, dev cards, discards, maritime and player trades, trade
  responses) from an MLP over the CLS, "me" and trade tokens.

The layout contract with the engine is ``catan_ai.engine.OBS_OFFSETS`` /
``ACTION_OFFSETS``; a checkpoint stores the encoding version and the model version it
was trained with.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from catan_ai import _engine
from catan_ai.engine import ACTION_OFFSETS, ACTION_SIZE, OBS_OFFSETS, OBS_SIZE

NUM_HEXES, NUM_VERTICES, NUM_EDGES, NUM_PLAYERS = 19, 54, 72, 4
MASK_VALUE = -1e9
# Bump when parameter layouts change; older checkpoints then refuse to load.
MODEL_VERSION = 2

# Token order: [CLS, hexes, vertices, players, trade, global]
TOK_CLS = 0
TOK_HEX = 1
TOK_VERTEX = TOK_HEX + NUM_HEXES
TOK_PLAYER = TOK_VERTEX + NUM_VERTICES
TOK_TRADE = TOK_PLAYER + NUM_PLAYERS
TOK_GLOBAL = TOK_TRADE + 1
NUM_TOKENS = TOK_GLOBAL + 1  # 80, a multiple of 16 (friendly to fused attention kernels)

# Relation ids for the attention bias.
REL_NONE, REL_SELF, REL_HEX_VERTEX, REL_VERTEX_VERTEX, REL_HEX_HEX = range(5)
NUM_RELATIONS = 5

# Number of "global" logits produced by the MLP head.
_N_GLOBAL_PRE = ACTION_OFFSETS["settlement"]  # end..choose_resource (12)
_N_GLOBAL_POST = ACTION_SIZE - ACTION_OFFSETS["discard"]  # discard..cancel (151)


@lru_cache(maxsize=1)
def _board_geometry() -> dict:
    return json.loads(_engine.Game(seed=0).board_json())


@lru_cache(maxsize=1)
def relation_matrix() -> np.ndarray:
    """``[NUM_TOKENS, NUM_TOKENS]`` matrix of relation ids from the static board topology."""
    board = _board_geometry()
    rel = np.full((NUM_TOKENS, NUM_TOKENS), REL_NONE, dtype=np.int64)
    np.fill_diagonal(rel, REL_SELF)

    def link(a: int, b: int, r: int) -> None:
        rel[a, b] = r
        rel[b, a] = r

    hex_vertices = [h["vertices"] for h in board["hexes"]]
    for h, verts in enumerate(hex_vertices):
        for v in verts:
            link(TOK_HEX + h, TOK_VERTEX + v, REL_HEX_VERTEX)
    for a, b in board["edges"]:
        link(TOK_VERTEX + a, TOK_VERTEX + b, REL_VERTEX_VERTEX)
    for h1 in range(NUM_HEXES):
        for h2 in range(h1 + 1, NUM_HEXES):
            if len(set(hex_vertices[h1]) & set(hex_vertices[h2])) == 2:
                link(TOK_HEX + h1, TOK_HEX + h2, REL_HEX_HEX)
    return rel


@lru_cache(maxsize=1)
def edge_endpoints() -> np.ndarray:
    """``[NUM_EDGES, 2]`` vertex ids of each edge."""
    return np.asarray(_board_geometry()["edges"], dtype=np.int64)


@dataclass
class ModelConfig:
    arch: str = "entity_transformer"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.0
    # Only used by the MLP baseline.
    hidden: list[int] = field(default_factory=lambda: [512, 512, 256])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _split_obs(obs: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    counts = {"hex": NUM_HEXES, "vertex": NUM_VERTICES, "edge": NUM_EDGES, "player": NUM_PLAYERS}
    for name, (off, width) in OBS_OFFSETS.items():
        n = counts.get(name, 1)
        out[name] = obs[:, off : off + n * width].reshape(obs.shape[0], n, width)
    return out


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASK_VALUE)


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.rel_bias = nn.Embedding(NUM_RELATIONS, heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).view(b, length, 3, h, d // h).permute(2, 0, 3, 1, 4)
        # [B, H, L, L] view (stride 0 over batch) of the per-head relation bias.
        bias = self.rel_bias(rel).permute(2, 0, 1).unsqueeze(0).to(q.dtype).expand(b, -1, -1, -1)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.proj(attn.transpose(1, 2).reshape(b, length, d))
        return x + self.ff(self.ln2(x))


class EntityTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        if d % cfg.n_heads:
            raise ValueError(f"d_model ({d}) must be divisible by n_heads ({cfg.n_heads})")
        if (d // cfg.n_heads) % 8:
            warnings.warn(
                f"head width {d // cfg.n_heads} (d_model/n_heads) is not a multiple of 8, so fused "
                "attention kernels can't be used; e.g. d_model=160 with n_heads=5 gives 32",
                stacklevel=2,
            )
        widths = {name: width for name, (_, width) in OBS_OFFSETS.items()}
        self.embed = nn.ModuleDict({name: nn.Linear(w, d) for name, w in widths.items()})
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, NUM_TOKENS, d) * 0.02)
        self.blocks = nn.ModuleList(
            _Block(d, cfg.n_heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.ln_out = nn.LayerNorm(d)
        self.vertex_head = nn.Linear(d, 2)
        self.road_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.hex_head = nn.Linear(d, 4)
        self.global_head = nn.Sequential(
            nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, _N_GLOBAL_PRE + _N_GLOBAL_POST)
        )
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("rel", torch.from_numpy(relation_matrix()), persistent=False)
        ends = torch.from_numpy(edge_endpoints())
        self.register_buffer("edge_a", ends[:, 0].contiguous(), persistent=False)
        self.register_buffer("edge_b", ends[:, 1].contiguous(), persistent=False)
        incidence = torch.zeros(NUM_VERTICES, NUM_EDGES)
        incidence[ends[:, 0], torch.arange(NUM_EDGES)] = 1.0
        incidence[ends[:, 1], torch.arange(NUM_EDGES)] = 1.0
        self.register_buffer("incidence", incidence, persistent=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits[B, ACTION_SIZE], value[B])``; logits are *unmasked*."""
        parts = _split_obs(obs)
        b = obs.shape[0]
        edge = self.embed["edge"](parts["edge"])  # [B, 72, d]
        # Each vertex also sees the roads touching it.
        vertex = self.embed["vertex"](parts["vertex"]) + torch.matmul(self.incidence.to(edge.dtype), edge)
        tokens = torch.cat(
            [
                self.cls.expand(b, -1, -1),
                self.embed["hex"](parts["hex"]),
                vertex,
                self.embed["player"](parts["player"]),
                self.embed["trade"](parts["trade"]),
                self.embed["global"](parts["global"]),
            ],
            dim=1,
        )
        x = tokens + self.pos
        for blk in self.blocks:
            x = blk(x, self.rel)
        x = self.ln_out(x)

        cls_tok = x[:, TOK_CLS]
        me_tok = x[:, TOK_PLAYER]
        trade_tok = x[:, TOK_TRADE]
        g = self.global_head(torch.cat([cls_tok, me_tok, trade_tok], dim=-1))
        vx = x[:, TOK_VERTEX:TOK_PLAYER]  # [B, 54, d]
        vert = self.vertex_head(vx)  # [B, 54, 2]
        va, vb = vx[:, self.edge_a], vx[:, self.edge_b]  # [B, 72, d] each
        # Symmetric in the two endpoints.
        road = self.road_head(torch.cat([va + vb, va * vb, edge.to(va.dtype)], dim=-1)).squeeze(-1)  # [B, 72]
        hexes = self.hex_head(x[:, TOK_HEX:TOK_VERTEX]).reshape(b, NUM_HEXES * 4)  # [B, 76]
        logits = torch.cat(
            [g[:, :_N_GLOBAL_PRE], vert[..., 0], vert[..., 1], road, hexes, g[:, _N_GLOBAL_PRE:]], dim=1
        )
        value = self.value_head(torch.cat([cls_tok, me_tok], dim=-1)).squeeze(-1)
        return logits, value


class MLPNet(nn.Module):
    """Simple baseline: residual MLP over the flat observation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        prev = OBS_SIZE
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(prev, ACTION_SIZE)
        self.value = nn.Linear(prev, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy(z), self.value(z).squeeze(-1)


ARCHS = {"entity_transformer": EntityTransformer, "mlp": MLPNet}


def build_model(cfg: ModelConfig) -> nn.Module:
    try:
        return ARCHS[cfg.arch](cfg)
    except KeyError as e:
        raise ValueError(f"unknown architecture {cfg.arch!r}; choose from {sorted(ARCHS)}") from e


def save_checkpoint(path: str, model: nn.Module, cfg: ModelConfig, extra: dict | None = None) -> None:
    from catan_ai.engine import ENCODING_VERSION

    payload = {
        "model_config": cfg.to_dict(),
        "state_dict": model.state_dict(),
        "encoding_version": ENCODING_VERSION,
        "model_version": MODEL_VERSION,
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SIZE,
        **(extra or {}),
    }
    torch.save(payload, path)


def check_compatible(payload: dict) -> None:
    """Raise a clear error if a checkpoint was made with another encoding or model layout."""
    from catan_ai.engine import ENCODING_VERSION

    if payload.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"checkpoint encoding v{payload.get('encoding_version')} does not match engine v{ENCODING_VERSION}"
        )
    if payload.get("model_version", 1) != MODEL_VERSION:
        raise ValueError(
            f"checkpoint was made with model v{payload.get('model_version', 1)} but this code uses "
            f"v{MODEL_VERSION}; it cannot be loaded, train a new model"
        )


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Load a model for inference. Returns ``(model, payload)``."""
    payload = torch.load(path, map_location=device, weights_only=False)
    check_compatible(payload)
    cfg = ModelConfig.from_dict(payload["model_config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
