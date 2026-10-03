import json

import numpy as np
import pytest

from catan_ai import _engine
from catan_ai.agents import ScriptedAgent, play_game
from catan_ai.engine import ACTION_OFFSETS, ACTION_SIZE, OBS_OFFSETS, OBS_SIZE, CatanGame


def test_constants_consistent():
    assert ACTION_SIZE == 419
    last_off, last_width = OBS_OFFSETS["global"]
    assert last_off + last_width == OBS_SIZE
    assert ACTION_OFFSETS["cancel"] == ACTION_SIZE - 1


def test_setup_flow_and_views():
    g = CatanGame(seed=3)
    assert g.phase == "setup_settlement"
    assert g.actors == [0]
    board = g.board()
    assert len(board["hexes"]) == 19 and len(board["vertices"]) == 54 and len(board["edges"]) == 72
    assert len(board["ports"]) == 9
    legal = g.legal_actions(0)
    assert all(a["type"] == "build_settlement" for a in legal)
    g.apply(0, legal[0])
    assert g.phase == "setup_road"
    view = g.view(1)
    assert view["players"][0]["resources"] is None  # hidden from seat 1
    assert g.view(0)["players"][0]["resources"] is not None


def test_illegal_action_raises():
    g = CatanGame(seed=1)
    with pytest.raises(ValueError):
        g.apply(0, {"type": "roll_dice"})
    with pytest.raises(ValueError):
        g.apply(1, {"type": "build_settlement", "vertex": 0})
    assert g.check(0, {"type": "end_turn"}) is not None


def test_json_roundtrip_preserves_state():
    g = play_game([ScriptedAgent("heuristic", 1) for _ in range(4)], seed=5, max_turns=10)
    g2 = CatanGame.from_json(g.to_json())
    assert g2.view(0) == g.view(0)


def test_index_encoding_roundtrip():
    g = CatanGame(seed=2)
    mask = g.legal_mask(0)
    for idx in np.flatnonzero(mask):
        a = g.index_to_action(0, idx)
        assert g.action_to_index(0, a) == idx


def test_full_games_with_scripted_agents():
    for seed in range(5):
        g = play_game([ScriptedAgent("heuristic", seed + i) for i in range(4)], seed=seed)
        assert g.is_over
        assert g.winner is not None
        assert g.total_vp(g.winner) >= 10


def test_events_and_redaction():
    g = CatanGame(seed=11)
    agents = [ScriptedAgent("random", i) for i in range(4)]
    while (p := g.next_actor) is not None and g.turn < 30:
        g.apply(p, agents[p].act(g, p))
    events = g.drain_events()
    assert any(e["type"] == "dice_rolled" for e in events)
    assert g.drain_events() == []
    for e in CatanGame.redact_events(events, None):
        if e["type"] == "dev_card_bought":
            assert e["card"] is None
        if e["type"] == "stolen":
            assert e["resource"] is None


def test_arbitrary_human_trade_and_ai_response():
    g = CatanGame(seed=4)
    while g.phase.startswith("setup"):
        p = g.current
        g.apply(p, g.bot_action(p))
    g.force_roll(2, 3)
    me = g.current
    hand = g.view(me)["players"][me]["resources"]
    give_r = int(np.argmax(hand))
    if hand[give_r] == 0:
        pytest.skip("no cards to offer")
    give = [0] * 5
    give[give_r] = 1
    want = [0] * 5
    want[(give_r + 1) % 5] = 1
    want[(give_r + 2) % 5] = 1  # 1-for-2 offers are outside the template catalogue
    g.apply(me, {"type": "offer_trade", "give": give, "want": want})
    assert g.phase == "trade_response"
    for p in g.actors:
        g.apply(p, g.bot_action(p))
    assert g.phase in ("trade_confirm", "main")


def test_vecenv_shapes_and_step():
    env = _engine.VecEnv(8, seed=1, bot_kind="random", num_bot_seats=2, vp_reward_scale=0.1)
    obs, mask, actor = env.observe()
    assert obs.shape == (8, OBS_SIZE) and mask.shape == (8, ACTION_SIZE) and actor.shape == (8,)
    seats = env.policy_seats()
    assert seats.sum(1).tolist() == [2] * 8
    assert all(seats[i, actor[i]] for i in range(8))
    rng = np.random.default_rng(0)
    episodes = 0
    for _ in range(2000):
        obs, mask, actor = env.observe()
        a = (rng.random(mask.shape) * mask).argmax(1)
        rewards, done, winner, turns = env.step(a)
        assert rewards.shape == (8, 4)
        episodes += int(done.sum())
    assert episodes > 0
    all_obs = env.observe_all_seats()
    assert all_obs.shape == (8, 4, OBS_SIZE)


def _play_random(env, steps, rng):
    for _ in range(steps):
        _, mask, _ = env.observe()
        env.step((rng.random(mask.shape) * mask).argmax(1))


def test_vecenv_fractional_bot_seats():
    env = _engine.VecEnv(400, seed=2, bot_kind="random", num_bot_seats=1.25)
    bots = (~env.policy_seats()).sum(1)
    assert set(bots.tolist()) == {1, 2}
    assert 0.15 < (bots == 2).mean() < 0.35
    env = _engine.VecEnv(400, seed=2, bot_kind="random", num_bot_seats=0.5)
    bots = (~env.policy_seats()).sum(1)
    assert set(bots.tolist()) == {0, 1} and 0.4 < bots.mean() < 0.6
    # Python always has a decision to make, even in envs without bots.
    _, _, actor = env.observe()
    assert env.policy_seats()[np.arange(400), actor].all()
    for bad in (-0.5, 3.5, 4):
        with pytest.raises(ValueError):
            _engine.VecEnv(2, bot_kind="random", num_bot_seats=bad)


def test_vecenv_mixed_player_counts():
    env = _engine.VecEnv(60, seed=4, player_counts=[2, 3, 4], bot_kind="random", num_bot_seats=1)
    assert env.num_players == 4
    counts = env.game_players()
    assert set(counts.tolist()) == {2, 3, 4}
    seats = env.policy_seats()
    assert (seats.sum(1) == counts - 1).all()
    assert not seats[counts == 2][:, 2:].any()
    _play_random(env, 3000, np.random.default_rng(0))
    counts = env.game_players()
    all_obs = env.observe_all_seats()
    assert all_obs.shape == (60, 4, OBS_SIZE)
    assert not all_obs[counts == 2][:, 2:].any() and all_obs[counts == 2][:, :2].any()
    with pytest.raises(ValueError):
        _engine.VecEnv(2, player_counts=[])
    with pytest.raises(ValueError):
        _engine.VecEnv(2, player_counts=[5])


def test_vecenv_rejects_illegal_actions():
    env = _engine.VecEnv(2, seed=0)
    _, mask, _ = env.observe()
    bad = np.array([int(np.flatnonzero(~mask[i])[0]) for i in range(2)])
    with pytest.raises(ValueError):
        env.step(bad)


def test_arena_counts():
    wins = _engine.arena(["heuristic", "random"], 50, seed=3)
    assert sum(wins) == 50
    assert wins[0] > wins[1]


def test_board_json_static_geometry():
    b1 = json.loads(_engine.Game(seed=1).board_json())
    b2 = json.loads(_engine.Game(seed=2).board_json())
    assert b1["vertices"] == b2["vertices"]
    assert b1["edges"] == b2["edges"]
