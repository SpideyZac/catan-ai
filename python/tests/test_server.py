import json
import time

import pytest
from fastapi.testclient import TestClient

from catan_ai.server import rooms as rooms_mod
from catan_ai.server.app import ServerSettings, create_app


@pytest.fixture(autouse=True)
def fast_bots(monkeypatch):
    monkeypatch.setattr(rooms_mod, "BOT_DELAYS", {"fast": 0.0, "normal": 0.0, "slow": 0.0})


@pytest.fixture()
def client(tmp_path):
    settings = ServerSettings(
        static_dir=str(tmp_path / "nostatic"), models_dir=None, data_dir=str(tmp_path / "rooms"), device="cpu"
    )
    with TestClient(create_app(settings)) as c:
        yield c


def recv_until(ws, pred, limit=5000):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == "error" and msg.get("fatal"):
            raise AssertionError(msg)
        if pred(msg):
            return msg
    raise AssertionError("condition not reached")


def join(client, code, name, token=None):
    ws = client.websocket_connect(f"/ws/{code}")
    ws.__enter__()
    ws.send_text(json.dumps({"type": "hello", "name": name, "token": token}))
    welcome = recv_until(ws, lambda m: m["type"] == "welcome")
    return ws, welcome


def test_health_and_bots(client):
    assert client.get("/api/health").json()["ok"]
    ids = [b["id"] for b in client.get("/api/bots").json()]
    assert "heuristic" in ids and "random" in ids


def test_create_and_join_room(client):
    r = client.post("/api/rooms", json={"name": "Alice"}).json()
    ws, welcome = join(client, r["code"], "Alice", r["token"])
    assert welcome["client_id"] == r["client_id"]
    state = recv_until(ws, lambda m: m["type"] == "state")
    assert state["you"]["is_host"]
    assert state["you"]["seats"] == [0]
    assert state["room"]["seats"][0]["name"] == "Alice"
    assert client.get(f"/api/rooms/{r['code']}").status_code == 200
    assert client.get("/api/rooms/NOPE1").status_code == 404
    ws.__exit__(None, None, None)


def test_all_bot_game_runs_to_completion(client):
    r = client.post("/api/rooms", json={"name": "Host"}).json()
    ws, _ = join(client, r["code"], "Host", r["token"])
    ws.send_text(json.dumps({"type": "configure", "seats": [{"kind": "bot", "bot": "heuristic"}] * 4}))
    ws.send_text(json.dumps({"type": "start"}))
    final = recv_until(ws, lambda m: m["type"] == "state" and m["room"]["status"] == "finished", limit=20000)
    view = final["game"]["views"]["spectator"]
    assert view["winner"] is not None
    ws.__exit__(None, None, None)


def test_human_vs_bots_and_hidden_information(client):
    r = client.post("/api/rooms", json={"name": "Human"}).json()
    code = r["code"]
    ws, _ = join(client, code, "Human", r["token"])
    spectator, _ = join(client, code, "Watcher")
    ws.send_text(json.dumps({"type": "start"}))
    moves = 0
    deadline = time.time() + 60
    while moves < 60 and time.time() < deadline:
        msg = recv_until(ws, lambda m: m["type"] == "state" and m["game"] is not None)
        legal = msg["game"]["legal"].get("0")
        if not legal:
            continue
        view = msg["game"]["views"]["0"]
        assert view["players"][0]["resources"] is not None
        assert all(p["resources"] is None for p in view["players"][1:])
        action = next((a for a in legal if a["type"] in ("end_turn", "roll_dice")), legal[0])
        ws.send_text(json.dumps({"type": "action", "seat": 0, "action": action}))
        moves += 1
    assert moves > 5
    smsg = recv_until(spectator, lambda m: m["type"] == "state" and m["game"] is not None)
    assert smsg["you"]["seats"] == []
    assert all(p["resources"] is None for p in smsg["game"]["views"]["spectator"]["players"])
    spectator.send_text(json.dumps({"type": "action", "seat": 0, "action": {"type": "end_turn"}}))
    err = recv_until(spectator, lambda m: m["type"] == "error")
    assert "control" in err["message"]
    ws.__exit__(None, None, None)
    spectator.__exit__(None, None, None)


def test_pass_and_play_and_reconnect(client):
    r = client.post("/api/rooms", json={"name": "Family"}).json()
    code = r["code"]
    ws, _ = join(client, code, "Family", r["token"])
    ws.send_text(
        json.dumps(
            {
                "type": "configure",
                "seats": [{"kind": "human"}, {"kind": "human"}, {"kind": "bot"}, {"kind": "bot"}],
            }
        )
    )
    ws.send_text(json.dumps({"type": "claim_seat", "seat": 1, "name": "Kid"}))
    state = recv_until(ws, lambda m: m["type"] == "state" and m["you"]["seats"] == [0, 1])
    assert state["room"]["seats"][1]["name"] == "Kid"
    ws.__exit__(None, None, None)
    # Reconnect with the same token: seats are kept.
    ws2, welcome = join(client, code, "Family", r["token"])
    assert welcome["client_id"] == r["client_id"]
    state = recv_until(ws2, lambda m: m["type"] == "state")
    assert state["you"]["seats"] == [0, 1]
    ws2.__exit__(None, None, None)


def test_non_host_cannot_start_and_invalid_messages_rejected(client):
    r = client.post("/api/rooms", json={"name": "Host"}).json()
    host, _ = join(client, r["code"], "Host", r["token"])
    guest, _ = join(client, r["code"], "Guest")
    guest.send_text(json.dumps({"type": "start"}))
    assert "host" in recv_until(guest, lambda m: m["type"] == "error")["message"]
    guest.send_text(json.dumps({"type": "bogus"}))
    assert recv_until(guest, lambda m: m["type"] == "error")
    guest.send_text("not json")
    assert recv_until(guest, lambda m: m["type"] == "error")
    host.__exit__(None, None, None)
    guest.__exit__(None, None, None)


def test_rooms_persist_across_restart(tmp_path):
    settings = ServerSettings(
        static_dir=str(tmp_path / "x"), models_dir=None, data_dir=str(tmp_path / "rooms"), device="cpu"
    )
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/rooms", json={"name": "Persist"}).json()
    with TestClient(create_app(settings)) as c:
        info = c.get(f"/api/rooms/{r['code']}")
        assert info.status_code == 200
        assert info.json()["seats"][0]["name"] == "Persist"
