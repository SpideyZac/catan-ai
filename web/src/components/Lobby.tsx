import { useEffect, useState } from "react";
import { api, type RoomConnection } from "../lib/api";
import type { BotLevel, RoomSettings, StateMessage } from "../lib/types";
import { ChatPanel } from "./GameScreen";
import { PLAYER_COLORS, SettlementShape } from "./art";

export function Lobby({ conn, state, onLeave }: { conn: RoomConnection; state: StateMessage; onLeave: () => void }) {
  const { room, you } = state;
  const [bots, setBots] = useState<BotLevel[]>([]);
  const [copied, setCopied] = useState(false);
  const [localName, setLocalName] = useState<Record<number, string>>({});
  useEffect(() => {
    api.bots().then(setBots).catch(() => setBots([]));
  }, []);

  const host = you.is_host;
  const settings = room.settings;
  const update = (patch: Partial<RoomSettings>) => conn.configure({ settings: patch });
  const setSeat = (i: number, kind: "human" | "bot", bot?: string) => conn.setSeat(i, kind, bot);
  const unclaimed = room.seats.filter((s) => s.kind === "human" && !s.client_id).length;
  const link = `${location.origin}/room/${room.code}`;
  const iHaveSeat = you.seats.length > 0;

  return (
    <div className="lobby">
      <header className="lobby-head">
        <div className="brand" onClick={onLeave} role="button">
          <span className="logo-hex" /> Catan <em>AI</em>
        </div>
        <div className="code-box">
          <span className="muted">Table code</span>
          <b>{room.code}</b>
          <button
            className="btn small"
            onClick={() => {
              navigator.clipboard?.writeText(link).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              });
            }}
          >
            {copied ? "Link copied!" : "Copy invite link"}
          </button>
        </div>
      </header>

      <div className="lobby-body">
        <section className="panel seats-panel">
          <h2>Seats</h2>
          <p className="muted">
            Invite friends with the link for online play, or add several local players to pass one device around.
          </p>
          <div className="seats">
            {room.seats.map((s) => {
              const mine = s.client_id === you.client_id;
              const c = PLAYER_COLORS[s.color];
              return (
                <div key={s.index} className={`seat ${mine ? "mine" : ""}`} style={{ ["--pc" as string]: c.fill }}>
                  <svg viewBox="-16 -18 32 34" className="seat-piece">
                    <SettlementShape color={s.color} />
                  </svg>
                  <div className="seat-main">
                    <div className="seat-name">
                      {s.kind === "bot" ? s.name : s.client_id ? s.name : <span className="muted">Open seat</span>}
                      {s.kind === "human" && s.client_id && !s.connected && <span className="offline"> · offline</span>}
                      {mine && <span className="you-tag">you</span>}
                    </div>
                    {host ? (
                      <div className="seat-controls">
                        <select
                          value={s.kind === "bot" ? (s.bot ?? "heuristic") : "human"}
                          onChange={(e) =>
                            e.target.value === "human" ? setSeat(s.index, "human") : setSeat(s.index, "bot", e.target.value)
                          }
                        >
                          <option value="human">Human player</option>
                          {bots.map((b) => (
                            <option key={b.id} value={b.id}>
                              AI · {b.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    ) : (
                      <div className="muted small-text">{s.kind === "bot" ? "Computer player" : "Human player"}</div>
                    )}
                  </div>
                  <div className="seat-actions">
                    {s.kind === "human" && !s.client_id && (
                      <>
                        {!iHaveSeat ? (
                          <button className="btn good small" onClick={() => conn.claimSeat(s.index)}>
                            Sit here
                          </button>
                        ) : (
                          <form
                            className="local-add"
                            onSubmit={(e) => {
                              e.preventDefault();
                              const n = (localName[s.index] ?? "").trim();
                              conn.claimSeat(s.index, n || `Player ${s.index + 1}`);
                            }}
                          >
                            <input
                              placeholder="Local player name"
                              maxLength={24}
                              value={localName[s.index] ?? ""}
                              onChange={(e) => setLocalName({ ...localName, [s.index]: e.target.value })}
                            />
                            <button className="btn small">Add</button>
                          </form>
                        )}
                      </>
                    )}
                    {mine && room.status !== "playing" && (
                      <button className="btn ghost small" onClick={() => conn.leaveSeat(s.index)}>
                        Leave
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="lobby-start">
            {host ? (
              <button className="btn good big" disabled={unclaimed > 0} onClick={conn.start}>
                {unclaimed > 0 ? `Waiting for ${unclaimed} player${unclaimed > 1 ? "s" : ""}` : "Start game"}
              </button>
            ) : (
              <div className="muted">Waiting for the host to start…</div>
            )}
          </div>
        </section>

        <section className="panel settings-panel">
          <h2>Table rules</h2>
          <label className="field">
            <span>Players</span>
            <select disabled={!host} value={settings.num_players} onChange={(e) => update({ num_players: +e.target.value })}>
              {[2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Points to win</span>
            <select disabled={!host} value={settings.vp_to_win} onChange={(e) => update({ vp_to_win: +e.target.value })}>
              {[6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Trade offers per turn</span>
            <select
              disabled={!host}
              value={settings.max_trade_offers_per_turn}
              onChange={(e) => update({ max_trade_offers_per_turn: +e.target.value })}
            >
              {[0, 1, 2, 3, 5, 8, 10].map((n) => (
                <option key={n} value={n}>
                  {n === 0 ? "No player trading" : n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>AI speed</span>
            <select
              disabled={!host}
              value={settings.bot_speed}
              onChange={(e) => update({ bot_speed: e.target.value as RoomSettings["bot_speed"] })}
            >
              <option value="slow">Relaxed</option>
              <option value="normal">Normal</option>
              <option value="fast">Fast</option>
            </select>
          </label>
          <label className="field check">
            <input
              type="checkbox"
              disabled={!host}
              checked={settings.beginner_board}
              onChange={(e) => update({ beginner_board: e.target.checked })}
            />
            <span>Beginner board layout</span>
          </label>
        </section>

        <section className="panel chat-panel">
          <h2>Table talk</h2>
          <ChatPanel conn={conn} state={state} />
        </section>
      </div>
      {conn.error && (
        <div className="error-toast" onClick={conn.clearError}>
          {conn.error}
        </div>
      )}
    </div>
  );
}
