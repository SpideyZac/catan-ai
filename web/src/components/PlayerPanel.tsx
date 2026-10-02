import type { GameView, SeatInfo } from "../lib/types";
import { PLAYER_COLORS, SettlementShape } from "./art";

function botLabel(bot: string | null): string {
  if (!bot || bot === "heuristic") return "Standard";
  if (bot === "random") return "Beginner";
  return bot.startsWith("neural:") ? "Neural" : bot;
}

function statusOf(view: GameView, seat: number): string | null {
  const p = view.players[seat];
  if (view.winner === seat) return "Winner!";
  if (p.discard_pending > 0) return `Discarding ${p.discard_pending}`;
  if (view.phase.name === "trade_response" && view.trade) {
    const r = view.trade.responses[seat];
    if (r?.kind === "pending") return "Considering offer…";
    if (r?.kind === "accept") return "Accepted";
    if (r?.kind === "reject") return "Declined";
    if (r?.kind === "counter") return "Countered";
  }
  return null;
}

export function PlayerPanel({
  view,
  seat,
  info,
  isMe,
  compact,
  onReplace,
}: {
  view: GameView;
  seat: number;
  info: SeatInfo;
  isMe: boolean;
  compact?: boolean;
  /** Host-only: hand an offline player's seat to the AI. */
  onReplace?: () => void;
}) {
  const p = view.players[seat];
  const c = PLAYER_COLORS[info.color];
  const current = view.current === seat && view.phase.name !== "game_over";
  const status = statusOf(view, seat);
  const vp = p.total_vp ?? p.public_vp;
  return (
    <div
      className={`ppanel ${current ? "current" : ""} ${isMe ? "me" : ""} ${compact ? "compact" : ""}`}
      style={{ ["--pc" as string]: c.fill, ["--pcs" as string]: c.stroke }}
    >
      <div className="ppanel-head">
        <svg viewBox="-16 -18 32 34" className="ppanel-piece">
          <SettlementShape color={info.color} />
        </svg>
        <div className="ppanel-name">
          <span className="name">{info.name}</span>
          <span className="sub">
            {info.kind === "bot" ? `AI · ${botLabel(info.bot)}` : isMe ? "You" : "Player"}
            {!info.connected && info.kind === "human" && <span className="offline"> · offline</span>}
          </span>
        </div>
        <div className="ppanel-vp" title={p.total_vp !== null && p.total_vp !== p.public_vp ? "Includes hidden VP cards" : "Victory points"}>
          <b>{vp}</b>
          <small>VP</small>
        </div>
      </div>
      <div className="ppanel-stats">
        <span title="Resource cards" className="stat">
          <i className="ico cards" /> {p.resource_count}
        </span>
        <span title="Development cards" className="stat">
          <i className="ico dev" /> {p.dev_card_count}
        </span>
        <span title="Knights played" className={`stat ${p.has_largest_army ? "award" : ""}`}>
          <i className="ico knight" /> {p.knights_played}
        </span>
        <span title="Longest road" className={`stat ${p.has_longest_road ? "award" : ""}`}>
          <i className="ico road" /> {p.longest_road}
        </span>
      </div>
      {(p.has_longest_road || p.has_largest_army) && !compact && (
        <div className="badges">
          {p.has_longest_road && <span className="badge">Longest Road</span>}
          {p.has_largest_army && <span className="badge">Largest Army</span>}
        </div>
      )}
      {status && <div className="ppanel-status">{status}</div>}
      {onReplace && (
        <button className="btn ghost small replace" onClick={onReplace} title="This player is offline">
          Replace with AI
        </button>
      )}
    </div>
  );
}
