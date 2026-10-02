import { useState, type ReactNode } from "react";
import type { Action, GameView, Hand, PlayerColor, SeatInfo } from "../lib/types";
import { RESOURCES, emptyHand, handTotal } from "../lib/types";
import { RESOURCE_LABEL } from "../lib/format";
import { HandStepper, ResourceCard } from "./Cards";
import { PLAYER_COLORS, SettlementShape } from "./art";

export function Modal({ title, children, onClose }: { title: ReactNode; children: ReactNode; onClose?: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

export function DiscardModal({ hand, count, onDiscard }: { hand: Hand; count: number; onDiscard: (h: Hand) => void }) {
  const [sel, setSel] = useState<Hand>(emptyHand());
  const left = count - handTotal(sel);
  return (
    <Modal title="The robber strikes!">
      <p>
        You hold too many cards. Choose <b>{count}</b> to discard.
      </p>
      <HandStepper label="Discard" value={sel} onChange={(h) => handTotal(h) <= count && setSel(h)} max={hand} />
      <div className="btn-row">
        <button className="btn good" disabled={left !== 0} onClick={() => onDiscard(sel)}>
          {left === 0 ? "Discard" : `Select ${left} more`}
        </button>
      </div>
    </Modal>
  );
}

export function ResourcePickerModal({
  title,
  hint,
  legal,
  onPick,
}: {
  title: string;
  hint: string;
  legal: Action[];
  onPick: (r: number) => void;
}) {
  const allowed = new Set(legal.flatMap((a) => (a.type === "choose_resource" ? [a.resource] : [])));
  return (
    <Modal title={title}>
      <p>{hint}</p>
      <div className="card-row center">
        {RESOURCES.map((r, i) => (
          <ResourceCard key={r} resource={r} onClick={allowed.has(i) ? () => onPick(i) : undefined} />
        ))}
      </div>
    </Modal>
  );
}

export function VictimModal({
  victims,
  seats,
  view,
  onPick,
  onCancel,
}: {
  victims: number[];
  seats: SeatInfo[];
  view: GameView;
  onPick: (v: number) => void;
  onCancel: () => void;
}) {
  return (
    <Modal title="Steal from whom?" onClose={onCancel}>
      <div className="victims">
        {victims.map((v) => (
          <button key={v} className="victim" onClick={() => onPick(v)} style={{ ["--pc" as string]: PLAYER_COLORS[seats[v].color].fill }}>
            <svg viewBox="-16 -18 32 34" width="34" height="34">
              <SettlementShape color={seats[v].color} />
            </svg>
            <span>{seats[v].name}</span>
            <small>{view.players[v].resource_count} cards</small>
          </button>
        ))}
      </div>
      <div className="btn-row">
        <button className="btn ghost" onClick={onCancel}>
          Pick another hex
        </button>
      </div>
    </Modal>
  );
}

export function PassDeviceCurtain({ name, color, onReveal }: { name: string; color: PlayerColor; onReveal: () => void }) {
  return (
    <div className="curtain">
      <div className="curtain-card" style={{ ["--pc" as string]: PLAYER_COLORS[color].fill }}>
        <svg viewBox="-16 -18 32 34" width="72" height="72">
          <SettlementShape color={color} />
        </svg>
        <h2>Pass the device to</h2>
        <h1>{name}</h1>
        <p className="muted">Other players, look away - your hand is about to be shown.</p>
        <button className="btn good big" onClick={onReveal}>
          I'm {name} - show my hand
        </button>
      </div>
    </div>
  );
}

export function GameOverModal({
  view,
  seats,
  isHost,
  onLobby,
  onClose,
}: {
  view: GameView;
  seats: SeatInfo[];
  isHost: boolean;
  onLobby: () => void;
  onClose: () => void;
}) {
  const rows = view.players
    .map((p) => ({ seat: p.seat, vp: p.total_vp ?? p.public_vp, p }))
    .sort((a, b) => b.vp - a.vp);
  const winner = view.winner;
  return (
    <Modal title={winner === null ? "Game over" : `${seats[winner].name} wins!`} onClose={onClose}>
      <table className="score-table">
        <thead>
          <tr>
            <th>Player</th>
            <th>VP</th>
            <th>Knights</th>
            <th>Road</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ seat, vp, p }) => (
            <tr key={seat} className={seat === winner ? "winner" : ""}>
              <td>
                <span className="dot" style={{ background: PLAYER_COLORS[seats[seat].color].fill }} /> {seats[seat].name}
              </td>
              <td>{vp}</td>
              <td>
                {p.knights_played}
                {p.has_largest_army ? " ★" : ""}
              </td>
              <td>
                {p.longest_road}
                {p.has_longest_road ? " ★" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="btn-row">
        {isHost && (
          <button className="btn good" onClick={onLobby}>
            Back to lobby
          </button>
        )}
        <button className="btn ghost" onClick={onClose}>
          View board
        </button>
      </div>
    </Modal>
  );
}

export function RulesModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Quick rules" onClose={onClose}>
      <ul className="rules">
        <li>Reach the target victory points first. Settlements are worth 1, cities 2, Longest Road and Largest Army 2 each.</li>
        <li>Roll the dice each turn: every tile showing that number produces for adjacent buildings (cities produce double).</li>
        <li>A 7 moves the robber: anyone holding more than 7 cards discards half, then the roller steals one card.</li>
        <li>Costs - Road: {RESOURCE_LABEL.wood} + {RESOURCE_LABEL.brick}. Settlement: {RESOURCE_LABEL.wood}, {RESOURCE_LABEL.brick}, {RESOURCE_LABEL.sheep}, {RESOURCE_LABEL.wheat}. City: 2 {RESOURCE_LABEL.wheat} + 3 {RESOURCE_LABEL.ore}. Development card: {RESOURCE_LABEL.sheep}, {RESOURCE_LABEL.wheat}, {RESOURCE_LABEL.ore}.</li>
        <li>Trade freely with other players on your turn: make an offer, others accept, decline or counter, and you pick a partner. Harbors improve bank rates to 3:1 or 2:1.</li>
        <li>One development card per turn, and never on the turn you bought it.</li>
      </ul>
      <div className="btn-row">
        <button className="btn" onClick={onClose}>
          Got it
        </button>
      </div>
    </Modal>
  );
}
