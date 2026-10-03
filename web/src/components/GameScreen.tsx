import { useEffect, useMemo, useRef, useState } from "react";
import type { RoomConnection } from "../lib/api";
import type { Action, GameEvent, Hand, LogEntry, StateMessage } from "../lib/types";
import { COSTS, DEV_CARDS, RESOURCES, covers, emptyHand } from "../lib/types";
import { DEV_LABEL, PHASE_HINT, describe } from "../lib/format";
import { type Sfx, play, sfxFor } from "../lib/sound";
import { useMusic } from "../lib/music";
import { MusicToggle, SoundToggle } from "./AudioToggles";
import { Board, NO_TARGETS, type BoardTargets } from "./Board";
import { DevCard, HandChips, ResourceCard } from "./Cards";
import { Dice } from "./Dice";
import { DiscardModal, GameOverModal, PassDeviceCurtain, ResourcePickerModal, RulesModal, VictimModal } from "./Modals";
import { PlayerPanel } from "./PlayerPanel";
import { TradePanel } from "./TradePanel";
import { PLAYER_COLORS } from "./art";

type BuildMode = "road" | "settlement" | "city" | null;
type SideTab = "trade" | "log" | "chat";

interface Toast {
  id: number;
  text: string;
  tone: "info" | "good" | "bad";
}

const TOAST_EVENTS = new Set<GameEvent["type"]>([
  "stolen",
  "trade_executed",
  "longest_road_changed",
  "largest_army_changed",
  "dev_card_played",
  "monopoly_taken",
  "game_won",
]);

export function GameScreen({ conn, state, onLeave }: { conn: RoomConnection; state: StateMessage; onLeave: () => void }) {
  const { room, you } = state;
  const game = state.game!;
  const seats = room.seats;
  const colors = seats.map((s) => s.color);
  const names = seats.map((s) => s.name);
  const mySeats = you.seats;
  const multi = mySeats.length > 1;
  const anyView = Object.values(game.views)[0];
  const actors = anyView.actors;
  const phase = anyView.phase.name;

  // ---------------------------------------------------------------- perspective (pass & play)
  const [focus, setFocus] = useState<number | null>(mySeats[0] ?? null);
  const [revealed, setRevealed] = useState<number | null>(multi ? null : (mySeats[0] ?? null));
  useEffect(() => {
    const acting = mySeats.filter((s) => actors.includes(s));
    let next: number | null = focus !== null && mySeats.includes(focus) ? focus : (mySeats[0] ?? null);
    if (acting.length) next = acting.includes(anyView.current) ? anyView.current : acting[0];
    if (anyView.trade && phase === "trade_response" && mySeats.includes(anyView.trade.proposer) && !acting.length) {
      next = anyView.trade.proposer;
    }
    if (next !== focus) setFocus(next);
    if (!multi && next !== revealed) setRevealed(next);
  }, [actors.join(), anyView.current, mySeats.join(), phase]); // eslint-disable-line react-hooks/exhaustive-deps

  const showCurtain = multi && focus !== null && revealed !== focus && actors.includes(focus) && phase !== "game_over";
  const view = (focus !== null ? game.views[String(focus)] : game.views.spectator) ?? anyView;
  const legal: Action[] = focus !== null && !showCurtain ? (game.legal[String(focus)] ?? []) : [];
  const canOffer = focus !== null ? !!game.can_offer[String(focus)] : false;
  const me = focus;
  const myHand: Hand = (me !== null && view.players[me].resources) || emptyHand();
  const has = (t: Action["type"]) => legal.some((a) => a.type === t);
  const act = (a: Action) => {
    if (me !== null) conn.act(me, a);
  };

  // ---------------------------------------------------------------- build mode & board targets
  const [mode, setMode] = useState<BuildMode>(null);
  const [robberHex, setRobberHex] = useState<number | null>(null);
  useEffect(() => {
    setMode(null);
    setRobberHex(null);
  }, [phase, view.current, game.id]);

  const targets: BoardTargets = useMemo(() => {
    if (!legal.length) return NO_TARGETS;
    const vertices = new Set<number>();
    const edges = new Set<number>();
    const hexes = new Set<number>();
    let vertexKind: BoardTargets["vertexKind"] = null;
    const auto = phase === "setup_settlement" || phase === "setup_road" || phase === "road_building";
    for (const a of legal) {
      if (a.type === "build_settlement" && (auto || mode === "settlement")) {
        vertices.add(a.vertex);
        vertexKind = "settlement";
      } else if (a.type === "build_city" && mode === "city") {
        vertices.add(a.vertex);
        vertexKind = "city";
      } else if (a.type === "build_road" && (auto || mode === "road")) edges.add(a.edge);
      else if (a.type === "move_robber") hexes.add(a.hex);
    }
    return { vertices, edges, hexes, vertexKind };
  }, [legal, mode, phase]);

  const onVertex = (v: number) => {
    const a = legal.find(
      (x) =>
        (x.type === "build_settlement" && x.vertex === v && targets.vertexKind === "settlement") ||
        (x.type === "build_city" && x.vertex === v && targets.vertexKind === "city"),
    );
    if (a) act(a);
    setMode(null);
  };
  const onEdge = (e: number) => {
    const a = legal.find((x) => x.type === "build_road" && x.edge === e);
    if (a) act(a);
    if (phase === "main") setMode(null);
  };
  const robberOptions = (h: number) =>
    legal.filter((a): a is Extract<Action, { type: "move_robber" }> => a.type === "move_robber" && a.hex === h);
  const onHex = (h: number) => {
    const opts = robberOptions(h);
    if (opts.length === 1) act(opts[0]);
    else if (opts.length > 1) setRobberHex(h);
  };

  // ---------------------------------------------------------------- events: dice + toasts
  const lastRoll = useMemo(() => {
    for (let i = game.log.length - 1; i >= 0; i--) {
      const e = game.log[i];
      if (e.event.type === "dice_rolled") return { dice: e.event.dice, seq: e.seq };
    }
    return null;
  }, [game.log]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seenSeq = useRef<number>(game.log.length ? game.log[game.log.length - 1].seq : 0);
  const nameOf = (s: number) => (mySeats.includes(s) && !multi ? "You" : names[s]);
  useEffect(() => {
    const fresh = game.log.filter((e) => e.seq > seenSeq.current);
    if (!fresh.length) return;
    seenSeq.current = fresh[fresh.length - 1].seq;
    const add: Toast[] = [];
    const sfx: Sfx[] = [];
    const ownSeat = (s: number | null) => s !== null && (mySeats.includes(s) || !mySeats.length);
    for (const { seq, event } of fresh) {
      const fx = sfxFor(event, ownSeat);
      if (fx) sfx.push(fx);
      if (event.type === "dice_rolled" && event.dice[0] + event.dice[1] === 7) sfx.push("seven");
      const mine =
        (event.type === "stolen" && (mySeats.includes(event.victim) || mySeats.includes(event.thief))) ||
        (event.type === "trade_executed" && (mySeats.includes(event.partner) || mySeats.includes(event.proposer)));
      if (event.type === "trade_cancelled" && mySeats.includes(event.proposer)) {
        add.push({ id: seq, text: "No deal - your offer was closed", tone: "info" });
        continue;
      }
      if (event.type === "trade_responded" && event.response.kind === "counter" && anyView.current !== undefined && mySeats.includes(anyView.current)) {
        add.push({ id: seq, text: `${names[event.player]} sent a counter-offer`, tone: "good" });
        continue;
      }
      if (TOAST_EVENTS.has(event.type) || (event.type === "dice_rolled" && event.dice[0] + event.dice[1] === 7)) {
        const tone: Toast["tone"] =
          event.type === "stolen" && mySeats.includes(event.victim) ? "bad" : mine ? "good" : "info";
        add.push({ id: seq, text: describe(event, nameOf), tone });
      }
    }
    play(sfx);
    if (!add.length) return;
    setToasts((t) => [...t, ...add].slice(-4));
    const ids = add.map((t) => t.id);
    const timer = window.setTimeout(() => setToasts((t) => t.filter((x) => !ids.includes(x.id))), 4500);
    return () => window.clearTimeout(timer);
  }, [game.log]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------- sounds: music, turn + chat
  useMusic("game");
  const prevCurrent = useRef(anyView.current);
  useEffect(() => {
    if (anyView.current !== prevCurrent.current && mySeats.includes(anyView.current) && phase !== "game_over") {
      window.setTimeout(() => play(["your_turn"]), 250);
    }
    prevCurrent.current = anyView.current;
  }, [anyView.current]); // eslint-disable-line react-hooks/exhaustive-deps
  const chatSeen = useRef(state.room.chat.length);
  useEffect(() => {
    const chat = state.room.chat;
    if (chat.slice(chatSeen.current).some((m) => m.client_id !== you.client_id)) play(["chat"]);
    chatSeen.current = chat.length;
  }, [state.room.chat.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------- side panel
  const [tab, setTab] = useState<SideTab>("log");
  const tradeActive = phase === "trade_response" || phase === "trade_confirm";
  useEffect(() => {
    if (tradeActive && me !== null && (anyView.trade?.proposer === me || actors.includes(me))) setTab("trade");
  }, [tradeActive, me, actors.join()]); // eslint-disable-line react-hooks/exhaustive-deps
  const [showRules, setShowRules] = useState(false);
  const [gameOverDismissed, setGameOverDismissed] = useState(false);
  useEffect(() => setGameOverDismissed(false), [game.id]);

  // ---------------------------------------------------------------- derived UI state
  const current = view.current;
  const myTurn = me !== null && current === me;
  const discardPending = me !== null ? view.players[me].discard_pending : 0;
  const hint =
    phase === "game_over"
      ? "Game over"
      : actors.length > 1 && phase === "discard"
        ? "Players are discarding"
        : `${names[current]} · ${PHASE_HINT[phase] ?? phase}`;
  const devCounts = me !== null ? view.players[me].dev_cards : null;
  const devNew = me !== null ? view.players[me].new_dev_cards : null;
  const devPlay: Partial<Record<(typeof DEV_CARDS)[number], Action>> = {
    knight: has("play_knight") ? { type: "play_knight" } : undefined,
    road_building: has("play_road_building") ? { type: "play_road_building" } : undefined,
    year_of_plenty: has("play_year_of_plenty") ? { type: "play_year_of_plenty" } : undefined,
    monopoly: has("play_monopoly") ? { type: "play_monopoly" } : undefined,
  };
  const buildable = (k: Exclude<BuildMode, null>) =>
    legal.some((a) => a.type === (k === "road" ? "build_road" : k === "settlement" ? "build_settlement" : "build_city"));

  return (
    <div className="game">
      <header className="topbar">
        <div className="brand small" onClick={onLeave} role="button" title="Leave game">
          <span className="logo-hex" /> Catan <em>AI</em>
        </div>
        <div className="turn-hint" style={{ ["--pc" as string]: PLAYER_COLORS[colors[current]].fill }}>
          <span className="dot" /> {hint}
        </div>
        <div className="topbar-right">
          <span className="room-code" title="Room code">
            {room.code}
          </span>
          <span className={`conn ${conn.status}`} title={`Connection: ${conn.status}`} />
          <MusicToggle />
          <SoundToggle />
          <button className="btn ghost small" onClick={() => setShowRules(true)}>
            Rules
          </button>
        </div>
      </header>

      <aside className="players">
        {seats.map((s) => (
          <PlayerPanel
            key={s.index}
            view={view}
            seat={s.index}
            info={s}
            isMe={mySeats.includes(s.index)}
            onReplace={
              you.is_host && s.kind === "human" && !s.connected && phase !== "game_over"
                ? () => conn.setSeat(s.index, "bot", "heuristic")
                : undefined
            }
          />
        ))}
        <div className="bank">
          <div className="bank-title">Bank</div>
          <div className="bank-cards">
            {RESOURCES.map((r, i) => (
              <ResourceCard key={r} resource={r} small count={view.bank[i]} />
            ))}
          </div>
          <div className="muted small-text">{view.dev_deck_count} development cards left</div>
        </div>
      </aside>

      <main className="board-wrap">
        <Board
          board={game.board}
          view={view}
          colors={colors}
          targets={showCurtain ? NO_TARGETS : targets}
          activeColor={me !== null ? colors[me] : null}
          rolled={lastRoll ? { total: lastRoll.dice[0] + lastRoll.dice[1], key: lastRoll.seq } : null}
          onVertex={onVertex}
          onEdge={onEdge}
          onHex={onHex}
        />
        <div className="dice-float">
          <Dice roll={lastRoll?.dice ?? null} rollKey={lastRoll?.seq ?? 0} />
        </div>
        <div className="toasts">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.tone}`}>
              {t.text}
            </div>
          ))}
        </div>
        {mode && (
          <div className="mode-banner">
            Placing a {mode} — click a highlighted spot
            <button className="btn ghost small" onClick={() => setMode(null)}>
              Cancel
            </button>
          </div>
        )}
      </main>

      <aside className="side">
        <nav className="tabs">
          {(["trade", "log", "chat"] as SideTab[]).map((t) => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t === "trade" ? "Trade" : t === "log" ? "Log" : "Chat"}
              {t === "trade" && tradeActive && <span className="pip" />}
            </button>
          ))}
        </nav>
        <div className="tab-body">
          {tab === "trade" &&
            (me !== null ? (
              <TradePanel view={view} me={me} legal={legal} canOffer={canOffer} names={names} colors={colors} act={act} />
            ) : (
              <p className="muted pad">Spectators can't trade.</p>
            ))}
          {tab === "log" && <LogPanel log={game.log} nameOf={nameOf} colors={colors} />}
          {tab === "chat" && <ChatPanel conn={conn} state={state} />}
        </div>
      </aside>

      <footer className="dock">
        {me === null ? (
          <div className="spectating">Spectating · {room.code}</div>
        ) : showCurtain ? (
          <div className="spectating">Hand hidden</div>
        ) : (
          <>
            <div className="hand">
              {RESOURCES.map((r, i) => (
                <ResourceCard key={r} resource={r} count={myHand[i]} />
              ))}
              <div className="devs">
                {DEV_CARDS.map((d, i) =>
                  devCounts && devNew && devCounts[i] + devNew[i] > 0 ? (
                    <DevCard
                      key={d}
                      card={d}
                      count={devCounts[i]}
                      fresh={devNew[i]}
                      onPlay={devPlay[d] ? () => act(devPlay[d]!) : undefined}
                    />
                  ) : null,
                )}
              </div>
            </div>
            <div className="actions">
              {has("roll_dice") && (
                <button className="btn roll" onClick={() => act({ type: "roll_dice" })}>
                  Roll dice
                </button>
              )}
              {phase === "main" && myTurn && (
                <>
                  {(["road", "settlement", "city"] as const).map((k) => (
                    <button
                      key={k}
                      className={`btn build ${mode === k ? "active" : ""}`}
                      disabled={!buildable(k)}
                      onClick={() => setMode(mode === k ? null : k)}
                      title={`Cost: ${COSTS[k].map((n, i) => (n ? `${n} ${RESOURCES[i]}` : "")).filter(Boolean).join(", ")}`}
                    >
                      <span>{k[0].toUpperCase() + k.slice(1)}</span>
                      <HandChips hand={COSTS[k]} />
                    </button>
                  ))}
                  <button
                    className="btn build"
                    disabled={!has("buy_dev_card")}
                    onClick={() => act({ type: "buy_dev_card" })}
                    title="Buy a development card"
                  >
                    <span>Dev card</span>
                    <HandChips hand={COSTS.dev} />
                  </button>
                  <button className="btn end" disabled={!has("end_turn")} onClick={() => act({ type: "end_turn" })}>
                    End turn
                  </button>
                </>
              )}
              {!myTurn && phase !== "game_over" && !legal.length && (
                <div className="waiting">Waiting for {names[current]}…</div>
              )}
              {myTurn && phase === "main" && !covers(myHand, COSTS.road) && !has("buy_dev_card") && (
                <div className="tip">Tip: open Trade to swap cards with players or the bank.</div>
              )}
            </div>
          </>
        )}
      </footer>

      {/* Modals */}
      {showCurtain && focus !== null && (
        <PassDeviceCurtain name={names[focus]} color={colors[focus]} onReveal={() => setRevealed(focus)} />
      )}
      {!showCurtain && phase === "discard" && discardPending > 0 && has("discard") && (
        <DiscardModal
          key={`${view.turn}-${me}`}
          hand={myHand}
          count={discardPending}
          onDiscard={(h) => h.forEach((n, r) => Array.from({ length: n }).forEach(() => act({ type: "discard", resource: r })))}
        />
      )}
      {!showCurtain && phase === "year_of_plenty" && myTurn && (
        <ResourcePickerModal
          title={DEV_LABEL.year_of_plenty}
          hint={`Choose a resource to take from the bank (${view.phase.remaining ?? 1} left).`}
          legal={legal}
          onPick={(r) => act({ type: "choose_resource", resource: r })}
        />
      )}
      {!showCurtain && phase === "monopoly" && myTurn && (
        <ResourcePickerModal
          title={DEV_LABEL.monopoly}
          hint="Every opponent gives you all of the resource you name."
          legal={legal}
          onPick={(r) => act({ type: "choose_resource", resource: r })}
        />
      )}
      {robberHex !== null && (
        <VictimModal
          victims={robberOptions(robberHex).flatMap((a) => (a.victim !== null ? [a.victim] : []))}
          seats={seats}
          view={view}
          onPick={(v) => {
            act({ type: "move_robber", hex: robberHex, victim: v });
            setRobberHex(null);
          }}
          onCancel={() => setRobberHex(null)}
        />
      )}
      {phase === "game_over" && !gameOverDismissed && (
        <GameOverModal
          view={view}
          seats={seats}
          isHost={you.is_host}
          onLobby={conn.backToLobby}
          onClose={() => setGameOverDismissed(true)}
        />
      )}
      {showRules && <RulesModal onClose={() => setShowRules(false)} />}
      {conn.error && (
        <div className="error-toast" onClick={conn.clearError}>
          {conn.error}
        </div>
      )}
    </div>
  );
}

function LogPanel({ log, nameOf, colors }: { log: LogEntry[]; nameOf: (s: number) => string; colors: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [log.length]);
  const actor = (e: GameEvent): number | null =>
    "player" in e && typeof e.player === "number"
      ? e.player
      : "proposer" in e
        ? e.proposer
        : "thief" in e
          ? e.thief
          : null;
  return (
    <div className="log" ref={ref}>
      {log.map(({ seq, event }) => {
          const who = actor(event);
          const color = who !== null ? PLAYER_COLORS[colors[who] as keyof typeof PLAYER_COLORS]?.fill : "#999";
          return (
            <div key={seq} className={`log-line ${event.type}`} style={{ ["--pc" as string]: color }}>
              {describe(event, nameOf)}
            </div>
          );
        })}
    </div>
  );
}

export function ChatPanel({ conn, state }: { conn: RoomConnection; state: StateMessage }) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const chat = state.room.chat;
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [chat.length]);
  return (
    <div className="chat">
      <div className="chat-lines" ref={ref}>
        {chat.length === 0 && <p className="muted">No messages yet. Say hi!</p>}
        {chat.map((m, i) => (
          <div key={i} className={`chat-line ${m.client_id === state.you.client_id ? "mine" : ""}`}>
            <b>{m.from}</b> {m.text}
          </div>
        ))}
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) conn.chat(text.trim());
          setText("");
        }}
      >
        <input value={text} maxLength={300} onChange={(e) => setText(e.target.value)} placeholder="Message…" />
        <button className="btn small">Send</button>
      </form>
    </div>
  );
}
