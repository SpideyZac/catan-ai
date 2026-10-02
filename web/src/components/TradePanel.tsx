import { useEffect, useState } from "react";
import type { Action, GameView, Hand, PlayerColor } from "../lib/types";
import { RESOURCES, covers, emptyHand, handTotal } from "../lib/types";
import { RESOURCE_LABEL } from "../lib/format";
import { HandChips, HandStepper, ResourceCard } from "./Cards";
import { PLAYER_COLORS, ResourceIcon } from "./art";

interface Props {
  view: GameView;
  me: number;
  legal: Action[];
  canOffer: boolean;
  names: string[];
  colors: PlayerColor[];
  act: (a: Action) => void;
}

function ratio(view: GameView, me: number, r: number): number {
  const ports = view.players[me].ports;
  if (ports.includes(RESOURCES[r])) return 2;
  if (ports.includes("generic")) return 3;
  return 4;
}

function Who({ seat, names, colors }: { seat: number; names: string[]; colors: PlayerColor[] }) {
  return (
    <b className="who" style={{ ["--pc" as string]: PLAYER_COLORS[colors[seat]].fill }}>
      {names[seat]}
    </b>
  );
}

/** Domestic (player-to-player) and maritime (bank/harbor) trading. */
export function TradePanel({ view, me, legal, canOffer, names, colors, act }: Props) {
  const [give, setGive] = useState<Hand>(emptyHand());
  const [want, setWant] = useState<Hand>(emptyHand());
  const [countering, setCountering] = useState(false);
  const [bankGive, setBankGive] = useState<number | null>(null);
  const hand = view.players[me].resources ?? emptyHand();
  const trade = view.trade;
  const phase = view.phase.name;

  // Reset the builder whenever a new offer appears or the turn changes.
  useEffect(() => {
    setCountering(false);
  }, [trade?.proposer, trade?.give.join(), trade?.want.join(), view.turn]);

  const overlap = give.some((g, i) => g > 0 && want[i] > 0);
  const valid = handTotal(give) > 0 && handTotal(want) > 0 && !overlap && covers(hand, give);

  // ---------------------------------------------------------- responding to someone else's offer
  if (trade && trade.proposer !== me && phase === "trade_response") {
    const my = trade.responses[me];
    const canAccept = legal.some((a) => a.type === "accept_trade");
    if (my?.kind !== "pending") {
      return (
        <div className="trade-panel">
          <h3>Trade offer</h3>
          <p className="muted">You responded. Waiting for others…</p>
        </div>
      );
    }
    return (
      <div className="trade-panel incoming">
        <h3>
          <Who seat={trade.proposer} names={names} colors={colors} /> wants to trade
        </h3>
        <div className="offer-line">
          <span>You get</span> <HandChips hand={trade.give} />
        </div>
        <div className="offer-line">
          <span>You give</span> <HandChips hand={trade.want} />
        </div>
        {!countering ? (
          <div className="btn-row">
            <button className="btn good" disabled={!canAccept} onClick={() => act({ type: "accept_trade" })}>
              {canAccept ? "Accept" : "Can't afford"}
            </button>
            <button className="btn" onClick={() => act({ type: "reject_trade" })}>
              Decline
            </button>
            <button
              className="btn ghost"
              onClick={() => {
                setGive([...trade.want] as Hand);
                setWant([...trade.give] as Hand);
                setCountering(true);
              }}
            >
              Counter…
            </button>
          </div>
        ) : (
          <>
            <HandStepper label="You give" value={give} onChange={setGive} max={hand} />
            <HandStepper label="You want" value={want} onChange={setWant} />
            <div className="btn-row">
              <button className="btn good" disabled={!valid} onClick={() => act({ type: "offer_trade", give, want })}>
                Send counter
              </button>
              <button className="btn ghost" onClick={() => setCountering(false)}>
                Back
              </button>
            </div>
          </>
        )}
      </div>
    );
  }

  // ---------------------------------------------------------- my offer is out
  if (trade && trade.proposer === me && (phase === "trade_response" || phase === "trade_confirm")) {
    const confirmable = legal.filter((a): a is Extract<Action, { type: "confirm_trade" }> => a.type === "confirm_trade");
    return (
      <div className="trade-panel">
        <h3>Your offer</h3>
        <div className="offer-line">
          <span>Give</span> <HandChips hand={trade.give} />
        </div>
        <div className="offer-line">
          <span>Get</span> <HandChips hand={trade.want} />
        </div>
        <ul className="responses">
          {trade.responses.map((r, seat) =>
            r.kind === "not_involved" ? null : (
              <li key={seat} className={`resp ${r.kind}`}>
                <Who seat={seat} names={names} colors={colors} />
                <span className="resp-kind">
                  {r.kind === "pending" && "thinking…"}
                  {r.kind === "accept" && "accepts"}
                  {r.kind === "reject" && "declines"}
                  {r.kind === "counter" && (
                    <>
                      counters: you give <HandChips hand={r.give} /> for <HandChips hand={r.want} />
                    </>
                  )}
                </span>
                {confirmable.some((a) => a.partner === seat) && (
                  <button className="btn good small" onClick={() => act({ type: "confirm_trade", partner: seat })}>
                    Trade
                  </button>
                )}
              </li>
            ),
          )}
        </ul>
        <div className="btn-row">
          <button className="btn" onClick={() => act({ type: "cancel_trade" })}>
            {phase === "trade_confirm" ? "No deal" : "Withdraw offer"}
          </button>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------- composing (my main phase)
  const maritime = legal.filter((a): a is Extract<Action, { type: "maritime_trade" }> => a.type === "maritime_trade");
  const myTurnMain = phase === "main" && view.current === me;
  if (!myTurnMain) {
    return (
      <div className="trade-panel">
        <h3>Trading</h3>
        <p className="muted">Trading opens after you roll on your turn.</p>
      </div>
    );
  }
  return (
    <div className="trade-panel">
      <h3>
        Trade with players <small className="muted">({view.trade_offers_left} offers left)</small>
      </h3>
      {canOffer || view.trade_offers_left > 0 ? (
        <>
          <HandStepper label="You give" value={give} onChange={setGive} max={hand} />
          <HandStepper label="You want" value={want} onChange={setWant} />
          <div className="btn-row">
            <button
              className="btn good"
              disabled={!valid || view.trade_offers_left <= 0}
              onClick={() => {
                act({ type: "offer_trade", give, want });
              }}
            >
              Offer to table
            </button>
            <button
              className="btn ghost"
              onClick={() => {
                setGive(emptyHand());
                setWant(emptyHand());
              }}
            >
              Clear
            </button>
          </div>
          {overlap && <p className="warn">You can't give and receive the same resource.</p>}
        </>
      ) : (
        <p className="muted">No offers left this turn.</p>
      )}

      <h3 className="mt">Trade with the bank</h3>
      <div className="bank-row">
        {RESOURCES.map((r, i) => {
          const rate = ratio(view, me, i);
          const enabled = maritime.some((a) => a.give === i);
          return (
            <div key={r} className="bank-give">
              <ResourceCard resource={r} small count={hand[i]} selected={bankGive === i} onClick={enabled ? () => setBankGive(i) : undefined} />
              <span className={`rate ${rate < 4 ? "port" : ""}`}>{rate}:1</span>
            </div>
          );
        })}
      </div>
      {bankGive !== null && (
        <div className="bank-get">
          <span>
            Give {ratio(view, me, bankGive)} {RESOURCE_LABEL[RESOURCES[bankGive]]} for:
          </span>
          <div className="chips">
            {RESOURCES.map((r, i) =>
              maritime.some((a) => a.give === bankGive && a.get === i) ? (
                <button
                  key={r}
                  className="chip-btn"
                  onClick={() => {
                    act({ type: "maritime_trade", give: bankGive, get: i });
                    setBankGive(null);
                  }}
                >
                  <ResourceIcon resource={r} size={22} /> {RESOURCE_LABEL[r]}
                </button>
              ) : null,
            )}
          </div>
        </div>
      )}
    </div>
  );
}
