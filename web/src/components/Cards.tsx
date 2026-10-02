import type { DevCardName, Hand, ResourceName } from "../lib/types";
import { RESOURCES } from "../lib/types";
import { DEV_HELP, DEV_LABEL, RESOURCE_LABEL } from "../lib/format";
import { DevIcon, RESOURCE_COLORS, ResourceIcon } from "./art";

export function ResourceCard({
  resource,
  count,
  onClick,
  selected,
  disabled,
  small,
}: {
  resource: ResourceName;
  count?: number;
  onClick?: () => void;
  selected?: boolean;
  disabled?: boolean;
  small?: boolean;
}) {
  return (
    <button
      type="button"
      className={`rcard ${small ? "small" : ""} ${selected ? "selected" : ""} ${count === 0 ? "empty" : ""}`}
      style={{ ["--rc" as string]: RESOURCE_COLORS[resource] }}
      onClick={onClick}
      disabled={disabled || !onClick}
      title={RESOURCE_LABEL[resource]}
    >
      <span className="rcard-art">
        <ResourceIcon resource={resource} size={small ? 22 : 34} />
      </span>
      {!small && <span className="rcard-label">{RESOURCE_LABEL[resource]}</span>}
      {count !== undefined && <span className="rcard-count">{count}</span>}
    </button>
  );
}

export function DevCard({
  card,
  count,
  fresh,
  onPlay,
}: {
  card: DevCardName;
  count: number;
  fresh: number;
  onPlay?: () => void;
}) {
  return (
    <button
      type="button"
      className={`dcard ${onPlay ? "playable" : ""}`}
      onClick={onPlay}
      disabled={!onPlay}
      title={`${DEV_LABEL[card]}: ${DEV_HELP[card]}${fresh ? " (bought this turn - playable next turn)" : ""}`}
    >
      <DevIcon card={card} size={30} />
      <span className="dcard-label">{DEV_LABEL[card]}</span>
      <span className="rcard-count">{count + fresh}</span>
      {fresh > 0 && <span className="dcard-new">new</span>}
    </button>
  );
}

/** Inline resource multiset, e.g. in trade offers and logs. */
export function HandChips({ hand, empty = "nothing" }: { hand: Hand; empty?: string }) {
  const items = hand.flatMap((n, i) => (n > 0 ? [[RESOURCES[i], n] as const] : []));
  if (!items.length) return <span className="muted">{empty}</span>;
  return (
    <span className="chips">
      {items.map(([r, n]) => (
        <span key={r} className="chip" style={{ ["--rc" as string]: RESOURCE_COLORS[r] }}>
          <ResourceIcon resource={r} size={16} />
          {n > 1 && <b>×{n}</b>}
        </span>
      ))}
    </span>
  );
}

/** Row of +/- steppers for building a resource multiset. */
export function HandStepper({
  value,
  onChange,
  max,
  label,
}: {
  value: Hand;
  onChange: (h: Hand) => void;
  max?: Hand;
  label: string;
}) {
  const set = (i: number, d: number) => {
    const next = [...value] as Hand;
    next[i] = Math.max(0, Math.min(max ? max[i] : 9, next[i] + d));
    onChange(next);
  };
  return (
    <div className="stepper">
      <div className="stepper-label">{label}</div>
      <div className="stepper-row">
        {RESOURCES.map((r, i) => (
          <div key={r} className="step" style={{ ["--rc" as string]: RESOURCE_COLORS[r] }}>
            <button type="button" className="step-btn" onClick={() => set(i, 1)} disabled={max && value[i] >= max[i]}>
              +
            </button>
            <div className={`step-val ${value[i] ? "on" : ""}`}>
              <ResourceIcon resource={r} size={22} />
              <b>{value[i]}</b>
            </div>
            <button type="button" className="step-btn" onClick={() => set(i, -1)} disabled={!value[i]}>
              −
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
