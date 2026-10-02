const PIPS: Record<number, [number, number][]> = {
  1: [[50, 50]],
  2: [
    [28, 28],
    [72, 72],
  ],
  3: [
    [26, 26],
    [50, 50],
    [74, 74],
  ],
  4: [
    [28, 28],
    [72, 28],
    [28, 72],
    [72, 72],
  ],
  5: [
    [27, 27],
    [73, 27],
    [50, 50],
    [27, 73],
    [73, 73],
  ],
  6: [
    [28, 24],
    [72, 24],
    [28, 50],
    [72, 50],
    [28, 76],
    [72, 76],
  ],
};

function Die({ value, red }: { value: number; red?: boolean }) {
  return (
    <svg viewBox="0 0 100 100" className="die">
      <rect x="4" y="4" width="92" height="92" rx="18" fill={red ? "#d8342a" : "#f7e9a8"} stroke="#3b2a14" strokeWidth="4" />
      <rect x="10" y="9" width="80" height="30" rx="12" fill="rgba(255,255,255,0.25)" />
      {PIPS[value]?.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="9" fill={red ? "#fff3d6" : "#2a1e10"} />
      ))}
    </svg>
  );
}

/** The two dice; `rollKey` changes on each new roll to replay the tumble animation. */
export function Dice({ roll, rollKey }: { roll: [number, number] | null; rollKey: number }) {
  if (!roll) {
    return (
      <div className="dice idle">
        <Die value={6} />
        <Die value={6} red />
      </div>
    );
  }
  return (
    <div className="dice" key={rollKey}>
      <div className="die-wrap tumble">
        <Die value={roll[0]} />
      </div>
      <div className="die-wrap tumble delay">
        <Die value={roll[1]} red />
      </div>
      <div className="dice-total">{roll[0] + roll[1]}</div>
    </div>
  );
}
