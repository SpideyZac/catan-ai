// Procedural SVG artwork: terrain illustrations, resource / development card icons and pieces.
// Everything is vector so the board stays crisp at any size and needs no image assets.
import type { ReactElement } from "react";
import type { DevCardName, PlayerColor, ResourceName, Terrain } from "../lib/types";

export const PLAYER_COLORS: Record<PlayerColor, { fill: string; stroke: string; light: string }> = {
  red: { fill: "#d33a2c", stroke: "#6e1410", light: "#ff8a7a" },
  blue: { fill: "#2c6bd3", stroke: "#0f2d63", light: "#8db7ff" },
  white: { fill: "#f4f1ea", stroke: "#4a4337", light: "#ffffff" },
  orange: { fill: "#f08a1c", stroke: "#7a3d05", light: "#ffc27a" },
};

export const RESOURCE_COLORS: Record<ResourceName, string> = {
  wood: "#2f7a32",
  brick: "#c4562b",
  sheep: "#8fd15c",
  wheat: "#f1c232",
  ore: "#8e96a3",
};

export const TERRAIN_GRADIENT: Record<Terrain, [string, string]> = {
  forest: ["#3f8a3a", "#1f5a22"],
  hills: ["#d9824b", "#a94d24"],
  pasture: ["#a6dc6c", "#6fb043"],
  fields: ["#f4d061", "#d9a92c"],
  mountains: ["#a3a8b3", "#6b717d"],
  desert: ["#f0dca4", "#d8bd7c"],
};

/** Deterministic PRNG so each tile always gets the same decoration. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Scatter points inside a hex of radius r (pointy-top) avoiding the center token. */
function scatter(seed: number, count: number, r: number, avoid = 0.36): [number, number][] {
  const rnd = mulberry32(seed);
  const pts: [number, number][] = [];
  let guard = 0;
  while (pts.length < count && guard++ < 500) {
    const x = (rnd() * 2 - 1) * r * 0.8;
    const y = (rnd() * 2 - 1) * r * 0.78;
    // Inside pointy-top hex (approximate) and away from the number token.
    if (Math.abs(x) > r * 0.8 || Math.abs(y) + Math.abs(x) * 0.55 > r * 0.86) continue;
    if (Math.hypot(x, y) < r * avoid) continue;
    if (pts.some(([px, py]) => Math.hypot(px - x, py - y) < r * 0.2)) continue;
    pts.push([x, y]);
  }
  return pts.sort((a, b) => a[1] - b[1]);
}

function Pine({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <ellipse cx="0" cy="13" rx="9" ry="3" fill="rgba(0,0,0,0.25)" />
      <rect x="-2" y="6" width="4" height="8" fill="#5b3a1e" />
      <polygon points="0,-18 -11,2 11,2" fill="#2d6e2e" />
      <polygon points="0,-12 -13,9 13,9" fill="#25602a" />
      <polygon points="0,-18 -4,-8 5,-6" fill="#4f9c4a" opacity="0.8" />
      <polygon points="0,-12 -5,0 6,2" fill="#3f8c3e" opacity="0.7" />
    </g>
  );
}

function Sheep({ x, y, s, flip }: { x: number; y: number; s: number; flip: boolean }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${flip ? -s : s} ${s})`}>
      <ellipse cx="0" cy="8" rx="10" ry="2.5" fill="rgba(0,0,0,0.2)" />
      <rect x="-6" y="2" width="2.2" height="6" fill="#3a3a3a" />
      <rect x="3" y="2" width="2.2" height="6" fill="#3a3a3a" />
      <circle cx="-5" cy="0" r="5" fill="#fbfbf7" />
      <circle cx="0" cy="-2" r="5.5" fill="#ffffff" />
      <circle cx="5" cy="0" r="5" fill="#f4f4ee" />
      <circle cx="0" cy="2" r="5" fill="#f0f0ea" />
      <ellipse cx="9" cy="-2" rx="3.4" ry="2.8" fill="#2f2f2f" />
    </g>
  );
}

function WheatStalk({ x, y, s, lean }: { x: number; y: number; s: number; lean: number }) {
  return (
    <g transform={`translate(${x} ${y}) rotate(${lean}) scale(${s})`}>
      <line x1="0" y1="10" x2="0" y2="-10" stroke="#a27a1d" strokeWidth="1.4" />
      {[-9, -5, -1, 3].map((dy) => (
        <g key={dy}>
          <ellipse cx="-2.2" cy={dy} rx="2" ry="3.2" fill="#f7da6a" transform={`rotate(-25 -2.2 ${dy})`} />
          <ellipse cx="2.2" cy={dy} rx="2" ry="3.2" fill="#eec549" transform={`rotate(25 2.2 ${dy})`} />
        </g>
      ))}
      <ellipse cx="0" cy="-12" rx="1.8" ry="3" fill="#f7da6a" />
    </g>
  );
}

function Mountain({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <polygon points="-22,14 0,-20 22,14" fill="#7b818d" />
      <polygon points="0,-20 22,14 4,14" fill="#5f6570" />
      <polygon points="0,-20 -7,-9 -2,-11 2,-7 7,-10" fill="#f5f7fa" />
      <polygon points="-22,14 -10,-2 -4,14" fill="#8d939f" opacity="0.6" />
    </g>
  );
}

function BrickPile({ x, y, s }: { x: number; y: number; s: number }) {
  const rows = [
    [-9, 4],
    [0, 4],
    [9, 4],
    [-4.5, -2],
    [4.5, -2],
    [0, -8],
  ];
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <ellipse cx="0" cy="9" rx="15" ry="3" fill="rgba(0,0,0,0.22)" />
      {rows.map(([bx, by], i) => (
        <g key={i}>
          <rect x={bx - 4.3} y={by - 3} width="8.6" height="6" rx="0.8" fill="#b8431c" stroke="#7d2a0e" strokeWidth="0.8" />
          <rect x={bx - 3.5} y={by - 2.4} width="7" height="1.6" fill="#e07a4a" opacity="0.55" />
        </g>
      ))}
    </g>
  );
}

function Dune({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path d="M-20,6 Q-6,-8 8,2 Q14,6 20,6 Z" fill="#e2c787" />
      <path d="M-20,6 Q-6,-8 8,2" stroke="#c9a764" strokeWidth="1.2" fill="none" />
    </g>
  );
}

function Cactus({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <ellipse cx="0" cy="11" rx="7" ry="2" fill="rgba(0,0,0,0.2)" />
      <rect x="-2.5" y="-12" width="5" height="23" rx="2.5" fill="#4f8f3c" />
      <path d="M-2.5,0 h-4 a2,2 0 0 1 -2,-2 v-6" stroke="#4f8f3c" strokeWidth="4" fill="none" strokeLinecap="round" />
      <path d="M2.5,-3 h3 a2,2 0 0 0 2,-2 v-5" stroke="#4f8f3c" strokeWidth="4" fill="none" strokeLinecap="round" />
    </g>
  );
}

/** Decorative illustration for one tile, centered at the origin, for a hex of radius r. */
export function TerrainArt({ terrain, seed, r }: { terrain: Terrain; seed: number; r: number }) {
  const k = r / 100;
  switch (terrain) {
    case "forest":
      return (
        <g>
          {scatter(seed, 9, r, 0.34).map(([x, y], i) => (
            <Pine key={i} x={x} y={y} s={k * (1.15 + ((i * 37) % 10) / 30)} />
          ))}
        </g>
      );
    case "pasture": {
      const rnd = mulberry32(seed + 5);
      const tufts = scatter(seed + 1, 14, r, 0.3);
      return (
        <g>
          {tufts.map(([x, y], i) => (
            <path
              key={`t${i}`}
              d={`M${x - 3 * k},${y} l${2 * k},${-6 * k} l${1 * k},${6 * k} l${2 * k},${-5 * k} l${1 * k},${5 * k}`}
              stroke="#4f8f2c"
              strokeWidth={1.4 * k}
              fill="none"
            />
          ))}
          {scatter(seed, 4, r, 0.4).map(([x, y], i) => (
            <Sheep key={i} x={x} y={y} s={k * 1.25} flip={rnd() > 0.5} />
          ))}
        </g>
      );
    }
    case "fields": {
      const rnd = mulberry32(seed + 3);
      return (
        <g>
          {[-0.55, -0.3, 0.3, 0.55].map((fy, row) => (
            <path
              key={row}
              d={`M${-r * 0.75},${fy * r} Q0,${fy * r - 10 * k} ${r * 0.75},${fy * r}`}
              stroke="#c99a25"
              strokeWidth={2 * k}
              fill="none"
              opacity="0.6"
            />
          ))}
          {scatter(seed, 16, r, 0.36).map(([x, y], i) => (
            <WheatStalk key={i} x={x} y={y} s={k * 1.3} lean={(rnd() - 0.5) * 24} />
          ))}
        </g>
      );
    }
    case "mountains":
      return (
        <g>
          {scatter(seed, 5, r, 0.42).map(([x, y], i) => (
            <Mountain key={i} x={x} y={y} s={k * (1.15 + (i % 3) * 0.15)} />
          ))}
        </g>
      );
    case "hills":
      return (
        <g>
          {scatter(seed + 2, 6, r, 0.4).map(([x, y], i) => (
            <path
              key={`m${i}`}
              d={`M${x - 18 * k},${y + 8 * k} Q${x},${y - 14 * k} ${x + 18 * k},${y + 8 * k} Z`}
              fill="#b5582c"
              opacity="0.55"
            />
          ))}
          {scatter(seed, 4, r, 0.42).map(([x, y], i) => (
            <BrickPile key={i} x={x} y={y} s={k * 1.25} />
          ))}
        </g>
      );
    case "desert":
      return (
        <g>
          {scatter(seed, 4, r, 0.38).map(([x, y], i) => (
            <Dune key={i} x={x} y={y} s={k * 1.4} />
          ))}
          {scatter(seed + 9, 2, r, 0.45).map(([x, y], i) => (
            <Cactus key={`c${i}`} x={x} y={y} s={k * 1.2} />
          ))}
        </g>
      );
  }
}

// ---------------------------------------------------------------- icons (24x24 viewBox)

const ICONS: Record<ResourceName, ReactElement> = {
  wood: (
    <g>
      <rect x="3" y="11" width="16" height="6" rx="3" fill="#8a5a2b" stroke="#4e2f12" strokeWidth="1" />
      <rect x="5" y="5" width="16" height="6" rx="3" fill="#a06a35" stroke="#4e2f12" strokeWidth="1" />
      <circle cx="19" cy="14" r="2.6" fill="#e3c08a" stroke="#4e2f12" strokeWidth="0.8" />
      <circle cx="21" cy="8" r="2.6" fill="#e9c995" stroke="#4e2f12" strokeWidth="0.8" />
      <path d="M8 19 q2 2 5 0" stroke="#2f7a32" strokeWidth="1.6" fill="none" />
    </g>
  ),
  brick: (
    <g stroke="#6b200a" strokeWidth="0.9">
      <rect x="2" y="13" width="9.5" height="6" rx="1" fill="#c4562b" />
      <rect x="12.5" y="13" width="9.5" height="6" rx="1" fill="#b54a22" />
      <rect x="7" y="6.5" width="9.5" height="6" rx="1" fill="#d0653a" />
    </g>
  ),
  sheep: (
    <g>
      <circle cx="9" cy="12" r="4.5" fill="#fff" stroke="#9aa" strokeWidth="0.6" />
      <circle cx="13" cy="10" r="4.8" fill="#fff" stroke="#9aa" strokeWidth="0.6" />
      <circle cx="15" cy="14" r="4.2" fill="#fafafa" stroke="#9aa" strokeWidth="0.6" />
      <ellipse cx="19.5" cy="10.5" rx="2.8" ry="2.3" fill="#333" />
      <rect x="9" y="16" width="1.6" height="4" fill="#333" />
      <rect x="14" y="17" width="1.6" height="4" fill="#333" />
    </g>
  ),
  wheat: (
    <g>
      <path d="M12 22 V6" stroke="#a27a1d" strokeWidth="1.4" />
      <path d="M12 22 L7 9 M12 22 L17 9" stroke="#a27a1d" strokeWidth="1.2" />
      {[6, 9, 12].map((y) => (
        <g key={y}>
          <ellipse cx="10" cy={y} rx="1.8" ry="2.8" fill="#f1c232" transform={`rotate(-25 10 ${y})`} />
          <ellipse cx="14" cy={y} rx="1.8" ry="2.8" fill="#e5b322" transform={`rotate(25 14 ${y})`} />
        </g>
      ))}
      <ellipse cx="12" cy="3.5" rx="1.6" ry="2.6" fill="#f1c232" />
    </g>
  ),
  ore: (
    <g stroke="#3c414a" strokeWidth="0.9">
      <polygon points="3,18 7,9 13,8 15,18" fill="#8e96a3" />
      <polygon points="11,19 14,11 20,10 22,19" fill="#6f7784" />
      <polygon points="7,9 10,6 13,8" fill="#c5ccd6" />
      <polygon points="14,11 17,8 20,10" fill="#aeb6c2" />
    </g>
  ),
};

export function ResourceIcon({ resource, size = 24 }: { resource: ResourceName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-label={resource} role="img">
      {ICONS[resource]}
    </svg>
  );
}

const DEV_ICONS: Record<DevCardName, ReactElement> = {
  knight: (
    <g>
      <path d="M12 2 L20 5 V11 C20 16 16 20 12 22 C8 20 4 16 4 11 V5 Z" fill="#7a5bb5" stroke="#3a2466" strokeWidth="1.2" />
      <path d="M12 6 V18 M7 11 H17" stroke="#f4e6c4" strokeWidth="2" />
    </g>
  ),
  victory_point: (
    <g>
      <polygon
        points="12,2 14.9,8.6 22,9.3 16.6,14 18.2,21 12,17.3 5.8,21 7.4,14 2,9.3 9.1,8.6"
        fill="#f1c232"
        stroke="#8a6508"
        strokeWidth="1.1"
      />
    </g>
  ),
  road_building: (
    <g>
      <rect x="2" y="13" width="9" height="4" rx="1.5" fill="#8a5a2b" transform="rotate(-30 6.5 15)" />
      <rect x="11" y="9" width="9" height="4" rx="1.5" fill="#a06a35" transform="rotate(-30 15.5 11)" />
      <path d="M4 21 H20" stroke="#5b8a3a" strokeWidth="1.5" />
    </g>
  ),
  year_of_plenty: (
    <g>
      <path d="M3 8 C8 22 18 22 21 8 Z" fill="#c48a3a" stroke="#6b4512" strokeWidth="1" />
      <circle cx="9" cy="8" r="3" fill="#d33a2c" />
      <circle cx="14" cy="7" r="3" fill="#f1c232" />
      <circle cx="17" cy="10" r="2.5" fill="#6fb043" />
    </g>
  ),
  monopoly: (
    <g>
      <path d="M3 18 L5 7 L9.5 12 L12 5 L14.5 12 L19 7 L21 18 Z" fill="#f1c232" stroke="#8a6508" strokeWidth="1.1" />
      <rect x="3" y="18" width="18" height="3" rx="1" fill="#d9a92c" stroke="#8a6508" strokeWidth="1" />
    </g>
  ),
};

export function DevIcon({ card, size = 24 }: { card: DevCardName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-label={card} role="img">
      {DEV_ICONS[card]}
    </svg>
  );
}

// ---------------------------------------------------------------- pieces

export function SettlementShape({ color, size = 1 }: { color: PlayerColor; size?: number }) {
  const c = PLAYER_COLORS[color];
  return (
    <g transform={`scale(${size})`}>
      <ellipse cx="0" cy="12" rx="13" ry="4" fill="rgba(0,0,0,0.35)" />
      <path d="M-11,11 V-3 L0,-14 L11,-3 V11 Z" fill={c.fill} stroke={c.stroke} strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M0,-14 L11,-3 V11 H3 V-6 Z" fill="rgba(0,0,0,0.13)" />
      <rect x="-3" y="2" width="6" height="9" fill={c.stroke} opacity="0.55" />
    </g>
  );
}

export function CityShape({ color, size = 1 }: { color: PlayerColor; size?: number }) {
  const c = PLAYER_COLORS[color];
  return (
    <g transform={`scale(${size})`}>
      <ellipse cx="0" cy="13" rx="19" ry="4.5" fill="rgba(0,0,0,0.35)" />
      <path
        d="M-17,12 V-2 L-8,-10 L1,-2 V-6 H17 V12 Z"
        fill={c.fill}
        stroke={c.stroke}
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
      <path d="M1,-6 H17 V12 H1 Z" fill="rgba(0,0,0,0.14)" />
      <rect x="-11" y="3" width="5" height="9" fill={c.stroke} opacity="0.55" />
      <rect x="5" y="0" width="3.5" height="4" fill={c.stroke} opacity="0.5" />
      <rect x="11" y="0" width="3.5" height="4" fill={c.stroke} opacity="0.5" />
    </g>
  );
}

export function RobberShape() {
  return (
    <g>
      <ellipse cx="0" cy="22" rx="15" ry="5" fill="rgba(0,0,0,0.45)" />
      <path d="M-12,22 C-12,8 -7,2 -6,-2 C-11,-6 -10,-18 0,-19 C10,-18 11,-6 6,-2 C7,2 12,8 12,22 Z" fill="#3b3b40" />
      <path d="M-4,-14 C-1,-17 4,-16 5,-12" stroke="#7d7d86" strokeWidth="2.4" fill="none" strokeLinecap="round" />
      <path d="M-8,18 C-8,10 -5,5 -3,2" stroke="#6a6a73" strokeWidth="2" fill="none" strokeLinecap="round" />
    </g>
  );
}

export function NumberToken({ number, dim }: { number: number; dim?: boolean }) {
  const red = number === 6 || number === 8;
  const pips = 6 - Math.abs(7 - number);
  return (
    <g opacity={dim ? 0.55 : 1}>
      <circle r="27" cy="2.5" fill="rgba(0,0,0,0.3)" />
      <circle r="27" fill="url(#token-grad)" stroke="#a88b56" strokeWidth="2" />
      <circle r="23" fill="none" stroke="#d9c69a" strokeWidth="1" />
      <text
        y="5"
        textAnchor="middle"
        fontFamily="'Cinzel', Georgia, serif"
        fontWeight={700}
        fontSize={red ? 25 : 22}
        fill={red ? "#c0261a" : "#2b2117"}
      >
        {number}
      </text>
      <g transform="translate(0 15)">
        {Array.from({ length: pips }, (_, i) => (
          <circle key={i} cx={(i - (pips - 1) / 2) * 5.2} r="1.9" fill={red ? "#c0261a" : "#2b2117"} />
        ))}
      </g>
    </g>
  );
}
