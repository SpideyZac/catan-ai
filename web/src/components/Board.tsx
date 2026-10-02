import { memo, useMemo } from "react";
import type { BoardInfo, GameView, PlayerColor, ResourceName } from "../lib/types";
import {
  CityShape,
  NumberToken,
  PLAYER_COLORS,
  RESOURCE_COLORS,
  ResourceIcon,
  RobberShape,
  SettlementShape,
  TERRAIN_GRADIENT,
  TerrainArt,
} from "./art";

/** Hex circumradius in SVG units; engine coordinates use radius 1. */
const S = 100;
const FRAME_R = 5.75 * S;
const TERRAINS = ["forest", "hills", "pasture", "fields", "mountains", "desert"] as const;

export interface BoardTargets {
  vertices: Set<number>;
  edges: Set<number>;
  hexes: Set<number>;
  /** What clicking a vertex does (affects the ghost piece preview). */
  vertexKind: "settlement" | "city" | null;
}

export const NO_TARGETS: BoardTargets = { vertices: new Set(), edges: new Set(), hexes: new Set(), vertexKind: null };

interface Props {
  board: BoardInfo;
  view: GameView;
  colors: PlayerColor[];
  targets: BoardTargets;
  activeColor: PlayerColor | null;
  rolled: { total: number; key: number } | null;
  onVertex: (v: number) => void;
  onEdge: (e: number) => void;
  onHex: (h: number) => void;
}

function hexPoints(cx: number, cy: number, r: number) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = ((60 * i - 90) * Math.PI) / 180;
    return `${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`;
  }).join(" ");
}

function frameHexPoints(r: number) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = ((60 * i) * Math.PI) / 180;
    return `${(r * Math.cos(a)).toFixed(2)},${(r * Math.sin(a)).toFixed(2)}`;
  }).join(" ");
}

const StaticBoard = memo(function StaticBoard({ board }: { board: BoardInfo }) {
  const V = board.vertices.map(([x, y]) => [x * S, y * S] as const);
  return (
    <>
      {/* Sea inside the wooden frame */}
      <polygon points={frameHexPoints(FRAME_R + 14)} fill="#6b4322" stroke="#3e2410" strokeWidth="6" />
      <polygon points={frameHexPoints(FRAME_R)} fill="url(#sea-grad)" />
      <polygon points={frameHexPoints(FRAME_R)} fill="url(#waves)" opacity="0.5" />
      <polygon points={frameHexPoints(FRAME_R)} fill="none" stroke="#a87a45" strokeWidth="5" />

      {/* Beach under the island */}
      <g filter="url(#soft)">
        {board.hexes.map((h) => (
          <polygon key={h.id} points={hexPoints(h.x * S, h.y * S, S * 1.13)} fill="#e9d39b" />
        ))}
      </g>

      {/* Harbors */}
      {board.ports.map((p) => {
        const [a, b] = p.vertices.map((v) => V[v]);
        const mx = (a[0] + b[0]) / 2;
        const my = (a[1] + b[1]) / 2;
        const len = Math.hypot(mx, my);
        const px = mx + (mx / len) * 62;
        const py = my + (my / len) * 62;
        const generic = p.kind === "generic";
        return (
          <g key={p.edge}>
            {[a, b].map(([vx, vy], i) => (
              <line key={i} x1={vx} y1={vy} x2={px} y2={py} stroke="#8a5a2b" strokeWidth="7" strokeLinecap="round" />
            ))}
            {[a, b].map(([vx, vy], i) => (
              <line
                key={`h${i}`}
                x1={vx}
                y1={vy}
                x2={px}
                y2={py}
                stroke="#c08a52"
                strokeWidth="2"
                strokeDasharray="4 6"
              />
            ))}
            <circle cx={px} cy={py + 2} r="27" fill="rgba(0,0,0,0.3)" />
            <circle
              cx={px}
              cy={py}
              r="27"
              fill="#f6ead0"
              stroke={generic ? "#6b4322" : RESOURCE_COLORS[p.kind as ResourceName]}
              strokeWidth="4"
            />
            {generic ? (
              <text x={px} y={py + 7} textAnchor="middle" className="port-text">
                3:1
              </text>
            ) : (
              <>
                <foreignObject x={px - 13} y={py - 21} width="26" height="26">
                  <ResourceIcon resource={p.kind as ResourceName} size={26} />
                </foreignObject>
                <text x={px} y={py + 18} textAnchor="middle" className="port-text small">
                  2:1
                </text>
              </>
            )}
          </g>
        );
      })}

      {/* Tiles */}
      {board.hexes.map((h) => {
        const cx = h.x * S;
        const cy = h.y * S;
        return (
          <g key={h.id} transform={`translate(${cx} ${cy})`}>
            <polygon points={hexPoints(0, 0, S * 0.985)} fill={`url(#tg-${h.terrain})`} />
            <g clipPath="url(#hex-clip)">
              <TerrainArt terrain={h.terrain} seed={h.id * 7919 + 13} r={S} />
            </g>
            <polygon
              points={hexPoints(0, 0, S * 0.985)}
              fill="none"
              stroke="rgba(60,40,20,0.55)"
              strokeWidth="3"
            />
            <polygon points={hexPoints(0, 0, S * 0.93)} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="2" />
          </g>
        );
      })}
    </>
  );
});

export function Board({ board, view, colors, targets, activeColor, rolled, onVertex, onEdge, onHex }: Props) {
  const V = useMemo(() => board.vertices.map(([x, y]) => [x * S, y * S] as const), [board]);
  const edgeEnds = (e: number) => {
    const [a, b] = board.edges[e];
    const [ax, ay] = V[a];
    const [bx, by] = V[b];
    const sx = (bx - ax) * 0.13;
    const sy = (by - ay) * 0.13;
    return { x1: ax + sx, y1: ay + sy, x2: bx - sx, y2: by - sy };
  };
  const active = activeColor ? PLAYER_COLORS[activeColor] : PLAYER_COLORS.white;

  return (
    <svg className="board" viewBox="-640 -545 1280 1090" preserveAspectRatio="xMidYMid meet">
      <defs>
        <radialGradient id="sea-grad" cx="50%" cy="45%" r="65%">
          <stop offset="0%" stopColor="#3d8fc9" />
          <stop offset="70%" stopColor="#21679f" />
          <stop offset="100%" stopColor="#174c7a" />
        </radialGradient>
        <pattern id="waves" width="60" height="30" patternUnits="userSpaceOnUse">
          <path d="M0 15 q15 -10 30 0 t30 0" fill="none" stroke="#9fd0f0" strokeWidth="1.6" opacity="0.5" />
        </pattern>
        <radialGradient id="token-grad" cx="40%" cy="35%" r="70%">
          <stop offset="0%" stopColor="#fffaf0" />
          <stop offset="100%" stopColor="#e8d5a6" />
        </radialGradient>
        {TERRAINS.map((t) => (
          <radialGradient key={t} id={`tg-${t}`} cx="45%" cy="40%" r="75%">
            <stop offset="0%" stopColor={TERRAIN_GRADIENT[t][0]} />
            <stop offset="100%" stopColor={TERRAIN_GRADIENT[t][1]} />
          </radialGradient>
        ))}
        <clipPath id="hex-clip">
          <polygon points={hexPoints(0, 0, S * 0.97)} />
        </clipPath>
        <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="5" />
        </filter>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <StaticBoard board={board} />

      {/* Number tokens, with a pulse on the hexes that just produced */}
      {board.hexes.map((h) =>
        h.number ? (
          <g key={h.id} transform={`translate(${h.x * S} ${h.y * S})`}>
            {rolled && rolled.total === h.number && view.robber !== h.id && (
              <circle key={rolled.key} r="40" className="produce-ring" />
            )}
            <NumberToken number={h.number} dim={view.robber === h.id} />
          </g>
        ) : null,
      )}

      {/* Roads */}
      {view.roads.map((r) => {
        const c = PLAYER_COLORS[colors[r.player]];
        const p = edgeEnds(r.edge);
        return (
          <g key={`road-${r.edge}`} className="pop">
            <line {...p} stroke="rgba(0,0,0,0.35)" strokeWidth="17" strokeLinecap="round" transform="translate(0 3)" />
            <line {...p} stroke={c.stroke} strokeWidth="16" strokeLinecap="round" />
            <line {...p} stroke={c.fill} strokeWidth="11" strokeLinecap="round" />
          </g>
        );
      })}

      {/* Buildings */}
      {view.buildings.map((b) => {
        const [x, y] = V[b.vertex];
        return (
          <g key={`b-${b.vertex}-${b.city}`} transform={`translate(${x} ${y})`} className="pop">
            {b.city ? <CityShape color={colors[b.player]} /> : <SettlementShape color={colors[b.player]} />}
          </g>
        );
      })}

      {/* Robber */}
      {(() => {
        const h = board.hexes[view.robber];
        return (
          <g className="robber" style={{ transform: `translate(${h.x * S - 30}px, ${h.y * S - 12}px)` }}>
            <RobberShape />
          </g>
        );
      })()}

      {/* Interactive targets */}
      {[...targets.hexes].map((hid) => {
        const h = board.hexes[hid];
        return (
          <polygon
            key={`th-${hid}`}
            points={hexPoints(h.x * S, h.y * S, S * 0.9)}
            className="target-hex"
            onClick={() => onHex(hid)}
          />
        );
      })}
      {[...targets.edges].map((e) => {
        const p = edgeEnds(e);
        return (
          <g key={`te-${e}`} className="target-edge" onClick={() => onEdge(e)}>
            <line {...p} stroke="transparent" strokeWidth="30" strokeLinecap="round" />
            <line {...p} className="target-edge-line" stroke={active.light} strokeWidth="9" strokeLinecap="round" />
          </g>
        );
      })}
      {[...targets.vertices].map((v) => {
        const [x, y] = V[v];
        return (
          <g key={`tv-${v}`} transform={`translate(${x} ${y})`} className="target-vertex" onClick={() => onVertex(v)}>
            <circle r="24" fill="transparent" />
            <circle r="13" className="target-dot" fill={active.light} stroke={active.stroke} strokeWidth="3" />
            {activeColor && (
              <g className="ghost">
                {targets.vertexKind === "city" ? (
                  <CityShape color={activeColor} size={0.9} />
                ) : (
                  <SettlementShape color={activeColor} size={0.9} />
                )}
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
