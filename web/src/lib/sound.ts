// Procedural sound effects (Web Audio, no asset files) and the mapping from game events to
// sounds. Browsers keep an AudioContext suspended until a user gesture, so the context is
// created lazily and resumed on the first pointer/key press.

import { useSyncExternalStore } from "react";
import type { GameEvent } from "./types";

export type Sfx =
  | "dice"
  | "seven"
  | "produce"
  | "road"
  | "settlement"
  | "city"
  | "card"
  | "dev_play"
  | "robber"
  | "steal"
  | "stolen_from_you"
  | "offer"
  | "deal"
  | "no_deal"
  | "award"
  | "your_turn"
  | "win"
  | "lose"
  | "chat";

const MUTE_KEY = "catan.muted";

let muted = readMuted();
const listeners = new Set<() => void>();
let ctx: AudioContext | null = null;
let master: GainNode | null = null;
let noiseBuf: AudioBuffer | null = null;

function readMuted(): boolean {
  try {
    return localStorage.getItem(MUTE_KEY) === "1";
  } catch {
    return false;
  }
}

export function setMuted(m: boolean) {
  muted = m;
  try {
    localStorage.setItem(MUTE_KEY, m ? "1" : "0");
  } catch {
    /* storage unavailable - preference lasts for this page only */
  }
  listeners.forEach((l) => l());
}

export function useMuted(): boolean {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => muted,
  );
}

function audio(): AudioContext | null {
  if (ctx) return ctx;
  const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  ctx = new Ctor();
  master = ctx.createGain();
  master.gain.value = 0.5;
  master.connect(ctx.destination);
  noiseBuf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
  const data = noiseBuf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  return ctx;
}

if (typeof window !== "undefined") {
  const unlock = () => {
    const c = audio();
    if (c && c.state === "suspended") void c.resume();
  };
  window.addEventListener("pointerdown", unlock, { capture: true });
  window.addEventListener("keydown", unlock, { capture: true });
}

// ------------------------------------------------------------------ primitives

interface ToneOpts {
  freq: number;
  to?: number; // glide target frequency
  type?: OscillatorType;
  at?: number; // start offset (s)
  dur?: number; // seconds
  gain?: number;
  attack?: number;
}

function tone(c: AudioContext, { freq, to, type = "sine", at = 0, dur = 0.2, gain = 0.3, attack = 0.005 }: ToneOpts) {
  const t = c.currentTime + at;
  const osc = c.createOscillator();
  const g = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t);
  if (to) osc.frequency.exponentialRampToValueAtTime(to, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g).connect(master!);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}

interface NoiseOpts {
  at?: number;
  dur?: number;
  gain?: number;
  filter?: BiquadFilterType;
  freq?: number;
  to?: number;
  q?: number;
}

function noise(c: AudioContext, { at = 0, dur = 0.05, gain = 0.3, filter = "bandpass", freq = 2000, to, q = 1 }: NoiseOpts) {
  const t = c.currentTime + at;
  const src = c.createBufferSource();
  src.buffer = noiseBuf;
  const f = c.createBiquadFilter();
  f.type = filter;
  f.Q.value = q;
  f.frequency.setValueAtTime(freq, t);
  if (to) f.frequency.exponentialRampToValueAtTime(to, t + dur);
  const g = c.createGain();
  g.gain.setValueAtTime(gain, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(f).connect(g).connect(master!);
  src.start(t, Math.random() * 0.5);
  src.stop(t + dur + 0.02);
}

const knock = (c: AudioContext, at: number, pitch: number, gain = 0.5) => {
  tone(c, { freq: pitch, to: pitch * 0.6, type: "triangle", at, dur: 0.12, gain });
  noise(c, { at, dur: 0.04, gain: gain * 0.5, freq: pitch * 4, q: 3 });
};

const chord = (c: AudioContext, notes: number[], at: number, step: number, dur: number, gain: number, type: OscillatorType = "triangle") =>
  notes.forEach((f, i) => tone(c, { freq: f, type, at: at + i * step, dur, gain }));

const SOUNDS: Record<Sfx, (c: AudioContext) => void> = {
  dice: (c) => {
    for (let i = 0; i < 7; i++) {
      noise(c, { at: i * 0.045 + Math.random() * 0.02, dur: 0.03, gain: 0.35 - i * 0.03, freq: 2500 + Math.random() * 1500, q: 4 });
    }
    knock(c, 0.34, 420, 0.25);
  },
  seven: (c) => {
    tone(c, { freq: 110, to: 70, type: "sawtooth", at: 0.42, dur: 0.6, gain: 0.12 });
    tone(c, { freq: 116, to: 74, type: "sawtooth", at: 0.42, dur: 0.6, gain: 0.1 });
  },
  produce: (c) => {
    tone(c, { freq: 880, type: "sine", at: 0.45, dur: 0.25, gain: 0.18 });
    tone(c, { freq: 1320, type: "sine", at: 0.52, dur: 0.3, gain: 0.12 });
  },
  road: (c) => {
    knock(c, 0, 260, 0.45);
  },
  settlement: (c) => {
    knock(c, 0, 220, 0.5);
    knock(c, 0.11, 250, 0.45);
  },
  city: (c) => {
    knock(c, 0, 150, 0.6);
    knock(c, 0.12, 170, 0.5);
    chord(c, [523, 659, 784], 0.22, 0.06, 0.4, 0.12);
  },
  card: (c) => {
    noise(c, { dur: 0.12, gain: 0.35, filter: "highpass", freq: 1200, to: 4000 });
    knock(c, 0.1, 600, 0.15);
  },
  dev_play: (c) => {
    chord(c, [659, 880, 1109, 1319], 0, 0.05, 0.35, 0.1, "sine");
  },
  robber: (c) => {
    tone(c, { freq: 90, to: 55, type: "sawtooth", dur: 0.45, gain: 0.15 });
    noise(c, { dur: 0.4, gain: 0.15, filter: "lowpass", freq: 400 });
  },
  steal: (c) => {
    noise(c, { dur: 0.22, gain: 0.3, freq: 800, to: 3000, q: 2 });
  },
  stolen_from_you: (c) => {
    noise(c, { dur: 0.22, gain: 0.3, freq: 3000, to: 600, q: 2 });
    tone(c, { freq: 392, to: 294, type: "triangle", at: 0.12, dur: 0.3, gain: 0.15 });
  },
  offer: (c) => {
    tone(c, { freq: 784, type: "sine", dur: 0.25, gain: 0.18 });
    tone(c, { freq: 988, type: "sine", at: 0.1, dur: 0.3, gain: 0.15 });
  },
  deal: (c) => {
    for (let i = 0; i < 3; i++) tone(c, { freq: 1568 + i * 200, type: "square", at: i * 0.05, dur: 0.12, gain: 0.05 });
    tone(c, { freq: 2093, type: "sine", at: 0.15, dur: 0.5, gain: 0.15 });
  },
  no_deal: (c) => {
    tone(c, { freq: 330, to: 247, type: "triangle", dur: 0.25, gain: 0.15 });
  },
  award: (c) => {
    chord(c, [523, 659, 784, 1047], 0, 0.08, 0.35, 0.12);
  },
  your_turn: (c) => {
    tone(c, { freq: 587, type: "sine", dur: 0.2, gain: 0.2 });
    tone(c, { freq: 880, type: "sine", at: 0.12, dur: 0.35, gain: 0.2 });
  },
  win: (c) => {
    chord(c, [523, 659, 784], 0, 0.12, 0.3, 0.15);
    chord(c, [1047, 1319, 1568], 0.4, 0, 0.9, 0.1);
  },
  lose: (c) => {
    chord(c, [392, 330, 262], 0, 0.18, 0.4, 0.13);
  },
  chat: (c) => {
    tone(c, { freq: 1200, to: 1500, type: "sine", dur: 0.08, gain: 0.12 });
  },
};

/** Play several effects as one batch: deduplicated, staggered and capped so a burst of bot
 *  actions doesn't turn into noise. */
export function play(effects: Sfx[]) {
  if (muted || !effects.length) return;
  const c = audio();
  if (!c || c.state !== "running") return;
  const unique = [...new Set(effects)].slice(0, 4);
  unique.forEach((s, i) => {
    if (i === 0) SOUNDS[s](c);
    else window.setTimeout(() => SOUNDS[s](c), i * 140);
  });
}

/** The effect for one log event, from the point of view of the seats this client controls. */
export function sfxFor(event: GameEvent, mine: (seat: number | null) => boolean): Sfx | null {
  switch (event.type) {
    case "dice_rolled":
      return "dice";
    case "produced":
      return mine(event.player) ? "produce" : null;
    case "road_built":
      return "road";
    case "settlement_built":
      return "settlement";
    case "city_built":
      return "city";
    case "dev_card_bought":
      return "card";
    case "dev_card_played":
      return "dev_play";
    case "robber_moved":
      return "robber";
    case "stolen":
      return mine(event.victim) ? "stolen_from_you" : "steal";
    case "maritime_traded":
    case "trade_executed":
      return "deal";
    case "trade_offered":
      return "offer";
    case "trade_cancelled":
      return "no_deal";
    case "longest_road_changed":
    case "largest_army_changed":
      return event.player !== null ? "award" : null;
    case "game_won":
      return mine(event.player) ? "win" : "lose";
    default:
      return null;
  }
}
