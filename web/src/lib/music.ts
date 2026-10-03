// Procedural background music (Web Audio, no asset files): a pastoral harbour air for the
// home screen and lobby, and a lively 6/8 settlers' jig during games. Both are folk
// arrangements in the white-key modes (D Dorian / G Mixolydian) for plucked lute, wooden
// flute/whistle, soft pads and a frame drum, scheduled a fraction of a second ahead with a
// look-ahead timer. Music has its own on/off switch, independent of the effects mute.

import { useEffect, useSyncExternalStore } from "react";
import { existingAudio } from "./sound";

export type MusicTrack = "lobby" | "game";

// ------------------------------------------------------------------ preference

const MUSIC_KEY = "catan.music";

function readOn(): boolean {
  try {
    return localStorage.getItem(MUSIC_KEY) !== "0";
  } catch {
    return true;
  }
}

let musicOn = readOn();
const listeners = new Set<() => void>();

export function setMusicOn(on: boolean) {
  musicOn = on;
  try {
    localStorage.setItem(MUSIC_KEY, on ? "1" : "0");
  } catch {
    /* storage unavailable - preference lasts for this page only */
  }
  listeners.forEach((l) => l());
  sync();
}

export function useMusicOn(): boolean {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => musicOn,
  );
}

// ------------------------------------------------------------------ score

/** [step within the bar (eighth notes), MIDI note, length in steps] */
type Note = [number, number, number];

interface Chord {
  bass: number;
  tones: number[]; // root, third, fifth, octave - one octave above the bass
}

interface Section {
  chords: Chord[]; // one per bar
  melody?: Note[][]; // one list per bar
  drums?: boolean;
}

interface Track {
  stepSec: number; // one eighth note
  sections: Section[];
  lead: "flute" | "whistle";
  pad: boolean;
  /** "arp": flowing broken chords on every eighth; "jig": bass on 1 and 4, chord on 3 and 6 */
  comp: "arp" | "jig";
  reverb: number;
}

const ch = (bass: number, minor = false): Chord => ({
  bass,
  tones: [bass + 12, bass + 12 + (minor ? 3 : 4), bass + 19, bass + 24],
});

// Lobby - "Harbour at Dawn": D Dorian, slow 6/8.
const Dm = ch(50, true), C = ch(48), G = ch(43), Am = ch(45, true), F = ch(41);
const LOBBY_PROG = [Dm, C, G, Dm, F, C, Am, Dm];
const LOBBY_TUNE: Note[][] = [
  [[0, 69, 3], [3, 74, 2], [5, 76, 1]],
  [[0, 76, 3], [3, 79, 2], [5, 77, 1]],
  [[0, 74, 2], [2, 71, 1], [3, 74, 3]],
  [[0, 69, 6]],
  [[0, 72, 2], [2, 77, 1], [3, 81, 2], [5, 79, 1]],
  [[0, 76, 3], [3, 72, 2], [5, 74, 1]],
  [[0, 76, 2], [2, 72, 1], [3, 69, 3]],
  [[0, 74, 6]],
];
const LOBBY_TUNE_B: Note[][] = [
  [[0, 81, 3], [3, 77, 2], [5, 76, 1]],
  [[0, 79, 3], [3, 76, 2], [5, 72, 1]],
  [[0, 71, 2], [2, 74, 1], [3, 79, 3]],
  [[0, 77, 3], [3, 76, 3]],
  [[0, 77, 2], [2, 81, 1], [3, 84, 3]],
  [[0, 79, 2], [2, 76, 1], [3, 72, 3]],
  [[0, 72, 2], [2, 76, 1], [3, 81, 2], [5, 79, 1]],
  [[0, 74, 6]],
];

const LOBBY: Track = {
  stepSec: 0.27,
  lead: "flute",
  pad: true,
  comp: "arp",
  reverb: 0.3,
  sections: [
    { chords: LOBBY_PROG },
    { chords: LOBBY_PROG, melody: LOBBY_TUNE },
    { chords: LOBBY_PROG, melody: LOBBY_TUNE_B },
    { chords: LOBBY_PROG, melody: LOBBY_TUNE },
  ],
};

// Game - "Settlers' Jig": G Mixolydian, brisk 6/8 with a frame drum.
const Dmi = ch(50, true);
const JIG_A_PROG = [G, G, F, F, G, C, Dmi, G];
const JIG_B_PROG = [C, G, F, G, C, G, Dmi, G];
const JIG_A: Note[][] = [
  [[0, 79, 2], [2, 81, 1], [3, 83, 2], [5, 81, 1]],
  [[0, 79, 2], [2, 74, 1], [3, 74, 2], [5, 76, 1]],
  [[0, 77, 2], [2, 76, 1], [3, 77, 1], [4, 79, 1], [5, 81, 1]],
  [[0, 84, 3], [3, 81, 2], [5, 79, 1]],
  [[0, 79, 2], [2, 81, 1], [3, 83, 2], [5, 86, 1]],
  [[0, 88, 2], [2, 86, 1], [3, 84, 2], [5, 83, 1]],
  [[0, 81, 2], [2, 77, 1], [3, 74, 2], [5, 77, 1]],
  [[0, 79, 5]],
];
const JIG_B: Note[][] = [
  [[0, 84, 1], [1, 83, 1], [2, 84, 1], [3, 88, 2], [5, 84, 1]],
  [[0, 86, 2], [2, 83, 1], [3, 79, 2], [5, 83, 1]],
  [[0, 84, 2], [2, 81, 1], [3, 77, 2], [5, 81, 1]],
  [[0, 83, 3], [3, 79, 3]],
  [[0, 88, 2], [2, 86, 1], [3, 84, 2], [5, 88, 1]],
  [[0, 86, 2], [2, 83, 1], [3, 86, 2], [5, 83, 1]],
  [[0, 81, 1], [1, 83, 1], [2, 84, 1], [3, 86, 2], [5, 77, 1]],
  [[0, 79, 6]],
];

const GAME: Track = {
  stepSec: 0.2,
  lead: "whistle",
  pad: false,
  comp: "jig",
  reverb: 0.2,
  sections: [
    { chords: JIG_A_PROG, drums: true },
    { chords: JIG_A_PROG, melody: JIG_A, drums: true },
    { chords: JIG_B_PROG, melody: JIG_B, drums: true },
    { chords: JIG_A_PROG, melody: JIG_A, drums: true },
    // A breather so a long game doesn't wear on the ears.
    { chords: JIG_B_PROG },
  ],
};

const TRACKS: Record<MusicTrack, Track> = { lobby: LOBBY, game: GAME };

// Both modes use the white keys; a grace note is the next of them above the main note.
const WHITE = new Set([0, 2, 4, 5, 7, 9, 11]);
const graceAbove = (m: number) => (WHITE.has((m + 1) % 12) ? m + 1 : m + 2);

// ------------------------------------------------------------------ instruments

const mtof = (m: number) => 440 * 2 ** ((m - 69) / 12);
const jitter = (amount: number) => (Math.random() * 2 - 1) * amount;

interface Out {
  c: AudioContext;
  dest: AudioNode;
  noise: AudioBuffer;
}

function env(g: GainNode, t: number, peak: number, attack: number, hold: number, release: number) {
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + attack);
  if (hold > 0) g.gain.setValueAtTime(peak, t + attack + hold);
  g.gain.exponentialRampToValueAtTime(0.0001, t + attack + hold + release);
}

/** Lute/harp-like pluck: bright attack that darkens as it decays. */
function pluck({ c, dest }: Out, t: number, midi: number, vel: number, decay = 1.3) {
  const f = mtof(midi);
  const lp = c.createBiquadFilter();
  lp.type = "lowpass";
  lp.Q.value = 1.5;
  lp.frequency.setValueAtTime(Math.min(f * 9, 7000), t);
  lp.frequency.exponentialRampToValueAtTime(Math.max(f * 1.2, 200), t + decay * 0.7);
  const g = c.createGain();
  env(g, t, vel, 0.004, 0, decay);
  const a = c.createOscillator();
  a.type = "triangle";
  a.frequency.value = f;
  const b = c.createOscillator();
  b.type = "sawtooth";
  b.frequency.value = f * 1.002;
  const bg = c.createGain();
  bg.gain.value = 0.35;
  a.connect(lp);
  b.connect(bg).connect(lp);
  lp.connect(g).connect(dest);
  for (const o of [a, b]) {
    o.start(t);
    o.stop(t + decay + 0.05);
  }
}

/** Wooden flute (soft, breathy) or tin whistle (brighter, quicker vibrato). */
function wind(out: Out, t: number, midi: number, dur: number, vel: number, kind: "flute" | "whistle") {
  const { c, dest } = out;
  const f = mtof(midi);
  const whistle = kind === "whistle";
  const g = c.createGain();
  const attack = whistle ? 0.025 : 0.07;
  const release = whistle ? 0.08 : 0.16;
  env(g, t, vel, attack, Math.max(0.01, dur - attack - release * 0.5), release);
  const o = c.createOscillator();
  o.type = "sine";
  o.frequency.value = f;
  const h = c.createOscillator();
  h.type = "triangle";
  h.frequency.value = f * 2;
  const hg = c.createGain();
  hg.gain.value = whistle ? 0.22 : 0.1;
  // Delayed vibrato, as a player would add on held notes.
  const lfo = c.createOscillator();
  lfo.frequency.value = whistle ? 6 : 5;
  const depth = c.createGain();
  depth.gain.setValueAtTime(0, t);
  depth.gain.linearRampToValueAtTime(0, t + 0.18);
  depth.gain.linearRampToValueAtTime(f * (whistle ? 0.007 : 0.005), t + 0.45);
  lfo.connect(depth);
  depth.connect(o.frequency);
  o.connect(g);
  h.connect(hg).connect(g);
  g.connect(dest);
  const end = t + dur + release + 0.1;
  for (const x of [o, h, lfo]) {
    x.start(t);
    x.stop(end);
  }
  // Breath chiff at the start of the note.
  const n = c.createBufferSource();
  n.buffer = out.noise;
  const bp = c.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = f * 2;
  bp.Q.value = 2;
  const ng = c.createGain();
  env(ng, t, vel * (whistle ? 0.25 : 0.4), 0.01, 0, 0.09);
  n.connect(bp).connect(ng).connect(dest);
  n.start(t, Math.random() * 0.5);
  n.stop(t + 0.15);
}

/** Soft sustained pad (bowed drone), fades in and out across a bar. */
function pad({ c, dest }: Out, t: number, midis: number[], dur: number, vel: number) {
  const lp = c.createBiquadFilter();
  lp.type = "lowpass";
  lp.frequency.value = 900;
  const g = c.createGain();
  env(g, t, vel, dur * 0.35, dur * 0.3, dur * 0.6);
  lp.connect(g).connect(dest);
  for (const m of midis) {
    for (const detune of [-7, 7]) {
      const o = c.createOscillator();
      o.type = "sawtooth";
      o.frequency.value = mtof(m);
      o.detune.value = detune;
      o.connect(lp);
      o.start(t);
      o.stop(t + dur * 1.3);
    }
  }
}

/** Bodhrán-style frame drum: pitched thump plus a skin slap. */
function drum({ c, dest, noise }: Out, t: number, vel: number) {
  const o = c.createOscillator();
  o.type = "sine";
  o.frequency.setValueAtTime(130, t);
  o.frequency.exponentialRampToValueAtTime(55, t + 0.14);
  const g = c.createGain();
  env(g, t, vel, 0.003, 0, 0.22);
  o.connect(g).connect(dest);
  o.start(t);
  o.stop(t + 0.26);
  const n = c.createBufferSource();
  n.buffer = noise;
  const lp = c.createBiquadFilter();
  lp.type = "lowpass";
  lp.frequency.value = 1100;
  const ng = c.createGain();
  env(ng, t, vel * 0.45, 0.002, 0, 0.06);
  n.connect(lp).connect(ng).connect(dest);
  n.start(t, Math.random() * 0.5);
  n.stop(t + 0.1);
}

// ------------------------------------------------------------------ player

const fx = new WeakMap<AudioContext, { out: GainNode; reverb: ConvolverNode; noise: AudioBuffer }>();

/** Per-context music output: volume node, a synthetic hall reverb and a noise buffer. */
function musicBus(c: AudioContext) {
  let bus = fx.get(c);
  if (bus) return bus;
  const out = c.createGain();
  out.gain.value = 0.32;
  out.connect(c.destination);
  const len = Math.floor(c.sampleRate * 2.4);
  const ir = c.createBuffer(2, len, c.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = ir.getChannelData(ch);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 3;
  }
  const reverb = c.createConvolver();
  reverb.buffer = ir;
  reverb.connect(out);
  const noise = c.createBuffer(1, c.sampleRate, c.sampleRate);
  const nd = noise.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  bus = { out, reverb, noise };
  fx.set(c, bus);
  return bus;
}

const FADE_IN = 2.5;
const FADE_OUT = 1.5;

class Player {
  private gain: GainNode | null = null;
  private out: Out | null = null;
  private timer: number;
  private nextTime = 0;
  private section = 0;
  private bar = 0;
  private step = 0;

  constructor(private track: Track) {
    this.timer = window.setInterval(() => this.tick(), 100);
    this.tick();
  }

  private tick() {
    const c = existingAudio();
    if (!c || c.state !== "running") {
      this.nextTime = 0; // resume cleanly once the context runs again
      return;
    }
    if (!this.out) {
      const bus = musicBus(c);
      this.gain = c.createGain();
      this.gain.gain.setValueAtTime(0.0001, c.currentTime);
      this.gain.gain.exponentialRampToValueAtTime(1, c.currentTime + FADE_IN);
      this.gain.connect(bus.out);
      const send = c.createGain();
      send.gain.value = this.track.reverb;
      this.gain.connect(send).connect(bus.reverb);
      this.out = { c, dest: this.gain, noise: bus.noise };
    }
    if (this.nextTime < c.currentTime) this.nextTime = c.currentTime + 0.05;
    // Background tabs throttle timers to ~1 s, so schedule further ahead there.
    const horizon = c.currentTime + (document.hidden ? 1.6 : 0.35);
    while (this.nextTime < horizon) {
      this.playStep(this.nextTime);
      this.nextTime += this.track.stepSec;
      if (++this.step === 6) {
        this.step = 0;
        if (++this.bar === this.track.sections[this.section].chords.length) {
          this.bar = 0;
          this.section = (this.section + 1) % this.track.sections.length;
        }
      }
    }
  }

  private playStep(t: number) {
    const out = this.out!;
    const tr = this.track;
    const sec = tr.sections[this.section];
    const chord = sec.chords[this.bar];
    const s = this.step;
    const sp = tr.stepSec;
    const h = () => t + jitter(0.006);

    if (tr.pad && s === 0) pad(out, t, [chord.bass + 12, chord.bass + 19], sp * 6, 0.022);

    if (tr.comp === "arp") {
      const order = [0, 2, 1, 3, 1, 2];
      if (s === 0) pluck(out, h(), chord.bass, 0.11, 2.2);
      pluck(out, h(), chord.tones[order[s]], (s === 0 ? 0.075 : 0.055) * (1 + jitter(0.15)));
    } else {
      if (s === 0) pluck(out, h(), chord.bass, 0.14, 0.9);
      if (s === 3) pluck(out, h(), chord.bass + 7, 0.1, 0.8);
      if (s === 2 || s === 5) {
        // Light strum: chord tones a few ms apart.
        chord.tones.slice(1).forEach((m, i) => pluck(out, t + i * 0.012 + jitter(0.004), m, 0.035, 0.45));
      }
    }

    if (sec.drums) {
      const v = s === 0 ? 0.24 : s === 3 ? 0.16 : Math.random() < 0.55 ? 0.05 : 0;
      if (v) drum(out, h(), v * (1 + jitter(0.1)));
    }

    for (const [at, midi, len] of sec.melody?.[this.bar] ?? []) {
      if (at !== s) continue;
      const vel = (tr.lead === "flute" ? 0.11 : 0.085) * (1 + jitter(0.12));
      const dur = len * sp * 0.95;
      // Occasional cut (grace note) on longer notes, as folk players ornament a tune.
      if (len >= 2 && Math.random() < 0.25) {
        wind(out, t, graceAbove(midi), 0.05, vel * 0.7, tr.lead);
        wind(out, t + 0.05, midi, dur - 0.05, vel, tr.lead);
      } else {
        wind(out, h(), midi, dur, vel, tr.lead);
      }
    }
  }

  stop() {
    window.clearInterval(this.timer);
    const g = this.gain;
    if (!g) return;
    const c = g.context;
    g.gain.cancelScheduledValues(c.currentTime);
    g.gain.setValueAtTime(Math.max(g.gain.value, 0.0001), c.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + FADE_OUT);
    window.setTimeout(() => g.disconnect(), FADE_OUT * 1000 + 2500);
  }
}

// ------------------------------------------------------------------ track selection

let desired: MusicTrack | null = null;
let playing: { name: MusicTrack; player: Player } | null = null;
let pending: number | undefined;

function sync() {
  window.clearTimeout(pending);
  const target = musicOn ? desired : null;
  if (playing?.name === target) return;
  playing?.player.stop();
  playing = target ? { name: target, player: new Player(TRACKS[target]) } : null;
}

/** Request a background track while the calling component is mounted. Unmounting releases it
 *  after a short grace period, so switching screens (home → connecting → lobby) doesn't
 *  restart the tune. */
export function useMusic(track: MusicTrack) {
  useEffect(() => {
    desired = track;
    window.clearTimeout(pending);
    pending = window.setTimeout(sync, 0);
    return () => {
      if (desired !== track) return;
      desired = null;
      window.clearTimeout(pending);
      pending = window.setTimeout(sync, 1200);
    };
  }, [track]);
}
