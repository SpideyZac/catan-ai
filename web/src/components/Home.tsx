import { useEffect, useState } from "react";
import { api, storage } from "../lib/api";
import { useMusic } from "../lib/music";
import type { BotLevel, RoomSettings } from "../lib/types";
import { TerrainArt } from "./art";
import { MusicToggle, SoundToggle } from "./AudioToggles";
import { RulesFields } from "./Lobby";
import { Modal } from "./Modals";

export type QuickMode = "ai" | "local" | "online" | "watch";

/** Table rules and AI level chosen on the home page for the instant-start modes. */
export interface QuickSetup {
  settings: RoomSettings;
  bot: string;
}

export interface QuickStart {
  mode: QuickMode;
  setup?: QuickSetup;
}

const MODES: { id: QuickMode; title: string; text: string }[] = [
  { id: "ai", title: "Play vs AI", text: "Pick the table size, AI level and rules, then jump straight into a game." },
  { id: "local", title: "Pass & Play", text: "Several people share this device; hands stay hidden between turns." },
  { id: "online", title: "Play online", text: "Open a table and send the invite link to friends on their own devices." },
  { id: "watch", title: "Watch the AI", text: "Sit back and watch AIs negotiate, trade and race to victory." },
];

const SETUP_KEY = "catan.quickSetup";
const DEFAULT_SETUP: QuickSetup = {
  settings: { num_players: 4, vp_to_win: 10, max_trade_offers_per_turn: 5, beginner_board: false, bot_speed: "normal" },
  bot: "heuristic",
};

function loadSetup(): QuickSetup {
  try {
    const saved = JSON.parse(localStorage.getItem(SETUP_KEY) ?? "null") as Partial<QuickSetup> | null;
    if (!saved) return DEFAULT_SETUP;
    return {
      settings: { ...DEFAULT_SETUP.settings, ...saved.settings },
      bot: typeof saved.bot === "string" ? saved.bot : DEFAULT_SETUP.bot,
    };
  } catch {
    return DEFAULT_SETUP;
  }
}

function saveSetup(setup: QuickSetup) {
  try {
    localStorage.setItem(SETUP_KEY, JSON.stringify(setup));
  } catch {
    /* storage unavailable - the choice just isn't remembered */
  }
}

export function Home({ onEnter }: { onEnter: (code: string, quick?: QuickStart) => void }) {
  const [name, setName] = useState(storage.name());
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [setupFor, setSetupFor] = useState<"ai" | "watch" | null>(null);
  useMusic("lobby");

  const validName = name.trim().length > 0;
  const choose = (mode: QuickMode) => {
    if (!validName) return setError("Enter your name first");
    setError(null);
    if (mode === "ai" || mode === "watch") setSetupFor(mode);
    else void create({ mode });
  };
  const create = async (quick: QuickStart) => {
    if (!validName) return setError("Enter your name first");
    setBusy(true);
    setError(null);
    try {
      storage.setName(name.trim());
      const r = await api.createRoom(name.trim());
      storage.setToken(r.code, r.token);
      onEnter(r.code, quick);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create a table");
    } finally {
      setBusy(false);
    }
  };
  const join = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validName) return setError("Enter your name first");
    const c = code.trim().toUpperCase();
    if (!c) return;
    try {
      await api.roomInfo(c);
      storage.setName(name.trim());
      onEnter(c);
    } catch {
      setError(`No table with code ${c}`);
    }
  };

  return (
    <div className="home">
      <svg className="home-bg" viewBox="-300 -300 600 600" aria-hidden>
        {(["forest", "fields", "pasture", "mountains", "hills", "forest", "fields"] as const).map((t, i) => {
          const a = (i * Math.PI) / 3;
          const x = i === 6 ? 0 : Math.cos(a) * 173;
          const y = i === 6 ? 0 : Math.sin(a) * 173;
          return (
            <g key={i} transform={`translate(${x} ${y})`} opacity="0.9">
              <polygon
                points={Array.from({ length: 6 }, (_, k) => {
                  const b = ((60 * k - 90) * Math.PI) / 180;
                  return `${Math.cos(b) * 98},${Math.sin(b) * 98}`;
                }).join(" ")}
                fill={`url(#hg-${t})`}
                stroke="#4a3216"
                strokeWidth="4"
              />
              <TerrainArt terrain={t} seed={i * 31 + 7} r={95} />
            </g>
          );
        })}
        <defs>
          {[
            ["forest", "#3f8a3a", "#1f5a22"],
            ["fields", "#f4d061", "#d9a92c"],
            ["pasture", "#a6dc6c", "#6fb043"],
            ["mountains", "#a3a8b3", "#6b717d"],
            ["hills", "#d9824b", "#a94d24"],
          ].map(([id, a, b]) => (
            <radialGradient key={id} id={`hg-${id}`}>
              <stop offset="0%" stopColor={a} />
              <stop offset="100%" stopColor={b} />
            </radialGradient>
          ))}
        </defs>
      </svg>

      <div className="home-card panel">
        <div className="audio-corner">
          <MusicToggle />
          <SoundToggle />
        </div>
        <h1 className="title">
          <span className="logo-hex big" /> Catan <em>AI</em>
        </h1>
        <p className="tagline">Settle the island. Trade like you mean it. Outwit an AI that actually negotiates.</p>
        <label className="field stacked">
          <span>Your name</span>
          <input
            value={name}
            maxLength={24}
            placeholder="e.g. Ada"
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
        </label>
        <div className="modes">
          {MODES.map((m) => (
            <button key={m.id} className="mode" disabled={busy} onClick={() => choose(m.id)}>
              <b>{m.title}</b>
              <span>{m.text}</span>
            </button>
          ))}
        </div>
        <form className="join" onSubmit={join}>
          <input
            value={code}
            maxLength={8}
            placeholder="Table code"
            onChange={(e) => setCode(e.target.value.toUpperCase())}
          />
          <button className="btn" disabled={!code.trim()}>
            Join table
          </button>
        </form>
        {error && <p className="warn">{error}</p>}
      </div>
      {setupFor && (
        <QuickSetupModal
          mode={setupFor}
          busy={busy}
          onCancel={() => setSetupFor(null)}
          onStart={(setup) => {
            saveSetup(setup);
            setSetupFor(null);
            void create({ mode: setupFor, setup });
          }}
        />
      )}
    </div>
  );
}

/** Rules and AI level for "Play vs AI" / "Watch the AI" before the game starts. */
function QuickSetupModal({
  mode,
  busy,
  onCancel,
  onStart,
}: {
  mode: "ai" | "watch";
  busy: boolean;
  onCancel: () => void;
  onStart: (setup: QuickSetup) => void;
}) {
  const [setup, setSetup] = useState(loadSetup);
  const [bots, setBots] = useState<BotLevel[]>([]);
  useEffect(() => {
    api
      .bots()
      .then(setBots)
      .catch(() => setBots([]));
  }, []);
  // A remembered level may have disappeared (e.g. a checkpoint removed from the server).
  const known = bots.length === 0 || bots.some((b) => b.id === setup.bot);
  const bot = known ? setup.bot : DEFAULT_SETUP.bot;
  const level = bots.find((b) => b.id === bot);
  const n = setup.settings.num_players;
  const opponents = mode === "ai" ? n - 1 : n;

  return (
    <Modal title={mode === "ai" ? "Play vs AI" : "Watch the AI"} onClose={onCancel}>
      <div className="quick-setup">
        <RulesFields
          settings={setup.settings}
          onChange={(patch) => setSetup({ ...setup, settings: { ...setup.settings, ...patch } })}
        />
        <label className="field">
          <span>AI level</span>
          <select value={bot} onChange={(e) => setSetup({ ...setup, bot: e.target.value })}>
            {(bots.length ? bots : [{ id: "heuristic", name: "Heuristic", description: "" }]).map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        </label>
        {level?.description && <p className="muted hint">{level.description}</p>}
        <p className="muted hint">
          {mode === "ai"
            ? `You against ${opponents} computer opponent${opponents > 1 ? "s" : ""}.`
            : `${opponents} computer players at the table.`}
        </p>
        <div className="btn-row">
          <button className="btn good" disabled={busy} onClick={() => onStart({ ...setup, bot })}>
            {mode === "ai" ? "Start game" : "Start watching"}
          </button>
          <button className="btn ghost" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}
