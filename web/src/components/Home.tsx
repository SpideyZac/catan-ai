import { useState } from "react";
import { api, storage } from "../lib/api";
import { TerrainArt } from "./art";

export type QuickStart = "ai" | "local" | "online" | "watch";

const MODES: { id: QuickStart; title: string; text: string }[] = [
  { id: "ai", title: "Play vs AI", text: "Jump straight into a game against three computer opponents." },
  { id: "local", title: "Pass & Play", text: "Several people share this device; hands stay hidden between turns." },
  { id: "online", title: "Play online", text: "Open a table and send the invite link to friends on their own devices." },
  { id: "watch", title: "Watch the AI", text: "Sit back and watch four AIs negotiate, trade and race to 10." },
];

export function Home({ onEnter }: { onEnter: (code: string, quick?: QuickStart) => void }) {
  const [name, setName] = useState(storage.name());
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validName = name.trim().length > 0;
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
            <button key={m.id} className="mode" disabled={busy} onClick={() => create(m.id)}>
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
    </div>
  );
}
