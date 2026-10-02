import { useEffect, useRef, useState } from "react";
import { storage, useRoom } from "./lib/api";
import { GameScreen } from "./components/GameScreen";
import { Home, type QuickStart } from "./components/Home";
import { Lobby } from "./components/Lobby";

function parseRoute(): { code: string | null } {
  const m = location.pathname.match(/^\/room\/([A-Za-z0-9]{3,12})\/?$/);
  return { code: m ? m[1].toUpperCase() : null };
}

export default function App() {
  const [route, setRoute] = useState(parseRoute);
  const [quick, setQuick] = useState<QuickStart | undefined>(
    () => (sessionStorage.getItem("catan.quick") as QuickStart | null) ?? undefined,
  );
  useEffect(() => {
    const onPop = () => setRoute(parseRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const go = (path: string) => {
    history.pushState(null, "", path);
    setRoute(parseRoute());
  };

  if (!route.code) {
    return (
      <Home
        onEnter={(code, q) => {
          if (q) sessionStorage.setItem("catan.quick", q);
          setQuick(q);
          go(`/room/${code}`);
        }}
      />
    );
  }
  return (
    <NameGate>
      {(name) => (
        <Room
          code={route.code!}
          name={name}
          quick={quick}
          onQuickDone={() => {
            sessionStorage.removeItem("catan.quick");
            setQuick(undefined);
          }}
          onLeave={() => go("/")}
        />
      )}
    </NameGate>
  );
}

/** Ask for a display name before joining via an invite link. */
function NameGate({ children }: { children: (name: string) => React.ReactNode }) {
  const [name, setName] = useState(() => {
    // Invite links may carry a name: /room/CODE?name=Ada
    const fromUrl = new URLSearchParams(location.search).get("name")?.trim().slice(0, 24);
    if (fromUrl) storage.setName(fromUrl);
    return fromUrl || storage.name();
  });
  const [draft, setDraft] = useState("");
  if (name) return <>{children(name)}</>;
  return (
    <div className="home">
      <form
        className="home-card panel narrow"
        onSubmit={(e) => {
          e.preventDefault();
          if (!draft.trim()) return;
          storage.setName(draft.trim());
          setName(draft.trim());
        }}
      >
        <h1 className="title">
          <span className="logo-hex big" /> Join the table
        </h1>
        <label className="field stacked">
          <span>Your name</span>
          <input value={draft} maxLength={24} autoFocus onChange={(e) => setDraft(e.target.value)} />
        </label>
        <button className="btn good big" disabled={!draft.trim()}>
          Continue
        </button>
      </form>
    </div>
  );
}

function Room({
  code,
  name,
  quick,
  onQuickDone,
  onLeave,
}: {
  code: string;
  name: string;
  quick?: QuickStart;
  onQuickDone: () => void;
  onLeave: () => void;
}) {
  const conn = useRoom(code, name);
  const state = conn.state;
  const quickSent = useRef(false);

  // Apply the quick-start choice from the home screen once we're connected as host.
  useEffect(() => {
    if (!quick || quickSent.current || !state || !state.you.is_host || state.room.status !== "lobby") return;
    quickSent.current = true;
    const n = state.room.seats.length;
    if (quick === "ai") {
      conn.configure({ seats: [{ kind: "human" }, ...Array.from({ length: n - 1 }, () => ({ kind: "bot" as const, bot: "heuristic" }))] });
      conn.start();
    } else if (quick === "watch") {
      conn.configure({ seats: Array.from({ length: n }, () => ({ kind: "bot" as const, bot: "heuristic" })) });
      conn.start();
    } else if (quick === "local") {
      conn.configure({ seats: Array.from({ length: n }, () => ({ kind: "human" as const })) });
    } else if (quick === "online") {
      conn.configure({ seats: [{ kind: "human" }, { kind: "human" }, ...Array.from({ length: n - 2 }, () => ({ kind: "bot" as const, bot: "heuristic" }))] });
    }
    onQuickDone();
  }, [state, quick]); // eslint-disable-line react-hooks/exhaustive-deps

  if (conn.status === "failed") {
    return (
      <div className="home">
        <div className="home-card panel narrow">
          <h1 className="title">Table unavailable</h1>
          <p>{conn.error ?? "This table does not exist anymore."}</p>
          <button className="btn good" onClick={onLeave}>
            Back home
          </button>
        </div>
      </div>
    );
  }
  if (!state) {
    return (
      <div className="loading">
        <span className="logo-hex big spin" />
        <p>Connecting to table {code}…</p>
      </div>
    );
  }
  if (state.game && state.room.status !== "lobby") {
    return <GameScreen conn={conn} state={state} onLeave={onLeave} />;
  }
  return <Lobby conn={conn} state={state} onLeave={onLeave} />;
}
