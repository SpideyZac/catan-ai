import { useCallback, useEffect, useRef, useState } from "react";
import type { Action, BotLevel, RoomSettings, StateMessage } from "./types";

const TOKEN_KEY = (code: string) => `catan.token.${code}`;
const NAME_KEY = "catan.name";

function safeGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable (private mode) - reconnect tokens just won't persist */
  }
}

export const storage = {
  name: () => safeGet(NAME_KEY) ?? "",
  setName: (n: string) => safeSet(NAME_KEY, n),
  token: (code: string) => safeGet(TOKEN_KEY(code)),
  setToken: (code: string, token: string) => safeSet(TOKEN_KEY(code), token),
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  createRoom: (name: string) =>
    fetch("/api/rooms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => json<{ code: string; client_id: string; token: string }>(r)),
  roomInfo: (code: string) => fetch(`/api/rooms/${code}`).then((r) => json<{ code: string; status: string }>(r)),
  bots: () => fetch("/api/bots").then((r) => json<BotLevel[]>(r)),
};

export type ConnectionStatus = "connecting" | "open" | "closed" | "failed";

export interface RoomConnection {
  status: ConnectionStatus;
  state: StateMessage | null;
  error: string | null;
  clearError: () => void;
  send: (msg: Record<string, unknown>) => void;
  act: (seat: number, action: Action) => void;
  claimSeat: (seat: number, name?: string) => void;
  leaveSeat: (seat: number) => void;
  configure: (cfg: { settings?: RoomSettings; seats?: { kind: "human" | "bot"; bot?: string | null }[] }) => void;
  start: () => void;
  chat: (text: string) => void;
  backToLobby: () => void;
}

/** Live connection to a room with automatic reconnection (exponential backoff). */
export function useRoom(code: string, name: string): RoomConnection {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [state, setState] = useState<StateMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const attempts = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let pinger: number | undefined;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/${code}`);
      wsRef.current = ws;
      let fatal = false;
      ws.onopen = () => {
        attempts.current = 0;
        ws.send(JSON.stringify({ type: "hello", name, token: storage.token(code) }));
        pinger = window.setInterval(() => ws.readyState === 1 && ws.send('{"type":"ping"}'), 25_000);
      };
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "welcome") {
          storage.setToken(code, msg.token);
          setStatus("open");
        } else if (msg.type === "state") {
          setState(msg as StateMessage);
        } else if (msg.type === "error") {
          setError(msg.message);
          if (msg.fatal) {
            fatal = true;
            setStatus("failed");
          }
        }
      };
      ws.onclose = () => {
        window.clearInterval(pinger);
        if (cancelled || fatal) return;
        setStatus("closed");
        const delay = Math.min(10_000, 500 * 2 ** attempts.current++);
        timer = window.setTimeout(connect, delay);
      };
    };
    connect();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.clearInterval(pinger);
      wsRef.current?.close();
    };
  }, [code, name]);

  const send = useCallback((msg: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
    else setError("Not connected - reconnecting...");
  }, []);

  return {
    status,
    state,
    error,
    clearError: () => setError(null),
    send,
    act: (seat, action) => send({ type: "action", seat, action }),
    claimSeat: (seat, seatName) => send({ type: "claim_seat", seat, name: seatName ?? null }),
    leaveSeat: (seat) => send({ type: "leave_seat", seat }),
    configure: (cfg) => send({ type: "configure", ...cfg }),
    start: () => send({ type: "start" }),
    chat: (text) => send({ type: "chat", text }),
    backToLobby: () => send({ type: "back_to_lobby" }),
  };
}
