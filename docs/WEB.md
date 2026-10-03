# Web client (`web/`)

React 19 + TypeScript + Vite. No UI framework, no image assets: the board and all icons are
procedural SVG, styled with one stylesheet (`src/styles.css`).

## Commands

```bash
cd web
npm ci
npm run dev        # http://localhost:5173, proxies /api and /ws to localhost:8000
npm run build      # typecheck + production bundle in web/dist (served by catan-server)
npm run typecheck
```

## Structure

| File | Role |
|---|---|
| `src/App.tsx` | Tiny router (`/` home, `/room/CODE` table), name gate (`?name=` supported), quick-start handling |
| `src/lib/api.ts` | REST helpers, `useRoom` hook: WebSocket with hello/token, exponential-backoff reconnect, pings, typed senders |
| `src/lib/types.ts` | Protocol & engine JSON types, costs, hand helpers |
| `src/lib/sound.ts` | Procedural sound effects (Web Audio), event → sound mapping, persisted mute toggle |
| `src/lib/format.ts` | Labels (Lumber/Brick/Wool/Grain/Ore), event descriptions, phase hints |
| `src/components/Home.tsx` | Landing page: Play vs AI, Pass & Play, Play online, Watch the AI, join by code |
| `src/components/Lobby.tsx` | Seats (human/AI level, claim, add local player), table rules, invite link, chat |
| `src/components/GameScreen.tsx` | Game layout, perspective logic, build modes, board targets, toasts, log, chat, modals |
| `src/components/Board.tsx` | SVG board: frame, sea, beach, harbors, tiles, tokens, roads, buildings, robber, click targets |
| `src/components/art.tsx` | Terrain illustrations, resource/dev icons, settlement/city/robber shapes, number tokens, palettes |
| `src/components/TradePanel.tsx` | Compose offers, answer/counter offers, pick partners, bank/harbor trades |
| `src/components/Modals.tsx` | Discard, resource picker (Year of Plenty / Monopoly), robber victim, pass-device curtain, game over, rules |
| `src/components/Cards.tsx`, `Dice.tsx`, `PlayerPanel.tsx` | Cards, steppers, chips, dice animation, player summaries |

## Perspective (pass & play)

A client may control several seats. `GameScreen` picks a **focus** seat: an acting seat
the client controls (preferring the current player), otherwise the proposer of a pending
offer, otherwise the last focus. When the client controls more than one seat and the focus
changes to a seat that hasn't been "revealed", a full-screen curtain asks the next player
to take the device before their hand is shown.

## Board interaction

The server sends the legal actions for each controlled seat (trade templates stripped).
The board derives click targets from them:

* setup / road building: vertices or edges are highlighted automatically;
* main phase: pressing Road / Settlement / City enters a build mode;
* robber: legal hexes are highlighted; if several opponents can be robbed, a victim picker opens.

## Sound

`src/lib/sound.ts` synthesizes every effect with the Web Audio API (oscillators + filtered
noise), so there are no audio files. `GameScreen` maps fresh log events to effects
(`sfxFor`): dice rattle (plus a low rumble on a 7), knocks for roads/settlements/cities,
card flip, robber growl, steal whoosh (a sadder variant when you are the victim), trade
offer bell, coin chime on any completed trade, fanfare for Longest Road / Largest Army and
win/lose stings. A chime plays when the turn passes to one of your seats and a pop on
incoming chat. Effects from one state update are deduplicated, staggered and capped at four.
The AudioContext is created lazily and resumed on the first click/key press (autoplay
policy). The speaker button in the top bar mutes; the choice is stored in `localStorage`
(`catan.muted`).

## Styling

Theme tokens live at the top of `styles.css` (wood, parchment, ink, gold). Fonts: Cinzel
(titles) and Nunito (UI) from Google Fonts. Layout is a CSS grid (players | board | side
panel, dock below) collapsing to a single column under 860px with a sticky dock.

## UI smoke test

`npm run ui-smoke` (with `uv run catan-server` running on :8000) drives headless Chrome via
`playwright-core`: home → Play vs AI → setup placements → roll (handling a 7) → compose and
send a trade offer; then Pass & Play lobby → add a local player → set AI seats → start →
pass-device curtain. It fails on any page error. Options: `BASE_URL`, `CHROME_PATH`
(defaults to the standard Windows install), `-- --out <dir>` for screenshots.
