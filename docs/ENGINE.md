# Engine reference (`crates/catan-core`)

The engine is a pure-Rust, dependency-light implementation of the Settlers of Catan
base game for 2-4 players, with first-class player-to-player trading. It is the single
source of truth for the rules: the Python trainer, the game server and the bots all go
through it.

## Module map

| Module | Responsibility |
|---|---|
| `types.rs` | Resources, dev cards, terrains, harbors, costs, piece limits, helpers on `Hand = [u8; 5]` |
| `rng.rs` | Serializable xoshiro256++ PRNG (no `rand` dependency, deterministic, cheap to clone) |
| `topology.rs` | Static board graph built once: 19 hexes, 54 vertices, 72 edges, adjacency arrays and `u64`/`u128` bitmasks, coastal edges and the 9 harbor edges |
| `board.rs` | Terrain / number / harbor layout (random with no adjacent 6-8, or the rulebook beginner layout) |
| `state.rs` | `GameConfig`, `GameState`, `PlayerState`, `Phase`, `TradeOffer`, `Response`, resource bookkeeping that keeps the public hand belief in sync |
| `rules.rs` | Game construction, `actors`, `legal_actions`, `check`, `apply`/`apply_unchecked`, production, robber, dev cards, longest road / largest army |
| `action.rs` | The `Action` enum and `ActionError` |
| `trade.rs` | The 120 trade templates used by learning agents, O(1) template lookup, well-formedness |
| `encode.rs` | Fixed discrete action space (419) with masks, seat-relative observation vector (1788 floats) |
| `event.rs` | Event log for UIs + per-viewer redaction |
| `view.rs` | Information-filtered JSON views (`game_view`, `board_view`) |
| `bots.rs` | `RandomBot`, `HeuristicBot`, `play_game` |

## Board geometry

Hexes are pointy-topped, axial coordinates `(q, r)` with radius 2, numbered row by row
(rows of 3-4-5-4-3, top to bottom). Vertices and edges are derived from hex corners and
sorted top-to-bottom, left-to-right, so ids are stable across runs and platforms.
`board_view` exports positions for a hex of circumradius 1 (y grows downward), which the
web client scales directly.

Harbors sit on 9 of the 30 coastal edges with the standard spacing pattern
`3,3,4,3,3,4,3,3,4`; their kinds (4 generic 3:1, one 2:1 per resource) are shuffled per game.

## State machine

```
SetupSettlement -> SetupRoad -> (snake order 0..n-1, n-1..0) -> PreRoll
PreRoll --roll 7--> Discard? -> MoveRobber -> Main
PreRoll --roll--> Main
PreRoll --knight--> MoveRobber -> PreRoll
Main --knight--> MoveRobber -> Main
Main --road building--> RoadBuilding{remaining} -> Main
Main --year of plenty--> YearOfPlenty{remaining} -> Main
Main --monopoly--> Monopoly -> Main
Main --offer--> TradeResponse -> (any taker) TradeConfirm -> Main
                              -> (no taker)  Main
Main --end turn--> PreRoll (next player)
any --winner / max_turns--> GameOver
```

* `actors()` returns a bitmask of seats that may act. It has several bits set during
  `Discard` (everyone over the limit discards simultaneously, one card per action) and
  `TradeResponse` (every opponent answers independently). Otherwise it is the current player.
* `next_actor()` is the canonical sequential order for self-play drivers (first actor in
  seat order starting from the current player).
* The proposer may `CancelTrade` during `TradeResponse` even though it is not an actor.
* A player only wins on their own turn: after every action the current player's *total*
  VP (including hidden VP cards) is compared to `vp_to_win`.

### Rules implemented

Everything in the base game: distance rule, snake-draft setup with resources for the
second settlement, production with the bank-shortage rule (if the bank can't pay everyone,
nobody gets that resource unless exactly one player is owed it), discards on 7 (more than
`discard_limit` cards → discard half, rounded down), robber must move and must steal when
possible, harbors, development cards (cannot play a card bought this turn, at most one
non-VP card per turn, knights before or after rolling), longest road (≥5, broken by
opponent settlements, ties keep the holder; if the holder loses it and others tie, nobody
holds it), largest army (≥3, must strictly exceed), piece limits (15/5/4).

Dev cards are drawn uniformly from the remaining deck composition at purchase time, so a
cloned state never leaks the future deck order.

## Trading

Trades are expressed from the **acting** player's point of view:
`OfferTrade { give, want }` means "I give `give`, I receive `want`".

1. On their turn (in `Main`) the current player offers to all opponents. Limits:
   `max_trade_offers_per_turn` (default 3; the server uses 5) and `max_trade_cards` per side
   (default 6). Both sides must be non-empty and share no resource. The proposer must hold `give`.
2. `TradeResponse`: each opponent answers `AcceptTrade` (requires holding `want`),
   `RejectTrade`, or `OfferTrade` = **counter-offer** (their own terms; they must hold
   what they give and the proposer must hold what they ask). Counters are stored in the
   proposer's orientation inside `Response::Counter`.
3. When everyone has answered: if some acceptance/counter is still affordable the phase
   becomes `TradeConfirm`, otherwise the offer closes (`TradeCancelled` event).
4. `TradeConfirm`: the proposer picks `ConfirmTrade { partner }` or `CancelTrade`.

Humans may offer **any** well-formed trade. Learning agents choose from the 120
templates in `trade.rs` (1:1, 2:1, 1:2, 2-for-1 and 1-for-2 mixed shapes) both for
proposals and counters; they can still *respond* to arbitrary human offers because the
observation encodes the actual offer and `AcceptTrade`/`RejectTrade` don't depend on templates.

Maritime trades (`MaritimeTrade { give, get }`) automatically use the best ratio the
player has (4:1, 3:1 generic harbor, 2:1 matching harbor).

## Hidden information and the public belief

Every `PlayerState` carries `belief: [f32; 5]`: what an observer who saw every public
event knows about that hand. Production, builds, trades, discards, monopolies and Year
of Plenty are public and update it exactly. A robber steal is hidden from third parties:
the stolen card is subtracted from the victim's belief proportionally and added to the
thief's. Beliefs are renormalized to the public card count. Opponent hands in
observations are this belief; your own hand is exact. Bots use it too, so no agent peeks.

`view::game_view(state, viewer)` hides other players' hands and dev cards;
`Event::redacted_for(viewer)` hides dev cards bought by others and the resource of steals
you weren't part of.

## JSON action schema

Serialized with `#[serde(tag = "type", rename_all = "snake_case")]`:

```json
{"type": "roll_dice"}
{"type": "end_turn"}
{"type": "build_settlement", "vertex": 12}
{"type": "build_city", "vertex": 12}
{"type": "build_road", "edge": 30}
{"type": "buy_dev_card"}
{"type": "play_knight"} | {"type": "play_road_building"} | {"type": "play_year_of_plenty"} | {"type": "play_monopoly"}
{"type": "choose_resource", "resource": 3}
{"type": "move_robber", "hex": 7, "victim": 2}          // victim null if nobody can be robbed
{"type": "discard", "resource": 0}
{"type": "maritime_trade", "give": 0, "get": 4}
{"type": "offer_trade", "give": [1,0,0,0,0], "want": [0,0,0,1,0]}
{"type": "accept_trade"} | {"type": "reject_trade"} | {"type": "cancel_trade"}
{"type": "confirm_trade", "partner": 1}
```

Resource order everywhere: `wood, brick, sheep, wheat, ore` (UI labels: Lumber, Brick,
Wool, Grain, Ore). Dev card order: `knight, victory_point, road_building, year_of_plenty, monopoly`.

## Discrete action space (`encode.rs`, `ENCODING_VERSION = 1`)

All seat references are **relative to the actor** (1 = next player clockwise).

| Offset | Size | Meaning |
|---|---|---|
| 0 | 1 | end turn |
| 1 | 1 | roll |
| 2 | 1 | buy dev card |
| 3-6 | 4 | play knight / road building / year of plenty / monopoly |
| 7 | 5 | choose resource |
| 12 | 54 | build settlement at vertex |
| 66 | 54 | build city at vertex |
| 120 | 72 | build road at edge |
| 192 | 76 | move robber: `hex * 4 + slot` (slot 0 = nobody, 1-3 = relative victim) |
| 268 | 5 | discard resource |
| 273 | 20 | maritime: `give * 4 + (get skipping give)` |
| 293 | 120 | trade template: propose (Main) or counter (TradeResponse) |
| 413 | 1 | accept |
| 414 | 1 | reject |
| 415 | 3 | confirm with relative seat 1-3 |
| 418 | 1 | cancel trade |

Total `ACTION_SIZE = 419`. `legal_mask` produces the boolean mask.

## Observation (`OBS_SIZE = 1788` floats)

| Block | Count × width | Contents |
|---|---|---|
| hex | 19 × 19 | terrain one-hot (6), pips/5, robber, number one-hot 2..12 |
| vertex | 54 × 15 | settlement/city per relative seat (8), harbor kind one-hot (6), "I can settle here" |
| edge | 72 × 5 | road per relative seat (4), "I can build here" |
| player | 4 × 35 | per relative seat: resources (exact for me, belief for others), card count, my dev cards by type, dev count, new dev count, knights, public VP, longest road length, LR/LA flags, pieces left, harbors, discard pending, production pips per resource, seat present, is current |
| trade | 75 | active, offer give/want (proposer view), proposer relative seat, response one-hot per relative seat (pending/accept/reject/counter/n.a.), counter terms per seat |
| global | 42 | phase one-hot (12), bank, dev deck by type, last roll one-hot, dice rolled, dev played, offers left, turn/100, phase counter, vp_to_win/10, player-count one-hot |

The Python side reads offsets from `catan_ai._engine.OBS_OFFSETS`; never hard-code them.
**Any change to the action space or observation layout must bump `ENCODING_VERSION`**
(checkpoints record it and refuse to load on mismatch) and update this table.

## Performance

Measured on the development machine (single thread, release build):

| Benchmark | Throughput |
|---|---|
| random bots, full games | ~1,400 games/s, ~3.0M actions/s |
| heuristic bots, full games | ~4,200 games/s, ~2.3M actions/s |
| heuristic + observe + mask per decision | ~1,400 games/s, ~0.76M decisions/s |
| Python `VecEnv` (256 envs, random legal actions incl. numpy) | ~117k decisions/s |

Design choices behind this: bitboards for vertices/edges, incremental `blocked` and
`road_vertices` masks, longest-road recomputation only when roads change or a settlement
cuts a road, no heap allocations in `GameState` (the event log is empty unless enabled),
a reusable scratch `Vec<Action>` for move generation, and rayon + GIL release in `VecEnv`.

Run `cargo run --release -p catan-core --example bench -- 3000 heuristic --encode`.

## Testing

`cargo test -p catan-core --release` runs unit tests plus `tests/rules.rs`, which plays
hundreds of random and heuristic games checking after **every** action: resource
conservation, dev-card conservation, piece accounting, belief consistency, longest road
equals a fresh recomputation, the distance rule, and that every generated move passes
`check` while non-actors have nothing to do. Targeted tests cover the trade protocol
(accept, counter, confirm, cancel, offer caps, malformed offers), discards, robber rules,
dev-card timing, monopoly, year of plenty, bank shortage, longest road breaking, serde
round-trips, determinism and 2-3 player games.
