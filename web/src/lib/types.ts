// Types mirroring the server protocol (see docs/SERVER.md) and engine JSON (docs/ENGINE.md).

export type Hand = [number, number, number, number, number];
export const RESOURCES = ["wood", "brick", "sheep", "wheat", "ore"] as const;
export type ResourceName = (typeof RESOURCES)[number];
export const DEV_CARDS = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"] as const;
export type DevCardName = (typeof DEV_CARDS)[number];
export type Terrain = "forest" | "hills" | "pasture" | "fields" | "mountains" | "desert";
export type PortKindName = ResourceName | "generic";

export interface HexInfo {
  id: number;
  q: number;
  r: number;
  x: number;
  y: number;
  terrain: Terrain;
  number: number;
  vertices: number[];
  edges: number[];
}

export interface BoardInfo {
  hexes: HexInfo[];
  ports: { kind: PortKindName; edge: number; vertices: [number, number] }[];
  vertices: [number, number][];
  edges: [number, number][];
}

export type Action =
  | { type: "roll_dice" }
  | { type: "end_turn" }
  | { type: "build_settlement"; vertex: number }
  | { type: "build_city"; vertex: number }
  | { type: "build_road"; edge: number }
  | { type: "buy_dev_card" }
  | { type: "play_knight" }
  | { type: "play_road_building" }
  | { type: "play_year_of_plenty" }
  | { type: "play_monopoly" }
  | { type: "choose_resource"; resource: number }
  | { type: "move_robber"; hex: number; victim: number | null }
  | { type: "discard"; resource: number }
  | { type: "maritime_trade"; give: number; get: number }
  | { type: "offer_trade"; give: Hand; want: Hand }
  | { type: "accept_trade" }
  | { type: "reject_trade" }
  | { type: "confirm_trade"; partner: number }
  | { type: "cancel_trade" };

export type PhaseName =
  | "setup_settlement"
  | "setup_road"
  | "pre_roll"
  | "discard"
  | "move_robber"
  | "main"
  | "road_building"
  | "year_of_plenty"
  | "monopoly"
  | "trade_response"
  | "trade_confirm"
  | "game_over";

export interface Phase {
  name: PhaseName;
  vertex?: number;
  remaining?: number;
}

export type TradeResponse =
  | { kind: "pending" }
  | { kind: "accept" }
  | { kind: "reject" }
  | { kind: "counter"; give: Hand; want: Hand }
  | { kind: "not_involved" };

export interface TradeOffer {
  proposer: number;
  give: Hand;
  want: Hand;
  responses: TradeResponse[];
}

export interface PlayerView {
  seat: number;
  resources: Hand | null;
  resource_count: number;
  dev_cards: Hand | null;
  new_dev_cards: Hand | null;
  dev_card_count: number;
  knights_played: number;
  public_vp: number;
  total_vp: number | null;
  longest_road: number;
  has_longest_road: boolean;
  has_largest_army: boolean;
  roads_left: number;
  settlements_left: number;
  cities_left: number;
  ports: PortKindName[];
  discard_pending: number;
}

export interface GameView {
  viewer: number | null;
  num_players: number;
  vp_to_win: number;
  phase: Phase;
  current: number;
  turn: number;
  robber: number;
  last_roll: [number, number] | null;
  dice_rolled: boolean;
  dev_played_this_turn: boolean;
  trade_offers_left: number;
  bank: Hand;
  dev_deck_count: number;
  buildings: { vertex: number; player: number; city: boolean }[];
  roads: { edge: number; player: number }[];
  players: PlayerView[];
  trade: TradeOffer | null;
  winner: number | null;
  actors: number[];
}

export type GameEvent =
  | { type: "turn_started"; player: number; turn: number }
  | { type: "dice_rolled"; player: number; dice: [number, number] }
  | { type: "produced"; player: number; resources: Hand }
  | { type: "settlement_built"; player: number; vertex: number }
  | { type: "city_built"; player: number; vertex: number }
  | { type: "road_built"; player: number; edge: number }
  | { type: "dev_card_bought"; player: number; card: number | null }
  | { type: "dev_card_played"; player: number; card: number }
  | { type: "robber_moved"; player: number; hex: number }
  | { type: "stolen"; thief: number; victim: number; resource: number | null }
  | { type: "discarded"; player: number; resource: number }
  | { type: "monopoly_taken"; player: number; resource: number; amount: number }
  | { type: "year_of_plenty_taken"; player: number; resource: number }
  | { type: "maritime_traded"; player: number; give: Hand; get: Hand }
  | { type: "trade_offered"; proposer: number; give: Hand; want: Hand }
  | { type: "trade_responded"; player: number; response: { kind: "accept" | "reject" | "counter"; give?: Hand; want?: Hand } }
  | { type: "trade_executed"; proposer: number; partner: number; give: Hand; want: Hand }
  | { type: "trade_cancelled"; proposer: number }
  | { type: "longest_road_changed"; player: number | null; length: number }
  | { type: "largest_army_changed"; player: number | null; knights: number }
  | { type: "game_won"; player: number }
  | { type: "game_truncated" };

export interface LogEntry {
  seq: number;
  ts: number;
  event: GameEvent;
}

export interface SeatInfo {
  index: number;
  kind: "human" | "bot";
  bot: string | null;
  client_id: string | null;
  name: string;
  color: PlayerColor;
  connected: boolean;
}

export type PlayerColor = "red" | "blue" | "white" | "orange";

export interface RoomSettings {
  num_players: number;
  vp_to_win: number;
  max_trade_offers_per_turn: number;
  beginner_board: boolean;
  bot_speed: "fast" | "normal" | "slow";
}

export interface ChatMessage {
  from: string;
  client_id: string;
  text: string;
  ts: number;
}

export interface RoomView {
  code: string;
  status: "lobby" | "playing" | "finished";
  host_id: string | null;
  settings: RoomSettings;
  seats: SeatInfo[];
  clients: { id: string; name: string; connected: boolean }[];
  chat: ChatMessage[];
  game_id: number;
}

export interface GamePayload {
  id: number;
  board: BoardInfo;
  views: Record<string, GameView>;
  legal: Record<string, Action[]>;
  can_offer: Record<string, boolean>;
  log: LogEntry[];
  trade_started_at: number | null;
}

export interface StateMessage {
  type: "state";
  room: RoomView;
  you: { client_id: string; seats: number[]; is_host: boolean };
  game: GamePayload | null;
}

export interface BotLevel {
  id: string;
  name: string;
  description: string;
}

export const emptyHand = (): Hand => [0, 0, 0, 0, 0];
export const handTotal = (h: Hand) => h.reduce((a, b) => a + b, 0);
export const covers = (have: Hand, need: Hand) => have.every((x, i) => x >= need[i]);

export const COSTS: Record<"road" | "settlement" | "city" | "dev", Hand> = {
  road: [1, 1, 0, 0, 0],
  settlement: [1, 1, 1, 1, 0],
  city: [0, 0, 0, 2, 3],
  dev: [0, 0, 1, 1, 1],
};
