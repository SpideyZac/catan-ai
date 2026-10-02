import type { DevCardName, GameEvent, Hand, ResourceName } from "./types";
import { DEV_CARDS, RESOURCES } from "./types";

export const RESOURCE_LABEL: Record<ResourceName, string> = {
  wood: "Lumber",
  brick: "Brick",
  sheep: "Wool",
  wheat: "Grain",
  ore: "Ore",
};

export const DEV_LABEL: Record<DevCardName, string> = {
  knight: "Knight",
  victory_point: "Victory Point",
  road_building: "Road Building",
  year_of_plenty: "Year of Plenty",
  monopoly: "Monopoly",
};

export const DEV_HELP: Record<DevCardName, string> = {
  knight: "Move the robber and steal a card. Counts toward Largest Army.",
  victory_point: "Worth 1 victory point. Kept hidden until the game ends.",
  road_building: "Place 2 roads for free.",
  year_of_plenty: "Take any 2 resources from the bank.",
  monopoly: "Name a resource; every opponent gives you all of theirs.",
};

export function handToText(h: Hand): string {
  return h
    .map((n, i) => (n ? `${n} ${RESOURCE_LABEL[RESOURCES[i]]}` : ""))
    .filter(Boolean)
    .join(", ");
}

const possessive = (n: string) => (n === "You" ? "Your" : `${n}'s`);

/** Plain-text description of a log event; `name(i)` resolves seat names. */
export function describe(e: GameEvent, name: (seat: number) => string): string {
  const res = (r: number | null) => (r === null ? "a card" : RESOURCE_LABEL[RESOURCES[r]]);
  switch (e.type) {
    case "turn_started":
      return `${possessive(name(e.player))} turn`;
    case "dice_rolled":
      return `${name(e.player)} rolled ${e.dice[0] + e.dice[1]} (${e.dice[0]}+${e.dice[1]})`;
    case "produced":
      return `${name(e.player)} received ${handToText(e.resources)}`;
    case "settlement_built":
      return `${name(e.player)} built a settlement`;
    case "city_built":
      return `${name(e.player)} upgraded to a city`;
    case "road_built":
      return `${name(e.player)} built a road`;
    case "dev_card_bought":
      return e.card === null
        ? `${name(e.player)} bought a development card`
        : `${name(e.player)} bought ${DEV_LABEL[DEV_CARDS[e.card]]}`;
    case "dev_card_played":
      return `${name(e.player)} played ${DEV_LABEL[DEV_CARDS[e.card]]}`;
    case "robber_moved":
      return `${name(e.player)} moved the robber`;
    case "stolen":
      return `${name(e.thief)} stole ${res(e.resource)} from ${name(e.victim)}`;
    case "discarded":
      return `${name(e.player)} discarded ${res(e.resource)}`;
    case "monopoly_taken":
      return `${name(e.player)} monopolized ${e.amount} ${res(e.resource)}`;
    case "year_of_plenty_taken":
      return `${name(e.player)} took ${res(e.resource)} from the bank`;
    case "maritime_traded":
      return `${name(e.player)} traded ${handToText(e.give)} with the bank for ${handToText(e.get)}`;
    case "trade_offered":
      return `${name(e.proposer)} offers ${handToText(e.give)} for ${handToText(e.want)}`;
    case "trade_responded":
      if (e.response.kind === "counter" && e.response.give && e.response.want)
        return `${name(e.player)} counters: wants ${handToText(e.response.give)} for ${handToText(e.response.want)}`;
      return `${name(e.player)} ${e.response.kind === "accept" ? "accepts" : "declines"}`;
    case "trade_executed":
      return `${name(e.proposer)} traded ${handToText(e.give)} to ${name(e.partner)} for ${handToText(e.want)}`;
    case "trade_cancelled":
      return `${possessive(name(e.proposer))} offer closed`;
    case "longest_road_changed":
      return e.player === null ? "Longest Road is unclaimed" : `${name(e.player)} takes Longest Road (${e.length})`;
    case "largest_army_changed":
      return e.player === null ? "Largest Army is unclaimed" : `${name(e.player)} takes Largest Army (${e.knights})`;
    case "game_won":
      return `${name(e.player)} wins the game!`;
    case "game_truncated":
      return "The game ended without a winner";
  }
}

export const PHASE_HINT: Record<string, string> = {
  setup_settlement: "Place a settlement",
  setup_road: "Place a road next to it",
  pre_roll: "Roll the dice",
  discard: "Discard half your cards",
  move_robber: "Move the robber",
  main: "Build, trade or end your turn",
  road_building: "Place your free roads",
  year_of_plenty: "Pick resources from the bank",
  monopoly: "Name a resource to monopolize",
  trade_response: "Waiting for trade responses",
  trade_confirm: "Choose a trade partner",
  game_over: "Game over",
};
