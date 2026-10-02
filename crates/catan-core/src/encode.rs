//! Fixed-size encodings for learning agents: a discrete action space with legality
//! masks, and a flat, seat-relative observation vector.
//!
//! Everything here is *relative to the acting player*: seat 0 in the encoding is
//! always "me", seat 1 is the next player clockwise, and so on. This makes a single
//! policy usable from any seat.
//!
//! The layout constants below are the contract with the Python side; any change must
//! bump [`ENCODING_VERSION`] and be documented in `docs/AI.md`.

use crate::action::Action;
use crate::state::{GameState, Phase, Response};
use crate::topology::{topo, NONE};
use crate::trade::{template_index, templates, NUM_TRADE_TEMPLATES};
use crate::types::*;

pub const ENCODING_VERSION: u32 = 1;

// ---------------------------------------------------------------- action space

pub const A_END_TURN: usize = 0;
pub const A_ROLL: usize = 1;
pub const A_BUY_DEV: usize = 2;
pub const A_PLAY_KNIGHT: usize = 3;
pub const A_PLAY_ROAD_BUILDING: usize = 4;
pub const A_PLAY_YOP: usize = 5;
pub const A_PLAY_MONOPOLY: usize = 6;
pub const A_CHOOSE_RESOURCE: usize = 7; // +5
pub const A_SETTLEMENT: usize = A_CHOOSE_RESOURCE + NUM_RESOURCES; // +54
pub const A_CITY: usize = A_SETTLEMENT + NUM_VERTICES; // +54
pub const A_ROAD: usize = A_CITY + NUM_VERTICES; // +72
/// hex * 4 + victim slot, where slot 0 = nobody and 1..=3 = relative seat.
pub const A_ROBBER: usize = A_ROAD + NUM_EDGES; // +76
pub const A_DISCARD: usize = A_ROBBER + NUM_HEXES * 4; // +5
/// give * 4 + (get index skipping `give`).
pub const A_MARITIME: usize = A_DISCARD + NUM_RESOURCES; // +20
/// Template index; means "propose" in the main phase and "counter" when responding.
pub const A_OFFER: usize = A_MARITIME + 20; // +120
pub const A_ACCEPT: usize = A_OFFER + NUM_TRADE_TEMPLATES;
pub const A_REJECT: usize = A_ACCEPT + 1;
/// Relative seat - 1 of the partner.
pub const A_CONFIRM: usize = A_REJECT + 1; // +3
pub const A_CANCEL: usize = A_CONFIRM + 3;
pub const ACTION_SIZE: usize = A_CANCEL + 1;

#[inline]
fn rel(n: u8, actor: u8, seat: u8) -> u8 {
    (seat + n - actor) % n
}

#[inline]
fn abs(n: u8, actor: u8, rel_seat: u8) -> u8 {
    (actor + rel_seat) % n
}

/// Encode an action taken by `actor` into the discrete action space.
/// Returns `None` for actions outside the catalogue (non-template trade offers).
pub fn action_to_index(state: &GameState, actor: u8, a: &Action) -> Option<usize> {
    let n = state.config.num_players;
    Some(match *a {
        Action::EndTurn => A_END_TURN,
        Action::RollDice => A_ROLL,
        Action::BuyDevCard => A_BUY_DEV,
        Action::PlayKnight => A_PLAY_KNIGHT,
        Action::PlayRoadBuilding => A_PLAY_ROAD_BUILDING,
        Action::PlayYearOfPlenty => A_PLAY_YOP,
        Action::PlayMonopoly => A_PLAY_MONOPOLY,
        Action::ChooseResource { resource } => A_CHOOSE_RESOURCE + resource as usize,
        Action::BuildSettlement { vertex } => A_SETTLEMENT + vertex as usize,
        Action::BuildCity { vertex } => A_CITY + vertex as usize,
        Action::BuildRoad { edge } => A_ROAD + edge as usize,
        Action::MoveRobber { hex, victim } => {
            let slot = victim.map_or(0, |v| rel(n, actor, v) as usize);
            A_ROBBER + hex as usize * 4 + slot
        }
        Action::Discard { resource } => A_DISCARD + resource as usize,
        Action::MaritimeTrade { give, get } => {
            let g = give as usize;
            let r = get as usize;
            A_MARITIME + g * 4 + if r > g { r - 1 } else { r }
        }
        Action::OfferTrade { give, want } => A_OFFER + template_index(&give, &want)?,
        Action::AcceptTrade => A_ACCEPT,
        Action::RejectTrade => A_REJECT,
        Action::ConfirmTrade { partner } => A_CONFIRM + rel(n, actor, partner) as usize - 1,
        Action::CancelTrade => A_CANCEL,
    })
}

/// Decode an action index chosen by `actor`. The result is not guaranteed legal;
/// check it against the mask (or [`GameState::check`]).
pub fn index_to_action(state: &GameState, actor: u8, idx: usize) -> Option<Action> {
    let n = state.config.num_players;
    Some(match idx {
        A_END_TURN => Action::EndTurn,
        A_ROLL => Action::RollDice,
        A_BUY_DEV => Action::BuyDevCard,
        A_PLAY_KNIGHT => Action::PlayKnight,
        A_PLAY_ROAD_BUILDING => Action::PlayRoadBuilding,
        A_PLAY_YOP => Action::PlayYearOfPlenty,
        A_PLAY_MONOPOLY => Action::PlayMonopoly,
        i if i < A_SETTLEMENT => Action::ChooseResource {
            resource: (i - A_CHOOSE_RESOURCE) as u8,
        },
        i if i < A_CITY => Action::BuildSettlement {
            vertex: (i - A_SETTLEMENT) as u8,
        },
        i if i < A_ROAD => Action::BuildCity {
            vertex: (i - A_CITY) as u8,
        },
        i if i < A_ROBBER => Action::BuildRoad {
            edge: (i - A_ROAD) as u8,
        },
        i if i < A_DISCARD => {
            let k = i - A_ROBBER;
            let slot = (k % 4) as u8;
            if slot >= n {
                return None;
            }
            Action::MoveRobber {
                hex: (k / 4) as u8,
                victim: (slot > 0).then(|| abs(n, actor, slot)),
            }
        }
        i if i < A_MARITIME => Action::Discard {
            resource: (i - A_DISCARD) as u8,
        },
        i if i < A_OFFER => {
            let k = i - A_MARITIME;
            let g = k / 4;
            let mut r = k % 4;
            if r >= g {
                r += 1;
            }
            Action::MaritimeTrade {
                give: g as u8,
                get: r as u8,
            }
        }
        i if i < A_ACCEPT => {
            let (give, want) = templates()[i - A_OFFER];
            Action::OfferTrade { give, want }
        }
        A_ACCEPT => Action::AcceptTrade,
        A_REJECT => Action::RejectTrade,
        i if i < A_CANCEL => {
            let r = (i - A_CONFIRM + 1) as u8;
            if r >= n {
                return None;
            }
            Action::ConfirmTrade {
                partner: abs(n, actor, r),
            }
        }
        A_CANCEL => Action::CancelTrade,
        _ => return None,
    })
}

/// Fill `mask` (length [`ACTION_SIZE`]) with the legal actions of `actor`.
/// `scratch` is reused to avoid allocations. Returns the number of legal actions.
pub fn legal_mask(state: &GameState, actor: u8, mask: &mut [bool], scratch: &mut Vec<Action>) -> usize {
    debug_assert_eq!(mask.len(), ACTION_SIZE);
    mask.fill(false);
    state.legal_actions(actor, scratch);
    let mut count = 0;
    for a in scratch.iter() {
        if let Some(i) = action_to_index(state, actor, a) {
            if !mask[i] {
                mask[i] = true;
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------- observation

pub const HEX_FEATURES: usize = 6 /*terrain*/ + 1 /*pips*/ + 1 /*robber*/ + 11 /*number one-hot 2..12*/;
pub const VERTEX_FEATURES: usize = MAX_PLAYERS * 2 /*settlement/city per rel seat*/ + 6 /*port kind*/ + 1 /*buildable for me*/;
pub const EDGE_FEATURES: usize = MAX_PLAYERS /*road per rel seat*/ + 1 /*buildable for me*/;
pub const PLAYER_FEATURES: usize = 5 /*resources (exact for me, belief for others)*/
    + 1 /*total cards*/
    + 5 /*dev cards by type (mine only)*/
    + 1 /*hidden dev count*/
    + 1 /*new dev cards*/
    + 1 /*knights played*/
    + 1 /*public vp*/
    + 1 /*longest road length*/
    + 1 /*has longest road*/
    + 1 /*has largest army*/
    + 3 /*pieces left*/
    + 6 /*ports*/
    + 1 /*discard pending*/
    + 5 /*production pips per resource*/
    + 1 /*seat present*/
    + 1 /*is current player*/;
pub const NUM_PHASES: usize = 12;
pub const TRADE_FEATURES: usize = 1 /*active*/ + 5 + 5 /*offer give/want (proposer view)*/
    + MAX_PLAYERS /*proposer rel seat*/
    + MAX_PLAYERS * 5 /*response one-hot: pending/accept/reject/counter/n.a.*/
    + MAX_PLAYERS * 10 /*counter terms*/;
pub const GLOBAL_FEATURES: usize = NUM_PHASES + 5 /*bank*/ + 5 /*dev deck*/ + 11 /*last roll*/
    + 1 /*dice rolled*/ + 1 /*dev played*/ + 1 /*offers left*/ + 1 /*turn*/ + 1 /*phase-specific counter*/
    + 1 /*vp to win*/ + 3 /*num players one-hot 2..4*/;

pub const OBS_HEX_OFFSET: usize = 0;
pub const OBS_VERTEX_OFFSET: usize = OBS_HEX_OFFSET + NUM_HEXES * HEX_FEATURES;
pub const OBS_EDGE_OFFSET: usize = OBS_VERTEX_OFFSET + NUM_VERTICES * VERTEX_FEATURES;
pub const OBS_PLAYER_OFFSET: usize = OBS_EDGE_OFFSET + NUM_EDGES * EDGE_FEATURES;
pub const OBS_TRADE_OFFSET: usize = OBS_PLAYER_OFFSET + MAX_PLAYERS * PLAYER_FEATURES;
pub const OBS_GLOBAL_OFFSET: usize = OBS_TRADE_OFFSET + TRADE_FEATURES;
pub const OBS_SIZE: usize = OBS_GLOBAL_OFFSET + GLOBAL_FEATURES;

pub fn phase_index(p: &Phase) -> usize {
    match p {
        Phase::SetupSettlement => 0,
        Phase::SetupRoad { .. } => 1,
        Phase::PreRoll => 2,
        Phase::Discard => 3,
        Phase::MoveRobber => 4,
        Phase::Main => 5,
        Phase::RoadBuilding { .. } => 6,
        Phase::YearOfPlenty { .. } => 7,
        Phase::Monopoly => 8,
        Phase::TradeResponse => 9,
        Phase::TradeConfirm => 10,
        Phase::GameOver => 11,
    }
}

/// Write the observation of `actor` into `out` (length [`OBS_SIZE`]).
/// Only information available to `actor` is encoded (opponent hands use the public belief).
pub fn observe(state: &GameState, actor: u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let t = topo();
    let n = state.config.num_players;
    let a = actor as usize;

    // Hexes.
    for h in 0..NUM_HEXES {
        let o = OBS_HEX_OFFSET + h * HEX_FEATURES;
        out[o + state.board.terrain[h] as usize] = 1.0;
        let num = state.board.numbers[h];
        out[o + 6] = pips(num) as f32 / 5.0;
        out[o + 7] = (state.robber as usize == h) as u8 as f32;
        if num >= 2 {
            out[o + 8 + (num as usize - 2)] = 1.0;
        }
    }

    // Vertices.
    let my_settle_spots = if state.phase == Phase::SetupSettlement {
        state.settlement_spots(a)
    } else {
        state.players[a].road_vertices & !state.blocked
    };
    for v in 0..NUM_VERTICES {
        let o = OBS_VERTEX_OFFSET + v * VERTEX_FEATURES;
        let owner = state.vertex_owner[v];
        if owner != NONE {
            let r = rel(n, actor, owner) as usize;
            let is_city = state.players[owner as usize].cities & (1u64 << v) != 0;
            out[o + r * 2 + is_city as usize] = 1.0;
        }
        if let Some(port) = state.board.vertex_port[v] {
            out[o + MAX_PLAYERS * 2 + port as usize] = 1.0;
        }
        out[o + MAX_PLAYERS * 2 + 6] = (my_settle_spots & (1u64 << v) != 0) as u8 as f32;
    }

    // Edges.
    let my_road_spots = state.road_spots(a);
    for e in 0..NUM_EDGES {
        let o = OBS_EDGE_OFFSET + e * EDGE_FEATURES;
        let owner = state.edge_owner[e];
        if owner != NONE {
            out[o + rel(n, actor, owner) as usize] = 1.0;
        }
        out[o + MAX_PLAYERS] = (my_road_spots & (1u128 << e) != 0) as u8 as f32;
    }

    // Players.
    for r in 0..n {
        let p = abs(n, actor, r) as usize;
        let ps = &state.players[p];
        let o = OBS_PLAYER_OFFSET + r as usize * PLAYER_FEATURES;
        let mut i = o;
        for res in 0..NUM_RESOURCES {
            out[i] = if p == a {
                ps.resources[res] as f32
            } else {
                ps.belief[res]
            } / 5.0;
            i += 1;
        }
        out[i] = ps.num_resources() as f32 / 10.0;
        i += 1;
        for d in 0..NUM_DEV_TYPES {
            if p == a {
                out[i] = (ps.dev_cards[d] + ps.new_dev_cards[d]) as f32 / 3.0;
            }
            i += 1;
        }
        out[i] = ps.num_dev_cards() as f32 / 5.0;
        i += 1;
        if p == a {
            out[i] = ps.new_dev_cards.iter().map(|&x| x as f32).sum::<f32>() / 3.0;
        }
        i += 1;
        out[i] = ps.knights_played as f32 / 5.0;
        i += 1;
        out[i] = state.public_vp(p) as f32 / 10.0;
        i += 1;
        out[i] = ps.longest_road as f32 / 10.0;
        i += 1;
        out[i] = (state.longest_road_owner == Some(p as u8)) as u8 as f32;
        i += 1;
        out[i] = (state.largest_army_owner == Some(p as u8)) as u8 as f32;
        i += 1;
        out[i] = ps.roads_left as f32 / MAX_ROADS as f32;
        out[i + 1] = ps.settlements_left as f32 / MAX_SETTLEMENTS as f32;
        out[i + 2] = ps.cities_left as f32 / MAX_CITIES as f32;
        i += 3;
        for k in 0..6 {
            out[i + k] = (ps.ports & (1 << k) != 0) as u8 as f32;
        }
        i += 6;
        out[i] = ps.discard_pending as f32 / 5.0;
        i += 1;
        // Expected production (pips) per resource, cities counted double, robber excluded.
        for h in 0..NUM_HEXES {
            if h == state.robber as usize {
                continue;
            }
            let Some(res) = state.board.hex_resource(h) else {
                continue;
            };
            let pip = pips(state.board.numbers[h]) as f32;
            for &v in &t.hex_vertices[h] {
                let bit = 1u64 << v;
                if ps.settlements & bit != 0 {
                    out[i + res as usize] += pip / 10.0;
                } else if ps.cities & bit != 0 {
                    out[i + res as usize] += 2.0 * pip / 10.0;
                }
            }
        }
        i += 5;
        out[i] = 1.0;
        i += 1;
        out[i] = (p as u8 == state.current) as u8 as f32;
        i += 1;
        debug_assert_eq!(i, o + PLAYER_FEATURES);
    }

    // Trade.
    if let Some(tr) = &state.trade {
        let o = OBS_TRADE_OFFSET;
        out[o] = 1.0;
        for r in 0..NUM_RESOURCES {
            out[o + 1 + r] = tr.give[r] as f32 / 2.0;
            out[o + 6 + r] = tr.want[r] as f32 / 2.0;
        }
        out[o + 11 + rel(n, actor, tr.proposer) as usize] = 1.0;
        for q in 0..n {
            let r = rel(n, actor, q) as usize;
            let ro = o + 11 + MAX_PLAYERS + r * 5;
            let k = match tr.responses[q as usize] {
                Response::Pending => 0,
                Response::Accept => 1,
                Response::Reject => 2,
                Response::Counter { give, want } => {
                    let co = o + 11 + MAX_PLAYERS + MAX_PLAYERS * 5 + r * 10;
                    for x in 0..NUM_RESOURCES {
                        out[co + x] = give[x] as f32 / 2.0;
                        out[co + 5 + x] = want[x] as f32 / 2.0;
                    }
                    3
                }
                Response::NotInvolved => 4,
            };
            out[ro + k] = 1.0;
        }
    }

    // Global.
    let o = OBS_GLOBAL_OFFSET;
    out[o + phase_index(&state.phase)] = 1.0;
    let mut i = o + NUM_PHASES;
    for r in 0..NUM_RESOURCES {
        out[i + r] = state.bank[r] as f32 / BANK_START as f32;
    }
    i += 5;
    for d in 0..NUM_DEV_TYPES {
        out[i + d] = state.dev_deck[d] as f32 / DEV_DECK[d] as f32;
    }
    i += 5;
    if let Some(d) = state.last_roll {
        out[i + (d[0] + d[1]) as usize - 2] = 1.0;
    }
    i += 11;
    out[i] = state.dice_rolled as u8 as f32;
    out[i + 1] = state.dev_played_this_turn as u8 as f32;
    out[i + 2] = state
        .config
        .max_trade_offers_per_turn
        .saturating_sub(state.trade_offers_this_turn) as f32
        / 3.0;
    out[i + 3] = (state.turn as f32 / 100.0).min(3.0);
    out[i + 4] = match state.phase {
        Phase::RoadBuilding { remaining } | Phase::YearOfPlenty { remaining } => remaining as f32 / 2.0,
        _ => 0.0,
    };
    out[i + 5] = state.config.vp_to_win as f32 / 10.0;
    out[i + 6 + (n as usize - 2)] = 1.0;
    debug_assert_eq!(i + 9, OBS_SIZE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GameConfig;

    #[test]
    fn action_space_layout() {
        assert_eq!(ACTION_SIZE, 419);
    }

    #[test]
    fn roundtrip_all_indices() {
        let s = GameState::new(GameConfig::default(), 3);
        for actor in 0..4u8 {
            for i in 0..ACTION_SIZE {
                if let Some(a) = index_to_action(&s, actor, i) {
                    assert_eq!(action_to_index(&s, actor, &a), Some(i), "{a:?}");
                }
            }
        }
    }
}
