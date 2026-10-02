//! Built-in scripted bots. They serve as baselines for evaluation, as opponents
//! during early training, and as the "easy"/"medium" AI levels in the web app.

use crate::action::Action;
use crate::rng::Rng;
use crate::state::{GameState, Phase, Response};
use crate::topology::{topo, NONE};
use crate::types::*;

pub trait Bot {
    /// Choose an action for `player`, who must be one of `state.actors()`.
    fn choose(&mut self, state: &GameState, player: u8) -> Action;
}

/// Coarse category of an action, used by [`RandomBot`] to avoid being swamped by
/// the many trade-offer templates.
fn kind(a: &Action) -> u8 {
    match a {
        Action::OfferTrade { .. } => 1,
        Action::MaritimeTrade { .. } => 2,
        Action::BuildRoad { .. } => 3,
        Action::BuildSettlement { .. } => 4,
        Action::BuildCity { .. } => 5,
        Action::MoveRobber { .. } => 6,
        _ => 0,
    }
}

/// Uniformly random over action *kinds*, then uniformly within the kind.
pub struct RandomBot {
    rng: Rng,
    buf: Vec<Action>,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot {
            rng: Rng::new(seed),
            buf: Vec::with_capacity(256),
        }
    }
}

impl Bot for RandomBot {
    fn choose(&mut self, state: &GameState, player: u8) -> Action {
        state.legal_actions(player, &mut self.buf);
        assert!(
            !self.buf.is_empty(),
            "no legal actions for player {player} in {:?}",
            state.phase
        );
        let mut kinds = [0u32; 7];
        for a in &self.buf {
            kinds[kind(a) as usize] = 1;
        }
        let k = self.rng.weighted_index(&kinds).unwrap() as u8;
        let n = self.buf.iter().filter(|a| kind(a) == k).count();
        let pick = self.rng.below(n as u32) as usize;
        *self.buf.iter().filter(|a| kind(a) == k).nth(pick).unwrap()
    }
}

const RES_WEIGHT: [f32; NUM_RESOURCES] = [1.0, 1.0, 0.8, 1.1, 1.05];

/// A reasonable rule-based player: values spots by pips and diversity, builds
/// greedily toward a target, trades surplus for missing cards (with the bank or with
/// players), robs the leader, and accepts trades that help it without feeding a leader.
pub struct HeuristicBot {
    rng: Rng,
    buf: Vec<Action>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    City,
    Settlement,
    Road,
    DevCard,
}

impl Target {
    fn cost(self) -> Hand {
        match self {
            Target::City => CITY_COST,
            Target::Settlement => SETTLEMENT_COST,
            Target::Road => ROAD_COST,
            Target::DevCard => DEV_CARD_COST,
        }
    }
}

fn missing(have: &Hand, need: &Hand) -> u32 {
    (0..NUM_RESOURCES).map(|r| need[r].saturating_sub(have[r]) as u32).sum()
}

impl HeuristicBot {
    pub fn new(seed: u64) -> Self {
        HeuristicBot {
            rng: Rng::new(seed),
            buf: Vec::with_capacity(256),
        }
    }

    /// Per-resource production (pips) of player `p`.
    fn production(state: &GameState, p: usize) -> [f32; NUM_RESOURCES] {
        let t = topo();
        let ps = &state.players[p];
        let mut prod = [0f32; NUM_RESOURCES];
        for h in 0..NUM_HEXES {
            let Some(r) = state.board.hex_resource(h) else { continue };
            let pip = pips(state.board.numbers[h]) as f32;
            for &v in &t.hex_vertices[h] {
                let bit = 1u64 << v;
                if ps.settlements & bit != 0 {
                    prod[r as usize] += pip;
                } else if ps.cities & bit != 0 {
                    prod[r as usize] += 2.0 * pip;
                }
            }
        }
        prod
    }

    fn vertex_value(state: &GameState, prod: &[f32; NUM_RESOURCES], v: u8) -> f32 {
        let t = topo();
        let mut val = 0.0;
        let mut seen = [false; NUM_RESOURCES];
        for &h in &t.vertex_hexes[v as usize] {
            if h == NONE {
                continue;
            }
            let Some(r) = state.board.hex_resource(h as usize) else {
                continue;
            };
            let mut pip = pips(state.board.numbers[h as usize]) as f32;
            if h == state.robber {
                pip *= 0.5;
            }
            val += pip * RES_WEIGHT[r as usize];
            if prod[r as usize] == 0.0 && !seen[r as usize] {
                val += 1.5;
                seen[r as usize] = true;
            }
        }
        match state.board.vertex_port[v as usize] {
            Some(PortKind::Generic) => val += 1.0,
            Some(port) => val += 0.5 + 0.25 * prod[port as usize],
            None => {}
        }
        val
    }

    /// Value of building road `e` for player `p`: the best open settlement spot it leads
    /// toward (within two steps), discounted by distance.
    fn road_value(state: &GameState, p: usize, prod: &[f32; NUM_RESOURCES], e: u8) -> f32 {
        let t = topo();
        let mut best = 0.0f32;
        let all_roads = state.players[..state.n()].iter().fold(0u128, |a, ps| a | ps.roads);
        for &v in &t.edge_vertices[e as usize] {
            if state.blocked & (1u64 << v) == 0 {
                best = best.max(Self::vertex_value(state, prod, v));
            }
            if state.vertex_owner[v as usize] != NONE && state.vertex_owner[v as usize] as usize != p {
                continue;
            }
            for &e2 in &t.vertex_edges[v as usize] {
                if e2 == NONE || e2 == e || all_roads & (1u128 << e2) != 0 {
                    continue;
                }
                let u = t.edge_other(e2, v);
                if state.blocked & (1u64 << u) == 0 {
                    best = best.max(0.6 * Self::vertex_value(state, prod, u));
                }
            }
        }
        best
    }

    fn pick_target(state: &GameState, p: usize) -> Target {
        let ps = &state.players[p];
        let have = &ps.resources;
        let can_city = ps.settlements != 0 && ps.cities_left > 0;
        let can_settle = ps.settlements_left > 0 && state.settlement_spots(p) != 0;
        let city_gap = if can_city { missing(have, &CITY_COST) } else { u32::MAX };
        let settle_gap = if can_settle {
            missing(have, &SETTLEMENT_COST)
        } else {
            u32::MAX
        };
        if city_gap == u32::MAX && settle_gap == u32::MAX {
            if ps.roads_left > 0 && ps.settlements_left > 0 && state.road_spots(p) != 0 {
                return Target::Road;
            }
            return Target::DevCard;
        }
        if city_gap <= settle_gap {
            Target::City
        } else {
            Target::Settlement
        }
    }

    fn best_by<F: Fn(&Action) -> f32>(&self, pred: impl Fn(&Action) -> bool, score: F) -> Option<Action> {
        let mut best: Option<(f32, Action)> = None;
        for a in self.buf.iter().filter(|a| pred(a)) {
            let s = score(a);
            if best.is_none_or(|(b, _)| s > b) {
                best = Some((s, *a));
            }
        }
        best.map(|(_, a)| a)
    }

    fn has(&self, a: &Action) -> bool {
        self.buf.contains(a)
    }

    fn robber_choice(&self, state: &GameState, p: usize) -> Option<Action> {
        let t = topo();
        self.best_by(
            |a| matches!(a, Action::MoveRobber { .. }),
            |a| {
                let Action::MoveRobber { hex, victim } = *a else {
                    unreachable!()
                };
                let h = hex as usize;
                let pip = pips(state.board.numbers[h]) as f32;
                let mut score = 0.0;
                for &v in &t.hex_vertices[h] {
                    let o = state.vertex_owner[v as usize];
                    if o == NONE {
                        continue;
                    }
                    let mult = if state.players[o as usize].cities & (1u64 << v) != 0 {
                        2.0
                    } else {
                        1.0
                    };
                    if o as usize == p {
                        score -= 3.0 * pip * mult;
                    } else {
                        score += pip * mult * (1.0 + state.public_vp(o as usize) as f32 / 5.0);
                    }
                }
                if let Some(vic) = victim {
                    score += 0.5 * state.players[vic as usize].num_resources().min(8) as f32
                        + state.public_vp(vic as usize) as f32;
                }
                score
            },
        )
    }

    fn discard_choice(&self, state: &GameState, p: usize) -> Option<Action> {
        let ps = &state.players[p];
        let need = Self::pick_target(state, p).cost();
        self.best_by(
            |a| matches!(a, Action::Discard { .. }),
            |a| {
                let Action::Discard { resource } = *a else {
                    unreachable!()
                };
                let r = resource as usize;
                let surplus = ps.resources[r] as f32 - need[r] as f32;
                surplus + 0.01 * r as f32
            },
        )
    }

    /// Would `p` like to receive `get` in exchange for `give`?
    fn trade_helps(state: &GameState, p: usize, give: &Hand, get: &Hand) -> bool {
        let ps = &state.players[p];
        if !hand_covers(&ps.resources, give) {
            return false;
        }
        let target = Self::pick_target(state, p).cost();
        let mut after = ps.resources;
        hand_sub(&mut after, give);
        hand_add(&mut after, get);
        let before_gap = missing(&ps.resources, &target);
        let after_gap = missing(&after, &target);
        let give_n = hand_total(give);
        let get_n = hand_total(get);
        after_gap < before_gap || (after_gap == before_gap && get_n > give_n)
    }

    fn main_choice(&mut self, state: &GameState, p: usize) -> Action {
        let ps = &state.players[p];
        let prod = Self::production(state, p);

        if let Some(a) = self.best_by(
            |a| matches!(a, Action::BuildCity { .. }),
            |a| {
                let Action::BuildCity { vertex } = *a else {
                    unreachable!()
                };
                Self::vertex_value(state, &[1.0; 5], vertex)
            },
        ) {
            return a;
        }
        if let Some(a) = self.best_by(
            |a| matches!(a, Action::BuildSettlement { .. }),
            |a| {
                let Action::BuildSettlement { vertex } = *a else {
                    unreachable!()
                };
                Self::vertex_value(state, &prod, vertex)
            },
        ) {
            return a;
        }
        // Knight if the robber sits on us.
        if self.has(&Action::PlayKnight) && self.robber_hurts(state, p) {
            return Action::PlayKnight;
        }
        if self.has(&Action::PlayYearOfPlenty) {
            return Action::PlayYearOfPlenty;
        }
        if self.has(&Action::PlayMonopoly) {
            let r = Self::monopoly_resource(state, p);
            let haul: f32 = (0..state.n())
                .filter(|&q| q != p)
                .map(|q| state.players[q].belief[r])
                .sum();
            if haul >= 3.0 {
                return Action::PlayMonopoly;
            }
        }
        if self.has(&Action::PlayRoadBuilding) {
            return Action::PlayRoadBuilding;
        }

        let target = Self::pick_target(state, p);
        // Roads: when expansion is the bottleneck, or to grab longest road.
        if target == Target::Road || state.settlement_spots(p) == 0 {
            if let Some(a) = self.best_by(
                |a| matches!(a, Action::BuildRoad { .. }),
                |a| {
                    let Action::BuildRoad { edge } = *a else { unreachable!() };
                    Self::road_value(state, p, &prod, edge)
                },
            ) {
                let Action::BuildRoad { edge } = a else { unreachable!() };
                if Self::road_value(state, p, &prod, edge) > 0.0 || ps.roads_left > 10 {
                    return a;
                }
            }
        }
        if self.has(&Action::BuyDevCard) && (target == Target::DevCard || missing(&ps.resources, &target.cost()) >= 3) {
            return Action::BuyDevCard;
        }

        // Trading toward the target.
        let need = target.cost();
        let gap = missing(&ps.resources, &need);
        if gap > 0 && gap <= 2 {
            let want = (0..NUM_RESOURCES).find(|&r| ps.resources[r] < need[r]).unwrap();
            // Bank / harbor first.
            for give in 0..NUM_RESOURCES {
                let surplus = ps.resources[give].saturating_sub(need[give]);
                if give != want && surplus >= ps.trade_ratio(give) {
                    let a = Action::MaritimeTrade {
                        give: give as u8,
                        get: want as u8,
                    };
                    if self.has(&a) {
                        return a;
                    }
                }
            }
            // Then a 1:1 offer to the table, giving our largest surplus.
            let mut surplus_res: Vec<usize> = (0..NUM_RESOURCES)
                .filter(|&r| r != want && ps.resources[r] > need[r])
                .collect();
            surplus_res.sort_by_key(|&r| std::cmp::Reverse(ps.resources[r] - need[r]));
            for give in surplus_res {
                let mut g = EMPTY_HAND;
                g[give] = 1;
                let mut w = EMPTY_HAND;
                w[want] = 1;
                let a = Action::OfferTrade { give: g, want: w };
                if self.has(&a) && self.rng.below(3) != 0 {
                    return a;
                }
            }
        }
        Action::EndTurn
    }

    /// Propose 1-for-1 instead: one of our surplus cards for one card the proposer offered
    /// that we actually need.
    fn counter_offer(&mut self, state: &GameState, p: usize, tr: &crate::state::TradeOffer) -> Option<Action> {
        let ps = &state.players[p];
        let need = Self::pick_target(state, p).cost();
        let want = (0..NUM_RESOURCES).find(|&r| tr.give[r] > 0 && ps.resources[r] < need[r])?;
        let give = (0..NUM_RESOURCES)
            .filter(|&r| r != want && ps.resources[r] > need[r])
            .max_by_key(|&r| ps.resources[r] - need[r])?;
        let mut g = EMPTY_HAND;
        g[give] = 1;
        let mut w = EMPTY_HAND;
        w[want] = 1;
        let a = Action::OfferTrade { give: g, want: w };
        (self.has(&a) && self.rng.below(2) == 0).then_some(a)
    }

    fn robber_hurts(&self, state: &GameState, p: usize) -> bool {
        let t = topo();
        t.hex_vertices[state.robber as usize]
            .iter()
            .any(|&v| state.vertex_owner[v as usize] as usize == p)
    }

    fn monopoly_resource(state: &GameState, p: usize) -> usize {
        (0..NUM_RESOURCES)
            .max_by(|&a, &b| {
                let sa: f32 = (0..state.n())
                    .filter(|&q| q != p)
                    .map(|q| state.players[q].belief[a])
                    .sum();
                let sb: f32 = (0..state.n())
                    .filter(|&q| q != p)
                    .map(|q| state.players[q].belief[b])
                    .sum();
                sa.partial_cmp(&sb).unwrap()
            })
            .unwrap()
    }

    fn decide(&mut self, state: &GameState, player: u8) -> Option<Action> {
        let p = player as usize;
        match state.phase {
            Phase::SetupSettlement => {
                let prod = Self::production(state, p);
                self.best_by(
                    |a| matches!(a, Action::BuildSettlement { .. }),
                    |a| {
                        let Action::BuildSettlement { vertex } = *a else {
                            unreachable!()
                        };
                        Self::vertex_value(state, &prod, vertex)
                    },
                )
            }
            Phase::SetupRoad { .. } | Phase::RoadBuilding { .. } => {
                let prod = Self::production(state, p);
                self.best_by(
                    |a| matches!(a, Action::BuildRoad { .. }),
                    |a| {
                        let Action::BuildRoad { edge } = *a else { unreachable!() };
                        Self::road_value(state, p, &prod, edge)
                    },
                )
            }
            Phase::PreRoll => {
                if self.has(&Action::PlayKnight) && self.robber_hurts(state, p) {
                    Some(Action::PlayKnight)
                } else {
                    Some(Action::RollDice)
                }
            }
            Phase::Discard => self.discard_choice(state, p),
            Phase::MoveRobber => self.robber_choice(state, p),
            Phase::Main => Some(self.main_choice(state, p)),
            Phase::YearOfPlenty { .. } => {
                let need = Self::pick_target(state, p).cost();
                let ps = &state.players[p];
                let r = (0..NUM_RESOURCES)
                    .find(|&r| ps.resources[r] < need[r] && state.bank[r] > 0)
                    .or_else(|| [3usize, 4, 0, 1, 2].into_iter().find(|&r| state.bank[r] > 0))?;
                Some(Action::ChooseResource { resource: r as u8 })
            }
            Phase::Monopoly => Some(Action::ChooseResource {
                resource: Self::monopoly_resource(state, p) as u8,
            }),
            Phase::TradeResponse => {
                let tr = state.trade.as_ref()?;
                if tr.proposer == player {
                    return Some(Action::CancelTrade);
                }
                let leader_threat = state.public_vp(tr.proposer as usize) + 2 >= state.config.vp_to_win;
                // We give what the proposer wants and get what the proposer gives.
                if !leader_threat && self.has(&Action::AcceptTrade) && Self::trade_helps(state, p, &tr.want, &tr.give) {
                    return Some(Action::AcceptTrade);
                }
                if !leader_threat {
                    if let Some(counter) = self.counter_offer(state, p, tr) {
                        return Some(counter);
                    }
                }
                Some(Action::RejectTrade)
            }
            Phase::TradeConfirm => {
                let tr = state.trade.as_ref()?;
                let mut best: Option<(u8, Action)> = None;
                for q in 0..state.n() {
                    let a = Action::ConfirmTrade { partner: q as u8 };
                    if !self.has(&a) {
                        continue;
                    }
                    let ok = match tr.responses[q] {
                        Response::Accept => true,
                        Response::Counter { give, want } => Self::trade_helps(state, p, &give, &want),
                        _ => false,
                    };
                    let vp = state.public_vp(q);
                    if ok && best.is_none_or(|(b, _)| vp < b) {
                        best = Some((vp, a));
                    }
                }
                Some(best.map(|(_, a)| a).unwrap_or(Action::CancelTrade))
            }
            Phase::GameOver => None,
        }
    }
}

impl Bot for HeuristicBot {
    fn choose(&mut self, state: &GameState, player: u8) -> Action {
        state.legal_actions(player, &mut self.buf);
        assert!(
            !self.buf.is_empty(),
            "no legal actions for player {player} in {:?}",
            state.phase
        );
        match self.decide(state, player) {
            Some(a) if self.buf.contains(&a) => a,
            _ => {
                // Fallback: something safe and legal.
                for safe in [
                    Action::EndTurn,
                    Action::RollDice,
                    Action::RejectTrade,
                    Action::CancelTrade,
                ] {
                    if self.buf.contains(&safe) {
                        return safe;
                    }
                }
                self.buf[self.rng.below(self.buf.len() as u32) as usize]
            }
        }
    }
}

/// Play a full game with the given bots (one per seat). Returns the final state.
pub fn play_game(state: &mut GameState, bots: &mut [Box<dyn Bot>]) {
    while let Some(p) = state.next_actor() {
        let a = bots[p as usize].choose(state, p);
        state.apply_unchecked(p, a);
    }
}
