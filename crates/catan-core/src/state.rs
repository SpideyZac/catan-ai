//! Game state containers.

use crate::board::Board;
use crate::event::Event;
use crate::rng::Rng;
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Tunable rule/engine options.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GameConfig {
    /// Number of seats (2..=4).
    pub num_players: u8,
    /// Victory points needed to win.
    pub vp_to_win: u8,
    /// Players holding more than this many cards discard half on a 7.
    pub discard_limit: u8,
    /// Max player-to-player trade proposals per turn (0 disables domestic trading).
    pub max_trade_offers_per_turn: u8,
    /// Max cards on either side of a player-to-player trade.
    pub max_trade_cards: u8,
    /// Game ends without a winner once this many turns have been played (0 = unlimited).
    pub max_turns: u32,
    /// Use the fixed beginner layout instead of a random one.
    pub beginner_board: bool,
    /// Record an [`Event`] log (for UIs). Disable for fast self-play.
    pub record_events: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            num_players: 4,
            vp_to_win: 10,
            discard_limit: 7,
            max_trade_offers_per_turn: 3,
            max_trade_cards: 6,
            max_turns: 0,
            beginner_board: false,
            record_events: false,
        }
    }
}

/// Phase of the turn state machine. See `docs/ENGINE.md` for the transition diagram.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum Phase {
    /// Initial placement: current player places a settlement.
    SetupSettlement,
    /// Initial placement: current player places a road touching `vertex`.
    SetupRoad {
        vertex: u8,
    },
    /// Start of turn: roll the dice (or play a knight first).
    PreRoll,
    /// After a 7: every player with `discard_pending > 0` discards one card at a time.
    Discard,
    /// Current player must move the robber.
    MoveRobber,
    /// Main action phase: build, buy, trade, play cards, end turn.
    Main,
    /// Road Building card: place up to `remaining` free roads.
    RoadBuilding {
        remaining: u8,
    },
    /// Year of Plenty card: pick `remaining` resources from the bank.
    YearOfPlenty {
        remaining: u8,
    },
    /// Monopoly card: pick a resource to collect.
    Monopoly,
    /// Waiting for opponents to answer the pending trade offer.
    TradeResponse,
    /// All responses in; the proposer picks a partner or cancels.
    TradeConfirm,
    GameOver,
}

/// An opponent's answer to the pending trade offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response {
    Pending,
    Accept,
    Reject,
    /// Counter-offer, stored from the proposer's point of view
    /// (`give` = proposer gives, `want` = proposer receives).
    Counter {
        give: Hand,
        want: Hand,
    },
    /// Not part of this trade (the proposer itself, or an empty seat).
    NotInvolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TradeOffer {
    pub proposer: u8,
    /// What the proposer gives.
    pub give: Hand,
    /// What the proposer receives.
    pub want: Hand,
    pub responses: [Response; MAX_PLAYERS],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    pub resources: Hand,
    /// Playable (or VP) development cards held since before this turn.
    pub dev_cards: [u8; NUM_DEV_TYPES],
    /// Development cards bought this turn (not playable until next turn).
    pub new_dev_cards: [u8; NUM_DEV_TYPES],
    pub knights_played: u8,
    pub settlements: u64,
    pub cities: u64,
    pub roads: u128,
    /// Endpoints of this player's roads.
    pub road_vertices: u64,
    pub roads_left: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub longest_road: u8,
    /// Bitmask over [`PortKind::bit`].
    pub ports: u8,
    /// Cards still to discard in the current discard phase.
    pub discard_pending: u8,
    /// Public belief of this player's hand, as tracked by an observer that saw every
    /// public event (exact except for hidden robber steals).
    pub belief: [f32; NUM_RESOURCES],
}

impl PlayerState {
    pub fn new() -> Self {
        PlayerState {
            resources: EMPTY_HAND,
            dev_cards: [0; NUM_DEV_TYPES],
            new_dev_cards: [0; NUM_DEV_TYPES],
            knights_played: 0,
            settlements: 0,
            cities: 0,
            roads: 0,
            road_vertices: 0,
            roads_left: MAX_ROADS,
            settlements_left: MAX_SETTLEMENTS,
            cities_left: MAX_CITIES,
            longest_road: 0,
            ports: 0,
            discard_pending: 0,
            belief: [0.0; NUM_RESOURCES],
        }
    }

    #[inline]
    pub fn buildings(&self) -> u64 {
        self.settlements | self.cities
    }

    #[inline]
    pub fn num_resources(&self) -> u32 {
        hand_total(&self.resources)
    }

    #[inline]
    pub fn num_dev_cards(&self) -> u32 {
        self.dev_cards
            .iter()
            .chain(self.new_dev_cards.iter())
            .map(|&x| x as u32)
            .sum()
    }

    #[inline]
    pub fn vp_cards(&self) -> u8 {
        self.dev_cards[DevCard::VictoryPoint as usize] + self.new_dev_cards[DevCard::VictoryPoint as usize]
    }

    /// Trade ratio this player gets when giving resource `r` to the bank.
    #[inline]
    pub fn trade_ratio(&self, r: usize) -> u8 {
        if self.ports & (1 << r) != 0 {
            2
        } else if self.ports & PortKind::Generic.bit() != 0 {
            3
        } else {
            4
        }
    }
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete game state. Cheap to clone (no heap allocations besides the optional
/// event log), so search algorithms can copy it freely.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub config: GameConfig,
    pub board: Board,
    pub players: [PlayerState; MAX_PLAYERS],
    pub bank: Hand,
    /// Remaining development cards by type (drawn uniformly at random at purchase).
    pub dev_deck: [u8; NUM_DEV_TYPES],
    pub robber: u8,
    pub phase: Phase,
    pub current: u8,
    /// Number of completed turns since setup finished.
    pub turn: u32,
    /// Index into the snake-draft setup order.
    pub setup_step: u8,
    pub last_roll: Option<[u8; 2]>,
    pub dice_rolled: bool,
    pub dev_played_this_turn: bool,
    pub trade_offers_this_turn: u8,
    pub trade: Option<TradeOffer>,
    pub longest_road_owner: Option<u8>,
    pub largest_army_owner: Option<u8>,
    pub winner: Option<u8>,
    /// Owner of each vertex (`u8::MAX` for none).
    #[serde(with = "crate::serde_arr")]
    pub vertex_owner: [u8; NUM_VERTICES],
    /// Owner of each edge (`u8::MAX` for none).
    #[serde(with = "crate::serde_arr")]
    pub edge_owner: [u8; NUM_EDGES],
    /// Vertices where a settlement may not be built (occupied or adjacent to a building).
    pub blocked: u64,
    pub rng: Rng,
    /// Event log (only filled when `config.record_events`).
    #[serde(default)]
    pub events: Vec<Event>,
}

impl GameState {
    #[inline]
    pub fn n(&self) -> usize {
        self.config.num_players as usize
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.phase == Phase::GameOver
    }

    /// Victory points visible to everyone (excludes hidden VP cards).
    pub fn public_vp(&self, p: usize) -> u8 {
        let ps = &self.players[p];
        let mut vp = ps.settlements.count_ones() as u8 + 2 * ps.cities.count_ones() as u8;
        if self.longest_road_owner == Some(p as u8) {
            vp += 2;
        }
        if self.largest_army_owner == Some(p as u8) {
            vp += 2;
        }
        vp
    }

    /// Actual victory points including hidden VP cards.
    pub fn total_vp(&self, p: usize) -> u8 {
        self.public_vp(p) + self.players[p].vp_cards()
    }

    #[inline]
    pub(crate) fn log(&mut self, e: Event) {
        if self.config.record_events {
            self.events.push(e);
        }
    }

    /// Drain and return the recorded events.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    // ---- resource bookkeeping (keeps the public belief in sync) ----

    /// Publicly visible transfer of resources from the bank to `p`.
    #[inline]
    pub(crate) fn bank_to_player(&mut self, p: usize, h: &Hand) {
        for r in 0..NUM_RESOURCES {
            self.bank[r] -= h[r];
            self.players[p].resources[r] += h[r];
            self.players[p].belief[r] += h[r] as f32;
        }
    }

    /// Publicly visible transfer of resources from `p` to the bank.
    #[inline]
    pub(crate) fn player_to_bank(&mut self, p: usize, h: &Hand) {
        for r in 0..NUM_RESOURCES {
            self.players[p].resources[r] -= h[r];
            self.bank[r] += h[r];
        }
        self.belief_remove(p, h);
    }

    /// Publicly visible transfer between two players.
    #[inline]
    pub(crate) fn player_to_player(&mut self, from: usize, to: usize, h: &Hand) {
        for r in 0..NUM_RESOURCES {
            self.players[from].resources[r] -= h[r];
            self.players[to].resources[r] += h[r];
            self.players[to].belief[r] += h[r] as f32;
        }
        self.belief_remove(from, h);
    }

    /// Hidden transfer of one card (robber steal). Observers only learn the count changed.
    pub(crate) fn hidden_steal(&mut self, from: usize, to: usize, r: usize) {
        self.players[from].resources[r] -= 1;
        self.players[to].resources[r] += 1;
        let b = self.players[from].belief;
        let total: f32 = b.iter().sum();
        let probs: [f32; NUM_RESOURCES] = if total > 1e-6 {
            std::array::from_fn(|i| b[i] / total)
        } else {
            [0.2; NUM_RESOURCES]
        };
        for i in 0..NUM_RESOURCES {
            self.players[from].belief[i] = (self.players[from].belief[i] - probs[i]).max(0.0);
            self.players[to].belief[i] += probs[i];
        }
        self.normalize_belief(from);
        self.normalize_belief(to);
    }

    fn belief_remove(&mut self, p: usize, h: &Hand) {
        let mut inconsistent = false;
        for r in 0..NUM_RESOURCES {
            let b = &mut self.players[p].belief[r];
            *b -= h[r] as f32;
            if *b < -1e-4 {
                inconsistent = true;
            }
            if *b < 0.0 {
                *b = 0.0;
            }
        }
        if inconsistent {
            self.normalize_belief(p);
        }
    }

    /// Rescale a belief so it sums to the (public) card count.
    pub(crate) fn normalize_belief(&mut self, p: usize) {
        let actual = self.players[p].num_resources() as f32;
        let b = &mut self.players[p].belief;
        let total: f32 = b.iter().sum();
        if actual <= 0.0 {
            *b = [0.0; NUM_RESOURCES];
        } else if total > 1e-6 {
            let s = actual / total;
            for x in b.iter_mut() {
                *x *= s;
            }
        } else {
            *b = [actual / NUM_RESOURCES as f32; NUM_RESOURCES];
        }
    }
}
