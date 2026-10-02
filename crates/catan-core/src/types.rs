//! Core value types and game constants.

use serde::{Deserialize, Serialize};

/// Maximum number of seats supported by the engine.
pub const MAX_PLAYERS: usize = 4;
/// Number of distinct resource types.
pub const NUM_RESOURCES: usize = 5;
/// Number of distinct development card types.
pub const NUM_DEV_TYPES: usize = 5;

pub const NUM_HEXES: usize = 19;
pub const NUM_VERTICES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_PORTS: usize = 9;

/// Piece limits per player.
pub const MAX_ROADS: u8 = 15;
pub const MAX_SETTLEMENTS: u8 = 5;
pub const MAX_CITIES: u8 = 4;

/// Bank starts with 19 of each resource.
pub const BANK_START: u8 = 19;

/// A resource multiset, indexed by [`Resource`] as `usize`.
pub type Hand = [u8; NUM_RESOURCES];

pub const EMPTY_HAND: Hand = [0; NUM_RESOURCES];
pub const ROAD_COST: Hand = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: Hand = [1, 1, 1, 1, 0];
pub const CITY_COST: Hand = [0, 0, 0, 2, 3];
pub const DEV_CARD_COST: Hand = [0, 0, 1, 1, 1];

/// Starting composition of the development deck, indexed by [`DevCard`].
pub const DEV_DECK: [u8; NUM_DEV_TYPES] = [14, 5, 2, 2, 2];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Resource {
    Wood = 0,
    Brick = 1,
    Sheep = 2,
    Wheat = 3,
    Ore = 4,
}

impl Resource {
    pub const ALL: [Resource; NUM_RESOURCES] = [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
        Resource::Ore,
    ];

    #[inline]
    pub fn from_index(i: usize) -> Resource {
        Self::ALL[i]
    }

    pub fn name(self) -> &'static str {
        match self {
            Resource::Wood => "wood",
            Resource::Brick => "brick",
            Resource::Sheep => "sheep",
            Resource::Wheat => "wheat",
            Resource::Ore => "ore",
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevCard {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

impl DevCard {
    pub const ALL: [DevCard; NUM_DEV_TYPES] = [
        DevCard::Knight,
        DevCard::VictoryPoint,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DevCard::Knight => "knight",
            DevCard::VictoryPoint => "victory_point",
            DevCard::RoadBuilding => "road_building",
            DevCard::YearOfPlenty => "year_of_plenty",
            DevCard::Monopoly => "monopoly",
        }
    }
}

/// Terrain of a land hex. Every terrain except the desert produces one resource.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Terrain {
    Forest = 0,
    Hills = 1,
    Pasture = 2,
    Fields = 3,
    Mountains = 4,
    Desert = 5,
}

impl Terrain {
    #[inline]
    pub fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Forest => Some(Resource::Wood),
            Terrain::Hills => Some(Resource::Brick),
            Terrain::Pasture => Some(Resource::Sheep),
            Terrain::Fields => Some(Resource::Wheat),
            Terrain::Mountains => Some(Resource::Ore),
            Terrain::Desert => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Terrain::Forest => "forest",
            Terrain::Hills => "hills",
            Terrain::Pasture => "pasture",
            Terrain::Fields => "fields",
            Terrain::Mountains => "mountains",
            Terrain::Desert => "desert",
        }
    }
}

/// Standard terrain distribution for the base game.
pub const TERRAIN_COUNTS: [(Terrain, u8); 6] = [
    (Terrain::Forest, 4),
    (Terrain::Hills, 3),
    (Terrain::Pasture, 4),
    (Terrain::Fields, 4),
    (Terrain::Mountains, 3),
    (Terrain::Desert, 1),
];

/// Standard number tokens for the base game.
pub const NUMBER_TOKENS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

/// Harbor kind. `Generic` trades 3:1, the others 2:1 for one resource.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortKind {
    Wood = 0,
    Brick = 1,
    Sheep = 2,
    Wheat = 3,
    Ore = 4,
    Generic = 5,
}

impl PortKind {
    pub const STANDARD_SET: [PortKind; NUM_PORTS] = [
        PortKind::Generic,
        PortKind::Generic,
        PortKind::Generic,
        PortKind::Generic,
        PortKind::Wood,
        PortKind::Brick,
        PortKind::Sheep,
        PortKind::Wheat,
        PortKind::Ore,
    ];

    pub fn name(self) -> &'static str {
        match self {
            PortKind::Generic => "generic",
            PortKind::Wood => "wood",
            PortKind::Brick => "brick",
            PortKind::Sheep => "sheep",
            PortKind::Wheat => "wheat",
            PortKind::Ore => "ore",
        }
    }

    /// Bit used in a player's port mask.
    #[inline]
    pub fn bit(self) -> u8 {
        1 << (self as u8)
    }
}

/// Number of ways to roll `n` with two dice (the "pips" on a number token).
#[inline]
pub const fn pips(n: u8) -> u8 {
    match n {
        2 | 12 => 1,
        3 | 11 => 2,
        4 | 10 => 3,
        5 | 9 => 4,
        6 | 8 => 5,
        _ => 0,
    }
}

#[inline]
pub fn hand_total(h: &Hand) -> u32 {
    h.iter().map(|&x| x as u32).sum()
}

#[inline]
pub fn hand_covers(have: &Hand, need: &Hand) -> bool {
    have.iter().zip(need.iter()).all(|(a, b)| a >= b)
}

#[inline]
pub fn hand_sub(have: &mut Hand, cost: &Hand) {
    for i in 0..NUM_RESOURCES {
        have[i] -= cost[i];
    }
}

#[inline]
pub fn hand_add(have: &mut Hand, gain: &Hand) {
    for i in 0..NUM_RESOURCES {
        have[i] += gain[i];
    }
}
