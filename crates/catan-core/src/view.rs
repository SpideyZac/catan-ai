//! Serializable, information-filtered views of the game for user interfaces.

use crate::state::{GameState, Phase, TradeOffer};
use crate::topology::{bits64, topo};
use crate::types::*;
use serde::Serialize;

#[derive(Serialize)]
pub struct HexView {
    pub id: u8,
    pub q: i8,
    pub r: i8,
    pub x: f32,
    pub y: f32,
    pub terrain: &'static str,
    pub number: u8,
    pub vertices: [u8; 6],
    pub edges: [u8; 6],
}

#[derive(Serialize)]
pub struct PortView {
    pub kind: &'static str,
    pub edge: u8,
    pub vertices: [u8; 2],
}

#[derive(Serialize)]
pub struct BoardView {
    pub hexes: Vec<HexView>,
    pub ports: Vec<PortView>,
    /// Vertex positions for a hex of circumradius 1.
    pub vertices: Vec<(f32, f32)>,
    pub edges: Vec<[u8; 2]>,
}

/// Static board description (terrain, numbers, harbors and geometry). Send once per game.
pub fn board_view(state: &GameState) -> BoardView {
    let t = topo();
    BoardView {
        hexes: (0..NUM_HEXES)
            .map(|h| HexView {
                id: h as u8,
                q: t.hex_coords[h].0,
                r: t.hex_coords[h].1,
                x: t.hex_center[h].0,
                y: t.hex_center[h].1,
                terrain: state.board.terrain[h].name(),
                number: state.board.numbers[h],
                vertices: t.hex_vertices[h],
                edges: t.hex_edges[h],
            })
            .collect(),
        ports: t
            .port_edges
            .iter()
            .enumerate()
            .map(|(i, &e)| PortView {
                kind: state.board.ports[i].name(),
                edge: e,
                vertices: t.edge_vertices[e as usize],
            })
            .collect(),
        vertices: t.vertex_pos.to_vec(),
        edges: t.edge_vertices.to_vec(),
    }
}

#[derive(Serialize)]
pub struct BuildingView {
    pub vertex: u8,
    pub player: u8,
    pub city: bool,
}

#[derive(Serialize)]
pub struct RoadView {
    pub edge: u8,
    pub player: u8,
}

#[derive(Serialize)]
pub struct PlayerView {
    pub seat: u8,
    /// Exact hand; only present for the viewer (or everyone once the game is over).
    pub resources: Option<Hand>,
    pub resource_count: u32,
    /// Dev cards held (playable + VP); only present for the viewer.
    pub dev_cards: Option<[u8; NUM_DEV_TYPES]>,
    /// Dev cards bought this turn; only present for the viewer.
    pub new_dev_cards: Option<[u8; NUM_DEV_TYPES]>,
    pub dev_card_count: u32,
    pub knights_played: u8,
    pub public_vp: u8,
    /// Including hidden VP cards; only present for the viewer or after the game ends.
    pub total_vp: Option<u8>,
    pub longest_road: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
    pub roads_left: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub ports: Vec<&'static str>,
    pub discard_pending: u8,
}

#[derive(Serialize)]
pub struct GameView {
    pub viewer: Option<u8>,
    pub num_players: u8,
    pub vp_to_win: u8,
    pub phase: Phase,
    pub current: u8,
    pub turn: u32,
    pub robber: u8,
    pub last_roll: Option<[u8; 2]>,
    pub dice_rolled: bool,
    pub dev_played_this_turn: bool,
    pub trade_offers_left: u8,
    pub bank: Hand,
    pub dev_deck_count: u32,
    pub buildings: Vec<BuildingView>,
    pub roads: Vec<RoadView>,
    pub players: Vec<PlayerView>,
    pub trade: Option<TradeOffer>,
    pub winner: Option<u8>,
    /// Seats that may act right now.
    pub actors: Vec<u8>,
}

/// Build the view of the game as seen by `viewer` (`None` = spectator).
pub fn game_view(state: &GameState, viewer: Option<u8>) -> GameView {
    let over = state.phase == Phase::GameOver;
    let n = state.n();
    let mut buildings = Vec::new();
    let mut roads = Vec::new();
    for p in 0..n {
        let ps = &state.players[p];
        for v in bits64(ps.settlements) {
            buildings.push(BuildingView {
                vertex: v,
                player: p as u8,
                city: false,
            });
        }
        for v in bits64(ps.cities) {
            buildings.push(BuildingView {
                vertex: v,
                player: p as u8,
                city: true,
            });
        }
    }
    for e in 0..NUM_EDGES {
        let o = state.edge_owner[e];
        if o != u8::MAX {
            roads.push(RoadView {
                edge: e as u8,
                player: o,
            });
        }
    }
    let players = (0..n)
        .map(|p| {
            let ps = &state.players[p];
            let me = viewer == Some(p as u8) || over;
            let ports = [
                PortKind::Wood,
                PortKind::Brick,
                PortKind::Sheep,
                PortKind::Wheat,
                PortKind::Ore,
                PortKind::Generic,
            ]
            .into_iter()
            .filter(|k| ps.ports & k.bit() != 0)
            .map(|k| k.name())
            .collect();
            PlayerView {
                seat: p as u8,
                resources: me.then_some(ps.resources),
                resource_count: ps.num_resources(),
                dev_cards: me.then_some(ps.dev_cards),
                new_dev_cards: me.then_some(ps.new_dev_cards),
                dev_card_count: ps.num_dev_cards(),
                knights_played: ps.knights_played,
                public_vp: state.public_vp(p),
                total_vp: me.then(|| state.total_vp(p)),
                longest_road: ps.longest_road,
                has_longest_road: state.longest_road_owner == Some(p as u8),
                has_largest_army: state.largest_army_owner == Some(p as u8),
                roads_left: ps.roads_left,
                settlements_left: ps.settlements_left,
                cities_left: ps.cities_left,
                ports,
                discard_pending: ps.discard_pending,
            }
        })
        .collect();
    let actors_mask = state.actors();
    GameView {
        viewer,
        num_players: state.config.num_players,
        vp_to_win: state.config.vp_to_win,
        phase: state.phase,
        current: state.current,
        turn: state.turn,
        robber: state.robber,
        last_roll: state.last_roll,
        dice_rolled: state.dice_rolled,
        dev_played_this_turn: state.dev_played_this_turn,
        trade_offers_left: state
            .config
            .max_trade_offers_per_turn
            .saturating_sub(state.trade_offers_this_turn),
        bank: state.bank,
        dev_deck_count: state.dev_deck.iter().map(|&x| x as u32).sum(),
        buildings,
        roads,
        players,
        trade: state.trade,
        winner: state.winner,
        actors: (0..n as u8).filter(|p| actors_mask & (1 << p) != 0).collect(),
    }
}
