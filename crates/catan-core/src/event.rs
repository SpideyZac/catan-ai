//! Game event log, used by UIs for animation and history.
//!
//! Events are only recorded when [`crate::GameConfig::record_events`] is set, so
//! self-play training pays nothing for them.

use crate::types::Hand;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TradeResponseKind {
    Accept,
    Reject,
    /// Counter-offer expressed from the *proposer's* point of view.
    Counter {
        give: Hand,
        want: Hand,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    TurnStarted {
        player: u8,
        turn: u32,
    },
    DiceRolled {
        player: u8,
        dice: [u8; 2],
    },
    Produced {
        player: u8,
        resources: Hand,
    },
    SettlementBuilt {
        player: u8,
        vertex: u8,
    },
    CityBuilt {
        player: u8,
        vertex: u8,
    },
    RoadBuilt {
        player: u8,
        edge: u8,
    },
    /// `card` is only visible to the buyer; it is `None` in other players' views.
    DevCardBought {
        player: u8,
        card: Option<u8>,
    },
    DevCardPlayed {
        player: u8,
        card: u8,
    },
    RobberMoved {
        player: u8,
        hex: u8,
    },
    /// `resource` is only visible to the thief and the victim.
    Stolen {
        thief: u8,
        victim: u8,
        resource: Option<u8>,
    },
    Discarded {
        player: u8,
        resource: u8,
    },
    MonopolyTaken {
        player: u8,
        resource: u8,
        amount: u8,
    },
    YearOfPlentyTaken {
        player: u8,
        resource: u8,
    },
    MaritimeTraded {
        player: u8,
        give: Hand,
        get: Hand,
    },
    TradeOffered {
        proposer: u8,
        give: Hand,
        want: Hand,
    },
    TradeResponded {
        player: u8,
        response: TradeResponseKind,
    },
    TradeExecuted {
        proposer: u8,
        partner: u8,
        give: Hand,
        want: Hand,
    },
    TradeCancelled {
        proposer: u8,
    },
    LongestRoadChanged {
        player: Option<u8>,
        length: u8,
    },
    LargestArmyChanged {
        player: Option<u8>,
        knights: u8,
    },
    GameWon {
        player: u8,
    },
    GameTruncated,
}

impl Event {
    /// Copy of the event with information hidden from `viewer` removed.
    /// `viewer = None` means a spectator.
    pub fn redacted_for(&self, viewer: Option<u8>) -> Event {
        match *self {
            Event::DevCardBought { player, .. } if viewer != Some(player) => {
                Event::DevCardBought { player, card: None }
            }
            Event::Stolen { thief, victim, .. } if viewer != Some(thief) && viewer != Some(victim) => Event::Stolen {
                thief,
                victim,
                resource: None,
            },
            e => e,
        }
    }
}
