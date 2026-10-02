//! Player actions.

use crate::types::Hand;
use serde::{Deserialize, Serialize};

/// Every move a player can make. Player ids inside actions are absolute seats.
///
/// Trades are always expressed from the *acting* player's point of view:
/// `give` is what the actor hands over, `want` is what the actor receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    RollDice,
    EndTurn,
    BuildSettlement {
        vertex: u8,
    },
    BuildCity {
        vertex: u8,
    },
    BuildRoad {
        edge: u8,
    },
    BuyDevCard,
    PlayKnight,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    PlayMonopoly,
    /// Pick a resource for Year of Plenty (once per card taken) or Monopoly.
    ChooseResource {
        resource: u8,
    },
    /// Move the robber to `hex` and steal from `victim` (required when any opponent can be robbed).
    MoveRobber {
        hex: u8,
        victim: Option<u8>,
    },
    /// Discard one card of the given resource (repeated until the discard obligation is met).
    Discard {
        resource: u8,
    },
    /// Trade with the bank or a harbor at the best rate available to the player.
    MaritimeTrade {
        give: u8,
        get: u8,
    },
    /// Propose a trade to all opponents (on your turn) or counter-offer the pending
    /// proposal (when responding).
    OfferTrade {
        give: Hand,
        want: Hand,
    },
    AcceptTrade,
    RejectTrade,
    /// Proposer finalizes the trade with a player who accepted or countered.
    ConfirmTrade {
        partner: u8,
    },
    /// Proposer withdraws the pending offer.
    CancelTrade,
}

/// Why an action was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionError {
    GameOver,
    NotYourTurn,
    WrongPhase,
    Illegal(&'static str),
}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionError::GameOver => write!(f, "game is over"),
            ActionError::NotYourTurn => write!(f, "it is not this player's decision"),
            ActionError::WrongPhase => write!(f, "action not allowed in the current phase"),
            ActionError::Illegal(why) => write!(f, "illegal action: {why}"),
        }
    }
}

impl std::error::Error for ActionError {}
