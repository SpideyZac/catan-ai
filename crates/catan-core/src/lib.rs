//! # catan-core
//!
//! A fast, allocation-light Settlers of Catan rules engine (base game, 2-4 players)
//! with first-class support for player-to-player trading: proposals, accept/reject,
//! counter-offers and proposer confirmation.
//!
//! Entry points:
//! - [`GameState::new`] to start a game,
//! - [`GameState::legal_actions`] / [`GameState::actors`] to see who may do what,
//! - [`GameState::apply`] to play a move,
//! - [`encode`] for fixed-size action/observation encodings used by learning agents,
//! - [`bots`] for scripted baseline players.
//!
//! See `docs/ENGINE.md` for the full rules model and phase machine.

// Parallel fixed-size arrays are indexed by resource/player id throughout; index loops read clearer.
#![allow(clippy::needless_range_loop)]

pub mod action;
pub mod board;
pub mod bots;
pub mod encode;
pub mod event;
pub mod rng;
pub mod rules;
mod serde_arr;
pub mod state;
pub mod topology;
pub mod trade;
pub mod types;
pub mod view;

pub use action::{Action, ActionError};
pub use board::Board;
pub use event::Event;
pub use rng::Rng;
pub use state::{GameConfig, GameState, Phase, PlayerState, Response, TradeOffer};
pub use types::*;
