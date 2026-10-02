//! Python bindings for the Catan engine, exposed as `catan_ai._engine`.
//!
//! Two entry points:
//! - [`Game`]: a single game with a JSON-based API for servers/UIs plus numpy
//!   encodings for agents.
//! - [`VecEnv`]: a batch of self-play games stepped in parallel (rayon, GIL released)
//!   for reinforcement-learning throughput.

use catan_core::bots::{Bot, HeuristicBot, RandomBot};
use catan_core::encode::{self, index_to_action, legal_mask, observe, ACTION_SIZE, OBS_SIZE};
use catan_core::{Action, Event, GameConfig, GameState, Rng, MAX_PLAYERS};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

fn value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_json<T: serde::Serialize>(v: &T) -> PyResult<String> {
    serde_json::to_string(v).map_err(value_err)
}

#[allow(clippy::too_many_arguments)]
fn make_config(
    num_players: u8,
    vp_to_win: u8,
    max_trade_offers_per_turn: u8,
    max_turns: u32,
    beginner_board: bool,
    record_events: bool,
    discard_limit: u8,
    max_trade_cards: u8,
) -> PyResult<GameConfig> {
    if !(2..=MAX_PLAYERS as u8).contains(&num_players) {
        return Err(PyValueError::new_err("num_players must be 2, 3 or 4"));
    }
    if vp_to_win < 3 {
        return Err(PyValueError::new_err("vp_to_win must be at least 3"));
    }
    Ok(GameConfig {
        num_players,
        vp_to_win,
        discard_limit,
        max_trade_offers_per_turn,
        max_trade_cards: max_trade_cards.max(1),
        max_turns,
        beginner_board,
        record_events,
    })
}

fn check_player(state: &GameState, player: u8) -> PyResult<()> {
    if (player as usize) < state.n() {
        Ok(())
    } else {
        Err(PyIndexError::new_err(format!("player {player} out of range")))
    }
}

/// Scripted bot selectable from Python.
enum ScriptedBot {
    Random(RandomBot),
    Heuristic(HeuristicBot),
}

impl ScriptedBot {
    fn new(kind: &str, seed: u64) -> PyResult<Self> {
        match kind {
            "random" => Ok(ScriptedBot::Random(RandomBot::new(seed))),
            "heuristic" => Ok(ScriptedBot::Heuristic(HeuristicBot::new(seed))),
            other => Err(PyValueError::new_err(format!(
                "unknown bot kind {other:?} (expected 'random' or 'heuristic')"
            ))),
        }
    }

    fn choose(&mut self, s: &GameState, p: u8) -> Action {
        match self {
            ScriptedBot::Random(b) => b.choose(s, p),
            ScriptedBot::Heuristic(b) => b.choose(s, p),
        }
    }
}

// ============================================================================ Game

/// A single Catan game.
///
/// Actions and views cross the boundary as JSON strings (see `docs/ENGINE.md` for the
/// schema); agent encodings cross as numpy arrays.
#[pyclass(module = "catan_ai._engine")]
#[derive(Clone)]
pub struct Game {
    state: GameState,
    scratch: Vec<Action>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (
        seed = None,
        num_players = 4,
        vp_to_win = 10,
        max_trade_offers_per_turn = 3,
        max_turns = 0,
        beginner_board = false,
        record_events = true,
        discard_limit = 7,
        max_trade_cards = 6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seed: Option<u64>,
        num_players: u8,
        vp_to_win: u8,
        max_trade_offers_per_turn: u8,
        max_turns: u32,
        beginner_board: bool,
        record_events: bool,
        discard_limit: u8,
        max_trade_cards: u8,
    ) -> PyResult<Self> {
        let config = make_config(
            num_players,
            vp_to_win,
            max_trade_offers_per_turn,
            max_turns,
            beginner_board,
            record_events,
            discard_limit,
            max_trade_cards,
        )?;
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });
        Ok(Game {
            state: GameState::new(config, seed),
            scratch: Vec::with_capacity(256),
        })
    }

    /// Restore a game from [`Game::to_json`] output.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let state: GameState = serde_json::from_str(s).map_err(value_err)?;
        Ok(Game {
            state,
            scratch: Vec::with_capacity(256),
        })
    }

    /// Full serialized state, including hidden information and the RNG. Server-side only.
    fn to_json(&self) -> PyResult<String> {
        to_json(&self.state)
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    #[getter]
    fn phase(&self) -> PyResult<String> {
        let v = serde_json::to_value(self.state.phase).map_err(value_err)?;
        Ok(v["name"].as_str().unwrap_or("unknown").to_string())
    }

    #[getter]
    fn current(&self) -> u8 {
        self.state.current
    }

    #[getter]
    fn turn(&self) -> u32 {
        self.state.turn
    }

    #[getter]
    fn num_players(&self) -> u8 {
        self.state.config.num_players
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.state.winner
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.state.is_over()
    }

    /// Seats that may act right now (several during discards and trade responses).
    #[getter]
    fn actors(&self) -> Vec<u32> {
        // (Vec<u8> would surface in Python as `bytes`.)
        let m = self.state.actors();
        (0..self.state.config.num_players as u32).filter(|p| m & (1 << p) != 0).collect()
    }

    /// Canonical next decision-maker for sequential drivers.
    #[getter]
    fn next_actor(&self) -> Option<u8> {
        self.state.next_actor()
    }

    fn public_vp(&self, player: u8) -> PyResult<u8> {
        check_player(&self.state, player)?;
        Ok(self.state.public_vp(player as usize))
    }

    fn total_vp(&self, player: u8) -> PyResult<u8> {
        check_player(&self.state, player)?;
        Ok(self.state.total_vp(player as usize))
    }

    /// JSON list of enumerable legal actions for `player`.
    fn legal_actions_json(&mut self, player: u8) -> PyResult<String> {
        check_player(&self.state, player)?;
        self.state.legal_actions(player, &mut self.scratch);
        to_json(&self.scratch)
    }

    /// Validate an action without applying it. Returns an error message or `None`.
    fn check_json(&self, player: u8, action: &str) -> PyResult<Option<String>> {
        check_player(&self.state, player)?;
        let a: Action = serde_json::from_str(action).map_err(value_err)?;
        Ok(self.state.check(player, &a).err().map(|e| e.to_string()))
    }

    /// Validate and apply a JSON action. Raises `ValueError` if illegal.
    fn apply_json(&mut self, player: u8, action: &str) -> PyResult<()> {
        check_player(&self.state, player)?;
        let a: Action = serde_json::from_str(action).map_err(value_err)?;
        self.state.apply(player, a).map_err(value_err)
    }

    /// Validate and apply an action from the discrete action space.
    fn apply_index(&mut self, player: u8, index: usize) -> PyResult<()> {
        check_player(&self.state, player)?;
        let a = index_to_action(&self.state, player, index)
            .ok_or_else(|| PyValueError::new_err(format!("invalid action index {index}")))?;
        self.state.apply(player, a).map_err(value_err)
    }

    fn action_to_index(&self, player: u8, action: &str) -> PyResult<Option<usize>> {
        check_player(&self.state, player)?;
        let a: Action = serde_json::from_str(action).map_err(value_err)?;
        Ok(encode::action_to_index(&self.state, player, &a))
    }

    fn index_to_action_json(&self, player: u8, index: usize) -> PyResult<Option<String>> {
        check_player(&self.state, player)?;
        index_to_action(&self.state, player, index)
            .map(|a| to_json(&a))
            .transpose()
    }

    /// Observation vector (float32, length `OBS_SIZE`) from `player`'s perspective.
    fn observe<'py>(&self, py: Python<'py>, player: u8) -> PyResult<Bound<'py, PyArray1<f32>>> {
        check_player(&self.state, player)?;
        let mut out = vec![0f32; OBS_SIZE];
        observe(&self.state, player, &mut out);
        Ok(Array1::from_vec(out).into_pyarray(py))
    }

    /// Legality mask (bool, length `ACTION_SIZE`) for `player`.
    fn legal_mask<'py>(&mut self, py: Python<'py>, player: u8) -> PyResult<Bound<'py, PyArray1<bool>>> {
        check_player(&self.state, player)?;
        let mut mask = vec![false; ACTION_SIZE];
        legal_mask(&self.state, player, &mut mask, &mut self.scratch);
        Ok(Array1::from_vec(mask).into_pyarray(py))
    }

    /// Information-filtered view for `viewer` (`None` = spectator) as JSON.
    #[pyo3(signature = (viewer = None))]
    fn view_json(&self, viewer: Option<u8>) -> PyResult<String> {
        to_json(&catan_core::view::game_view(&self.state, viewer))
    }

    /// Static board description (terrain, numbers, harbors, geometry) as JSON.
    fn board_json(&self) -> PyResult<String> {
        to_json(&catan_core::view::board_view(&self.state))
    }

    /// Drain the unredacted event log as JSON. Use `redact_events_json` before sending
    /// events to a particular client.
    fn drain_events_json(&mut self) -> PyResult<String> {
        to_json(&self.state.take_events())
    }

    #[staticmethod]
    #[pyo3(signature = (events, viewer = None))]
    fn redact_events_json(events: &str, viewer: Option<u8>) -> PyResult<String> {
        let evs: Vec<Event> = serde_json::from_str(events).map_err(value_err)?;
        let out: Vec<Event> = evs.iter().map(|e| e.redacted_for(viewer)).collect();
        to_json(&out)
    }

    /// Ask a scripted bot ("random" or "heuristic") for `player`'s move, as JSON.
    #[pyo3(signature = (player, kind = "heuristic", seed = 0))]
    fn bot_action_json(&self, player: u8, kind: &str, seed: u64) -> PyResult<String> {
        check_player(&self.state, player)?;
        if self.state.actors() & (1 << player) == 0 {
            return Err(PyValueError::new_err(format!("player {player} has no decision to make")));
        }
        let mut bot = ScriptedBot::new(kind, seed)?;
        to_json(&bot.choose(&self.state, player))
    }

    /// Force a dice result for the current player (testing / tutorials). Only valid in PreRoll.
    fn force_roll(&mut self, d1: u8, d2: u8) -> PyResult<()> {
        if self.state.phase != catan_core::Phase::PreRoll || !(1..=6).contains(&d1) || !(1..=6).contains(&d2) {
            return Err(PyValueError::new_err("can only force a valid roll during pre-roll"));
        }
        self.state.apply_roll([d1, d2]);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(players={}, turn={}, current={}, phase={:?}, winner={:?})",
            self.state.config.num_players, self.state.turn, self.state.current, self.state.phase, self.state.winner
        )
    }
}

// ============================================================================ VecEnv

struct Slot {
    state: GameState,
    /// Scripted bot per seat (`None` = controlled by Python policies).
    bots: Vec<Option<ScriptedBot>>,
    prev_vp: [u8; MAX_PLAYERS],
    scratch: Vec<Action>,
    rng: Rng,
}

struct StepOut {
    rewards: [f32; MAX_PLAYERS],
    done: bool,
    /// Winner seat, -1 for truncated, -2 when not done.
    winner: i8,
    turns: u32,
}

struct EnvSettings {
    config: GameConfig,
    bot_kind: Option<String>,
    num_bot_seats: usize,
    vp_reward_scale: f32,
    zero_sum: bool,
}

impl Slot {
    fn new(settings: &EnvSettings, seed: u64) -> Slot {
        let mut slot = Slot {
            state: GameState::new(settings.config.clone(), seed),
            bots: Vec::new(),
            prev_vp: [0; MAX_PLAYERS],
            scratch: Vec::with_capacity(256),
            rng: Rng::new(seed ^ 0x5EED_5EED_5EED_5EED),
        };
        slot.reset(settings);
        slot
    }

    fn reset(&mut self, settings: &EnvSettings) {
        let seed = self.rng.next_u64();
        self.state = GameState::new(settings.config.clone(), seed);
        let n = settings.config.num_players as usize;
        let mut seats: Vec<usize> = (0..n).collect();
        self.rng.shuffle(&mut seats);
        let bot_seats = &seats[..settings.num_bot_seats.min(n.saturating_sub(1))];
        self.bots = (0..n)
            .map(|s| match &settings.bot_kind {
                Some(kind) if bot_seats.contains(&s) => ScriptedBot::new(kind, self.rng.next_u64()).ok(),
                _ => None,
            })
            .collect();
        self.prev_vp = [0; MAX_PLAYERS];
        self.advance_bots(settings, &mut [0.0; MAX_PLAYERS]);
    }

    fn accumulate_vp(&mut self, settings: &EnvSettings, rewards: &mut [f32; MAX_PLAYERS]) {
        if settings.vp_reward_scale == 0.0 {
            return;
        }
        for p in 0..self.state.n() {
            let vp = self.state.total_vp(p);
            rewards[p] += settings.vp_reward_scale * (vp as f32 - self.prev_vp[p] as f32);
            self.prev_vp[p] = vp;
        }
    }

    /// Let scripted seats act until a Python-controlled seat must decide or the game ends.
    fn advance_bots(&mut self, settings: &EnvSettings, rewards: &mut [f32; MAX_PLAYERS]) {
        while let Some(p) = self.state.next_actor() {
            let Some(bot) = self.bots[p as usize].as_mut() else { break };
            let a = bot.choose(&self.state, p);
            self.state.apply_unchecked(p, a);
            self.accumulate_vp(settings, rewards);
        }
    }

    fn step(&mut self, settings: &EnvSettings, index: i64) -> Result<StepOut, String> {
        let mut rewards = [0f32; MAX_PLAYERS];
        let actor = self.state.next_actor().ok_or("stepping a finished game")?;
        let action = usize::try_from(index)
            .ok()
            .and_then(|i| index_to_action(&self.state, actor, i))
            .ok_or_else(|| format!("invalid action index {index}"))?;
        self.state
            .check(actor, &action)
            .map_err(|e| format!("action {index} ({action:?}) illegal for seat {actor}: {e}"))?;
        self.state.apply_unchecked(actor, action);
        self.accumulate_vp(settings, &mut rewards);
        self.advance_bots(settings, &mut rewards);

        if !self.state.is_over() {
            return Ok(StepOut {
                rewards,
                done: false,
                winner: -2,
                turns: self.state.turn,
            });
        }
        let n = self.state.n();
        let winner = self.state.winner;
        if let Some(w) = winner {
            for p in 0..n {
                rewards[p] += if p == w as usize {
                    1.0
                } else if settings.zero_sum {
                    -1.0 / (n as f32 - 1.0)
                } else {
                    0.0
                };
            }
        }
        let turns = self.state.turn;
        self.reset(settings);
        Ok(StepOut {
            rewards,
            done: true,
            winner: winner.map_or(-1, |w| w as i8),
            turns,
        })
    }
}

/// A batch of games for self-play training.
///
/// Each environment always has exactly one pending decision for a Python-controlled
/// seat. Seats can optionally be filled by scripted bots (`bot_kind`,
/// `num_bot_seats`), which act inside `step` without returning to Python. Finished
/// games are reset automatically.
///
/// Rewards are per *seat* (shape `[num_envs, 4]`) because a step may change the
/// outcome for seats other than the one that acted (e.g. the final move of the game).
#[pyclass(module = "catan_ai._engine")]
pub struct VecEnv {
    slots: Vec<Slot>,
    settings: EnvSettings,
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (
        num_envs,
        seed = 0,
        num_players = 4,
        vp_to_win = 10,
        max_trade_offers_per_turn = 3,
        max_turns = 300,
        bot_kind = None,
        num_bot_seats = 0,
        vp_reward_scale = 0.0,
        zero_sum = true,
        beginner_board = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        seed: u64,
        num_players: u8,
        vp_to_win: u8,
        max_trade_offers_per_turn: u8,
        max_turns: u32,
        bot_kind: Option<String>,
        num_bot_seats: usize,
        vp_reward_scale: f32,
        zero_sum: bool,
        beginner_board: bool,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        if let Some(k) = &bot_kind {
            ScriptedBot::new(k, 0)?;
        }
        if num_bot_seats >= num_players as usize {
            return Err(PyValueError::new_err("at least one seat must be policy-controlled"));
        }
        let config = make_config(
            num_players,
            vp_to_win,
            max_trade_offers_per_turn,
            max_turns,
            beginner_board,
            false,
            7,
            6,
        )?;
        let settings = EnvSettings {
            config,
            bot_kind,
            num_bot_seats,
            vp_reward_scale,
            zero_sum,
        };
        let mut seeder = Rng::new(seed);
        let seeds: Vec<u64> = (0..num_envs).map(|_| seeder.next_u64()).collect();
        let slots = seeds.par_iter().map(|&s| Slot::new(&settings, s)).collect();
        Ok(VecEnv { slots, settings })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    #[getter]
    fn num_players(&self) -> u8 {
        self.settings.config.num_players
    }

    /// Reset every game.
    fn reset(&mut self, py: Python<'_>) {
        let settings = &self.settings;
        let slots = &mut self.slots;
        py.detach(|| slots.par_iter_mut().for_each(|s| s.reset(settings)));
    }

    /// Encode the pending decision of every env.
    ///
    /// Returns `(obs[f32; N x OBS_SIZE], mask[bool; N x ACTION_SIZE], actor[i64; N])`.
    #[allow(clippy::type_complexity)]
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let n = self.slots.len();
        let slots = &mut self.slots;
        let (obs, mask, actors) = py.detach(|| {
            let mut obs = vec![0f32; n * OBS_SIZE];
            let mut mask = vec![false; n * ACTION_SIZE];
            let mut actors = vec![0i64; n];
            slots
                .par_iter_mut()
                .zip(obs.par_chunks_mut(OBS_SIZE))
                .zip(mask.par_chunks_mut(ACTION_SIZE))
                .zip(actors.par_iter_mut())
                .for_each(|(((slot, o), m), a)| {
                    let p = slot.state.next_actor().expect("env invariant: pending decision");
                    *a = p as i64;
                    observe(&slot.state, p, o);
                    legal_mask(&slot.state, p, m, &mut slot.scratch);
                });
            (obs, mask, actors)
        });
        let obs = Array2::from_shape_vec((n, OBS_SIZE), obs).map_err(value_err)?;
        let mask = Array2::from_shape_vec((n, ACTION_SIZE), mask).map_err(value_err)?;
        Ok((obs.into_pyarray(py), mask.into_pyarray(py), Array1::from_vec(actors).into_pyarray(py)))
    }

    /// Apply one action index per env.
    ///
    /// Returns `(rewards[f32; N x 4], done[bool; N], winner[i64; N], turns[i64; N])` where
    /// `winner` is the winning seat, -1 if the game was truncated, -2 if still running,
    /// and `turns` is the game length (for finished games).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let actions = actions.as_slice()?.to_vec();
        let n = self.slots.len();
        if actions.len() != n {
            return Err(PyValueError::new_err(format!("expected {n} actions, got {}", actions.len())));
        }
        let settings = &self.settings;
        let slots = &mut self.slots;
        let results: Vec<Result<StepOut, String>> = py.detach(|| {
            slots
                .par_iter_mut()
                .zip(actions.par_iter())
                .map(|(slot, &a)| slot.step(settings, a))
                .collect()
        });
        let mut rewards = Array2::<f32>::zeros((n, MAX_PLAYERS));
        let mut done = vec![false; n];
        let mut winner = vec![-2i64; n];
        let mut turns = vec![0i64; n];
        for (i, r) in results.into_iter().enumerate() {
            let out = r.map_err(|e| PyValueError::new_err(format!("env {i}: {e}")))?;
            for p in 0..MAX_PLAYERS {
                rewards[[i, p]] = out.rewards[p];
            }
            done[i] = out.done;
            winner[i] = out.winner as i64;
            turns[i] = out.turns as i64;
        }
        Ok((
            rewards.into_pyarray(py),
            Array1::from_vec(done).into_pyarray(py),
            Array1::from_vec(winner).into_pyarray(py),
            Array1::from_vec(turns).into_pyarray(py),
        ))
    }

    /// Observation of every env from every seat's perspective (`f32[N x P x OBS_SIZE]`).
    /// Used to bootstrap value estimates for seats that are not currently acting.
    fn observe_all_seats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let n = self.slots.len();
        let p = self.settings.config.num_players as usize;
        let slots = &self.slots;
        let obs = py.detach(|| {
            let mut obs = vec![0f32; n * p * OBS_SIZE];
            obs.par_chunks_mut(p * OBS_SIZE).zip(slots.par_iter()).for_each(|(chunk, slot)| {
                for (seat, o) in chunk.chunks_mut(OBS_SIZE).enumerate() {
                    observe(&slot.state, seat as u8, o);
                }
            });
            obs
        });
        let obs = Array3::from_shape_vec((n, p, OBS_SIZE), obs).map_err(value_err)?;
        Ok(obs.into_pyarray(py))
    }

    /// Seats controlled by Python in each env (`bool[N x 4]`); changes on reset.
    fn policy_seats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<bool>> {
        let n = self.slots.len();
        let mut out = Array2::<bool>::from_elem((n, MAX_PLAYERS), false);
        for (i, s) in self.slots.iter().enumerate() {
            for (p, b) in s.bots.iter().enumerate() {
                out[[i, p]] = b.is_none();
            }
        }
        out.into_pyarray(py)
    }

    /// Snapshot of one env as a standalone `Game` (for debugging / visualization).
    fn game(&self, index: usize) -> PyResult<Game> {
        let slot = self
            .slots
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("env index out of range"))?;
        Ok(Game {
            state: slot.state.clone(),
            scratch: Vec::new(),
        })
    }
}

// ============================================================================ module

/// Play `games` full games between scripted bots (one kind per seat) entirely in Rust.
/// Returns the win count per seat plus a final entry for truncated games.
#[pyfunction]
#[pyo3(signature = (kinds, games, seed = 0, max_turns = 500))]
fn arena(py: Python<'_>, kinds: Vec<String>, games: usize, seed: u64, max_turns: u32) -> PyResult<Vec<u64>> {
    let n = kinds.len();
    let config = make_config(n as u8, 10, 3, max_turns, false, false, 7, 6)?;
    for k in &kinds {
        ScriptedBot::new(k, 0)?;
    }
    let results: Vec<Option<u8>> = py.detach(|| {
        (0..games)
            .into_par_iter()
            .map(|g| {
                let gs = seed.wrapping_mul(1_000_003).wrapping_add(g as u64);
                let mut s = GameState::new(config.clone(), gs);
                // Rotate seats so no bot kind benefits from going first.
                let shift = g % n;
                let mut bots: Vec<ScriptedBot> = (0..n)
                    .map(|seat| ScriptedBot::new(&kinds[(seat + shift) % n], gs ^ seat as u64).unwrap())
                    .collect();
                while let Some(p) = s.next_actor() {
                    let a = bots[p as usize].choose(&s, p);
                    s.apply_unchecked(p, a);
                }
                s.winner.map(|w| ((w as usize + shift) % n) as u8)
            })
            .collect()
    });
    let mut wins = vec![0u64; n + 1];
    for r in results {
        wins[r.map_or(n, |w| w as usize)] += 1;
    }
    Ok(wins)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<VecEnv>()?;
    m.add_function(wrap_pyfunction!(arena, m)?)?;
    m.add("ACTION_SIZE", ACTION_SIZE)?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ENCODING_VERSION", encode::ENCODING_VERSION)?;
    m.add("NUM_TRADE_TEMPLATES", catan_core::trade::NUM_TRADE_TEMPLATES)?;
    m.add(
        "ACTION_OFFSETS",
        vec![
            ("end_turn", encode::A_END_TURN),
            ("roll", encode::A_ROLL),
            ("buy_dev", encode::A_BUY_DEV),
            ("play_knight", encode::A_PLAY_KNIGHT),
            ("play_road_building", encode::A_PLAY_ROAD_BUILDING),
            ("play_year_of_plenty", encode::A_PLAY_YOP),
            ("play_monopoly", encode::A_PLAY_MONOPOLY),
            ("choose_resource", encode::A_CHOOSE_RESOURCE),
            ("settlement", encode::A_SETTLEMENT),
            ("city", encode::A_CITY),
            ("road", encode::A_ROAD),
            ("robber", encode::A_ROBBER),
            ("discard", encode::A_DISCARD),
            ("maritime", encode::A_MARITIME),
            ("offer", encode::A_OFFER),
            ("accept", encode::A_ACCEPT),
            ("reject", encode::A_REJECT),
            ("confirm", encode::A_CONFIRM),
            ("cancel", encode::A_CANCEL),
        ],
    )?;
    m.add(
        "OBS_OFFSETS",
        vec![
            ("hex", encode::OBS_HEX_OFFSET, encode::HEX_FEATURES),
            ("vertex", encode::OBS_VERTEX_OFFSET, encode::VERTEX_FEATURES),
            ("edge", encode::OBS_EDGE_OFFSET, encode::EDGE_FEATURES),
            ("player", encode::OBS_PLAYER_OFFSET, encode::PLAYER_FEATURES),
            ("trade", encode::OBS_TRADE_OFFSET, encode::TRADE_FEATURES),
            ("global", encode::OBS_GLOBAL_OFFSET, encode::GLOBAL_FEATURES),
        ],
    )?;
    Ok(())
}
