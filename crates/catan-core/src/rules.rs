//! The rules: game construction, legal move generation, validation and state transitions.

use crate::action::{Action, ActionError};
use crate::board::Board;
use crate::event::{Event, TradeResponseKind};
use crate::rng::Rng;
use crate::state::*;
use crate::topology::{bits128, bits64, topo, NONE};
use crate::trade::{is_well_formed, templates};
use crate::types::*;

const ALL_VERTICES: u64 = (1u64 << NUM_VERTICES) - 1;

impl GameState {
    /// Create a new game. The board is generated from `seed` unless the config asks
    /// for the beginner layout.
    pub fn new(config: GameConfig, seed: u64) -> GameState {
        assert!(
            (2..=MAX_PLAYERS as u8).contains(&config.num_players),
            "num_players must be between 2 and {MAX_PLAYERS}"
        );
        let mut rng = Rng::new(seed);
        let board = if config.beginner_board {
            Board::beginner()
        } else {
            Board::random(&mut rng)
        };
        GameState::with_board(config, board, rng)
    }

    /// Create a new game on a specific board.
    pub fn with_board(config: GameConfig, board: Board, rng: Rng) -> GameState {
        let robber = board.desert();
        let mut s = GameState {
            config,
            board,
            players: std::array::from_fn(|_| PlayerState::new()),
            bank: [BANK_START; NUM_RESOURCES],
            dev_deck: DEV_DECK,
            robber,
            phase: Phase::SetupSettlement,
            current: 0,
            turn: 0,
            setup_step: 0,
            last_roll: None,
            dice_rolled: false,
            dev_played_this_turn: false,
            trade_offers_this_turn: 0,
            trade: None,
            longest_road_owner: None,
            largest_army_owner: None,
            winner: None,
            vertex_owner: [NONE; NUM_VERTICES],
            edge_owner: [NONE; NUM_EDGES],
            blocked: 0,
            rng,
            events: Vec::new(),
        };
        s.log(Event::TurnStarted { player: 0, turn: 0 });
        s
    }

    #[inline]
    fn setup_player(&self, step: u8) -> u8 {
        let n = self.config.num_players;
        if step < n {
            step
        } else {
            2 * n - 1 - step
        }
    }

    /// Bitmask of players allowed to make a decision right now.
    pub fn actors(&self) -> u8 {
        match self.phase {
            Phase::GameOver => 0,
            Phase::Discard => {
                let mut m = 0;
                for p in 0..self.n() {
                    if self.players[p].discard_pending > 0 {
                        m |= 1 << p;
                    }
                }
                m
            }
            Phase::TradeResponse => {
                let t = self.trade.as_ref().expect("trade response without offer");
                let mut m = 0;
                for p in 0..self.n() {
                    if t.responses[p] == Response::Pending {
                        m |= 1 << p;
                    }
                }
                m
            }
            _ => 1 << self.current,
        }
    }

    /// Canonical next decision-maker (for sequential drivers such as self-play):
    /// the first actor in seat order starting from the current player.
    pub fn next_actor(&self) -> Option<u8> {
        let m = self.actors();
        if m == 0 {
            return None;
        }
        let n = self.config.num_players;
        (0..n).map(|i| (self.current + i) % n).find(|&p| m & (1 << p) != 0)
    }

    // ------------------------------------------------------------------
    // Candidate sets
    // ------------------------------------------------------------------

    #[inline]
    fn all_roads(&self) -> u128 {
        self.players[..self.n()].iter().fold(0u128, |acc, p| acc | p.roads)
    }

    #[inline]
    fn opponent_buildings(&self, p: usize) -> u64 {
        let mut m = 0;
        for q in 0..self.n() {
            if q != p {
                m |= self.players[q].buildings();
            }
        }
        m
    }

    /// Vertices where `p` could place a settlement ignoring cost and piece limits.
    #[inline]
    pub fn settlement_spots(&self, p: usize) -> u64 {
        match self.phase {
            Phase::SetupSettlement => ALL_VERTICES & !self.blocked,
            _ => self.players[p].road_vertices & !self.blocked,
        }
    }

    /// Edges where `p` could build a road ignoring cost and piece limits.
    pub fn road_spots(&self, p: usize) -> u128 {
        let t = topo();
        let ps = &self.players[p];
        let frontier = ps.buildings() | (ps.road_vertices & !self.opponent_buildings(p));
        let mut edges = 0u128;
        for v in bits64(frontier) {
            edges |= t.vertex_edge_mask[v as usize];
        }
        edges & !self.all_roads()
    }

    /// Opponents of `p` that can be robbed if the robber is placed on `hex`.
    pub fn robbable(&self, p: usize, hex: usize) -> u8 {
        let t = topo();
        let mut m = 0u8;
        for &v in &t.hex_vertices[hex] {
            let o = self.vertex_owner[v as usize];
            if o != NONE && o as usize != p && self.players[o as usize].num_resources() > 0 {
                m |= 1 << o;
            }
        }
        m
    }

    #[inline]
    fn can_play_dev(&self, p: usize, card: DevCard) -> bool {
        p == self.current as usize && !self.dev_played_this_turn && self.players[p].dev_cards[card as usize] > 0
    }

    // ------------------------------------------------------------------
    // Legal move generation
    // ------------------------------------------------------------------

    /// Append every enumerable legal action for player `p` to `out`.
    ///
    /// Player-to-player offers are enumerated from the template catalogue
    /// ([`crate::trade::templates`]); arbitrary offers are still accepted by [`Self::apply`].
    pub fn legal_actions(&self, p: u8, out: &mut Vec<Action>) {
        out.clear();
        let pu = p as usize;
        if self.phase == Phase::GameOver || pu >= self.n() {
            return;
        }
        let t = topo();
        let ps = &self.players[pu];
        let is_current = p == self.current;
        match self.phase {
            Phase::GameOver => {}
            Phase::SetupSettlement => {
                if is_current {
                    for v in bits64(self.settlement_spots(pu)) {
                        out.push(Action::BuildSettlement { vertex: v });
                    }
                }
            }
            Phase::SetupRoad { vertex } => {
                if is_current {
                    let free = t.vertex_edge_mask[vertex as usize] & !self.all_roads();
                    for e in bits128(free) {
                        out.push(Action::BuildRoad { edge: e });
                    }
                }
            }
            Phase::PreRoll => {
                if is_current {
                    out.push(Action::RollDice);
                    if self.can_play_dev(pu, DevCard::Knight) {
                        out.push(Action::PlayKnight);
                    }
                }
            }
            Phase::Discard => {
                if ps.discard_pending > 0 {
                    for r in 0..NUM_RESOURCES {
                        if ps.resources[r] > 0 {
                            out.push(Action::Discard { resource: r as u8 });
                        }
                    }
                }
            }
            Phase::MoveRobber => {
                if is_current {
                    for h in 0..NUM_HEXES {
                        if h == self.robber as usize {
                            continue;
                        }
                        let victims = self.robbable(pu, h);
                        if victims == 0 {
                            out.push(Action::MoveRobber {
                                hex: h as u8,
                                victim: None,
                            });
                        } else {
                            for v in bits64(victims as u64) {
                                out.push(Action::MoveRobber {
                                    hex: h as u8,
                                    victim: Some(v),
                                });
                            }
                        }
                    }
                }
            }
            Phase::Main => {
                if is_current {
                    self.main_actions(pu, out);
                }
            }
            Phase::RoadBuilding { .. } => {
                if is_current {
                    for e in bits128(self.road_spots(pu)) {
                        out.push(Action::BuildRoad { edge: e });
                    }
                }
            }
            Phase::YearOfPlenty { .. } => {
                if is_current {
                    for r in 0..NUM_RESOURCES {
                        if self.bank[r] > 0 {
                            out.push(Action::ChooseResource { resource: r as u8 });
                        }
                    }
                }
            }
            Phase::Monopoly => {
                if is_current {
                    for r in 0..NUM_RESOURCES {
                        out.push(Action::ChooseResource { resource: r as u8 });
                    }
                }
            }
            Phase::TradeResponse => {
                let trade = self.trade.as_ref().unwrap();
                if trade.proposer == p {
                    out.push(Action::CancelTrade);
                } else if trade.responses[pu] == Response::Pending {
                    if hand_covers(&ps.resources, &trade.want) {
                        out.push(Action::AcceptTrade);
                    }
                    out.push(Action::RejectTrade);
                    let proposer_hand = &self.players[trade.proposer as usize].resources;
                    for (give, want) in templates() {
                        // Responder gives `give`, receives `want` (which the proposer must hold).
                        if hand_covers(&ps.resources, give)
                            && hand_covers(proposer_hand, want)
                            && !(give == &trade.want && want == &trade.give)
                        {
                            out.push(Action::OfferTrade {
                                give: *give,
                                want: *want,
                            });
                        }
                    }
                }
            }
            Phase::TradeConfirm => {
                let trade = self.trade.as_ref().unwrap();
                if trade.proposer == p {
                    for q in 0..self.n() {
                        if self.confirmable(trade, q) {
                            out.push(Action::ConfirmTrade { partner: q as u8 });
                        }
                    }
                    out.push(Action::CancelTrade);
                }
            }
        }
    }

    fn main_actions(&self, p: usize, out: &mut Vec<Action>) {
        let ps = &self.players[p];
        out.push(Action::EndTurn);

        if ps.settlements_left > 0 && hand_covers(&ps.resources, &SETTLEMENT_COST) {
            for v in bits64(self.settlement_spots(p)) {
                out.push(Action::BuildSettlement { vertex: v });
            }
        }
        if ps.cities_left > 0 && hand_covers(&ps.resources, &CITY_COST) {
            for v in bits64(ps.settlements) {
                out.push(Action::BuildCity { vertex: v });
            }
        }
        if ps.roads_left > 0 && hand_covers(&ps.resources, &ROAD_COST) {
            for e in bits128(self.road_spots(p)) {
                out.push(Action::BuildRoad { edge: e });
            }
        }
        if self.dev_deck.iter().any(|&c| c > 0) && hand_covers(&ps.resources, &DEV_CARD_COST) {
            out.push(Action::BuyDevCard);
        }
        if self.can_play_dev(p, DevCard::Knight) {
            out.push(Action::PlayKnight);
        }
        if self.can_play_dev(p, DevCard::RoadBuilding) && ps.roads_left > 0 && self.road_spots(p) != 0 {
            out.push(Action::PlayRoadBuilding);
        }
        if self.can_play_dev(p, DevCard::YearOfPlenty) && self.bank.iter().any(|&b| b > 0) {
            out.push(Action::PlayYearOfPlenty);
        }
        if self.can_play_dev(p, DevCard::Monopoly) {
            out.push(Action::PlayMonopoly);
        }
        for give in 0..NUM_RESOURCES {
            let ratio = ps.trade_ratio(give);
            if ps.resources[give] >= ratio {
                for get in 0..NUM_RESOURCES {
                    if get != give && self.bank[get] > 0 {
                        out.push(Action::MaritimeTrade {
                            give: give as u8,
                            get: get as u8,
                        });
                    }
                }
            }
        }
        if self.trade_offers_this_turn < self.config.max_trade_offers_per_turn {
            for (give, want) in templates() {
                if hand_covers(&ps.resources, give) {
                    out.push(Action::OfferTrade {
                        give: *give,
                        want: *want,
                    });
                }
            }
        }
    }

    /// Can the proposer finalize with `q` right now?
    fn confirmable(&self, trade: &TradeOffer, q: usize) -> bool {
        let (give, want) = match trade.responses[q] {
            Response::Accept => (trade.give, trade.want),
            Response::Counter { give, want } => (give, want),
            _ => return false,
        };
        hand_covers(&self.players[trade.proposer as usize].resources, &give)
            && hand_covers(&self.players[q].resources, &want)
    }

    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Check whether `p` may take `action` now, without mutating the state.
    pub fn check(&self, p: u8, action: &Action) -> Result<(), ActionError> {
        use ActionError::*;
        if self.phase == Phase::GameOver {
            return Err(GameOver);
        }
        let pu = p as usize;
        if pu >= self.n() {
            return Err(NotYourTurn);
        }
        let ps = &self.players[pu];
        let is_current = p == self.current;
        let t = topo();
        let need_current = |ok: bool| if ok { Ok(()) } else { Err(NotYourTurn) };

        match (*action, self.phase) {
            (Action::BuildSettlement { vertex }, Phase::SetupSettlement) => {
                need_current(is_current)?;
                if vertex as usize >= NUM_VERTICES || self.settlement_spots(pu) & (1u64 << vertex) == 0 {
                    return Err(Illegal("settlement spot unavailable"));
                }
                Ok(())
            }
            (Action::BuildRoad { edge }, Phase::SetupRoad { vertex }) => {
                need_current(is_current)?;
                if edge as usize >= NUM_EDGES
                    || t.vertex_edge_mask[vertex as usize] & (1u128 << edge) == 0
                    || self.edge_owner[edge as usize] != NONE
                {
                    return Err(Illegal("setup road must touch the new settlement"));
                }
                Ok(())
            }
            (Action::RollDice, Phase::PreRoll) => need_current(is_current),
            (Action::PlayKnight, Phase::PreRoll | Phase::Main) => {
                need_current(is_current)?;
                if !self.can_play_dev(pu, DevCard::Knight) {
                    return Err(Illegal("no playable knight"));
                }
                Ok(())
            }
            (Action::Discard { resource }, Phase::Discard) => {
                if ps.discard_pending == 0 {
                    return Err(NotYourTurn);
                }
                if resource as usize >= NUM_RESOURCES || ps.resources[resource as usize] == 0 {
                    return Err(Illegal("no such card to discard"));
                }
                Ok(())
            }
            (Action::MoveRobber { hex, victim }, Phase::MoveRobber) => {
                need_current(is_current)?;
                if hex as usize >= NUM_HEXES || hex == self.robber {
                    return Err(Illegal("robber must move to a different hex"));
                }
                let victims = self.robbable(pu, hex as usize);
                match victim {
                    None if victims == 0 => Ok(()),
                    None => Err(Illegal("must choose a player to rob")),
                    Some(v) if v < 8 && victims & (1 << v) != 0 => Ok(()),
                    Some(_) => Err(Illegal("that player cannot be robbed there")),
                }
            }
            (Action::EndTurn, Phase::Main) => need_current(is_current),
            (Action::BuildSettlement { vertex }, Phase::Main) => {
                need_current(is_current)?;
                if ps.settlements_left == 0 || !hand_covers(&ps.resources, &SETTLEMENT_COST) {
                    return Err(Illegal("cannot afford a settlement"));
                }
                if vertex as usize >= NUM_VERTICES || self.settlement_spots(pu) & (1u64 << vertex) == 0 {
                    return Err(Illegal("settlement spot unavailable"));
                }
                Ok(())
            }
            (Action::BuildCity { vertex }, Phase::Main) => {
                need_current(is_current)?;
                if ps.cities_left == 0 || !hand_covers(&ps.resources, &CITY_COST) {
                    return Err(Illegal("cannot afford a city"));
                }
                if vertex as usize >= NUM_VERTICES || ps.settlements & (1u64 << vertex) == 0 {
                    return Err(Illegal("city must replace your settlement"));
                }
                Ok(())
            }
            (Action::BuildRoad { edge }, Phase::Main) => {
                need_current(is_current)?;
                if ps.roads_left == 0 || !hand_covers(&ps.resources, &ROAD_COST) {
                    return Err(Illegal("cannot afford a road"));
                }
                if edge as usize >= NUM_EDGES || self.road_spots(pu) & (1u128 << edge) == 0 {
                    return Err(Illegal("road spot unavailable"));
                }
                Ok(())
            }
            (Action::BuildRoad { edge }, Phase::RoadBuilding { .. }) => {
                need_current(is_current)?;
                if edge as usize >= NUM_EDGES || self.road_spots(pu) & (1u128 << edge) == 0 {
                    return Err(Illegal("road spot unavailable"));
                }
                Ok(())
            }
            (Action::BuyDevCard, Phase::Main) => {
                need_current(is_current)?;
                if !hand_covers(&ps.resources, &DEV_CARD_COST) {
                    return Err(Illegal("cannot afford a development card"));
                }
                if self.dev_deck.iter().all(|&c| c == 0) {
                    return Err(Illegal("development deck is empty"));
                }
                Ok(())
            }
            (Action::PlayRoadBuilding, Phase::Main) => {
                need_current(is_current)?;
                if !self.can_play_dev(pu, DevCard::RoadBuilding) || ps.roads_left == 0 || self.road_spots(pu) == 0 {
                    return Err(Illegal("cannot play road building"));
                }
                Ok(())
            }
            (Action::PlayYearOfPlenty, Phase::Main) => {
                need_current(is_current)?;
                if !self.can_play_dev(pu, DevCard::YearOfPlenty) || self.bank.iter().all(|&b| b == 0) {
                    return Err(Illegal("cannot play year of plenty"));
                }
                Ok(())
            }
            (Action::PlayMonopoly, Phase::Main) => {
                need_current(is_current)?;
                if !self.can_play_dev(pu, DevCard::Monopoly) {
                    return Err(Illegal("cannot play monopoly"));
                }
                Ok(())
            }
            (Action::ChooseResource { resource }, Phase::YearOfPlenty { .. }) => {
                need_current(is_current)?;
                if resource as usize >= NUM_RESOURCES || self.bank[resource as usize] == 0 {
                    return Err(Illegal("bank has none of that resource"));
                }
                Ok(())
            }
            (Action::ChooseResource { resource }, Phase::Monopoly) => {
                need_current(is_current)?;
                if resource as usize >= NUM_RESOURCES {
                    return Err(Illegal("unknown resource"));
                }
                Ok(())
            }
            (Action::MaritimeTrade { give, get }, Phase::Main) => {
                need_current(is_current)?;
                let (g, r) = (give as usize, get as usize);
                if g >= NUM_RESOURCES || r >= NUM_RESOURCES || g == r {
                    return Err(Illegal("invalid maritime trade"));
                }
                if ps.resources[g] < ps.trade_ratio(g) || self.bank[r] == 0 {
                    return Err(Illegal("cannot afford maritime trade"));
                }
                Ok(())
            }
            (Action::OfferTrade { give, want }, Phase::Main) => {
                need_current(is_current)?;
                if self.trade_offers_this_turn >= self.config.max_trade_offers_per_turn {
                    return Err(Illegal("no trade offers left this turn"));
                }
                if !is_well_formed(&give, &want, self.config.max_trade_cards) {
                    return Err(Illegal("malformed trade"));
                }
                if !hand_covers(&ps.resources, &give) {
                    return Err(Illegal("you do not hold the offered cards"));
                }
                Ok(())
            }
            (Action::OfferTrade { give, want }, Phase::TradeResponse) => {
                let trade = self.trade.as_ref().unwrap();
                if trade.responses[pu] != Response::Pending {
                    return Err(NotYourTurn);
                }
                if !is_well_formed(&give, &want, self.config.max_trade_cards) {
                    return Err(Illegal("malformed trade"));
                }
                if !hand_covers(&ps.resources, &give) {
                    return Err(Illegal("you do not hold the offered cards"));
                }
                if !hand_covers(&self.players[trade.proposer as usize].resources, &want) {
                    return Err(Illegal("proposer does not hold the requested cards"));
                }
                Ok(())
            }
            (Action::AcceptTrade, Phase::TradeResponse) => {
                let trade = self.trade.as_ref().unwrap();
                if trade.responses[pu] != Response::Pending {
                    return Err(NotYourTurn);
                }
                if !hand_covers(&ps.resources, &trade.want) {
                    return Err(Illegal("you do not hold the requested cards"));
                }
                Ok(())
            }
            (Action::RejectTrade, Phase::TradeResponse) => {
                let trade = self.trade.as_ref().unwrap();
                if trade.responses[pu] != Response::Pending {
                    return Err(NotYourTurn);
                }
                Ok(())
            }
            (Action::ConfirmTrade { partner }, Phase::TradeConfirm) => {
                let trade = self.trade.as_ref().unwrap();
                if trade.proposer != p {
                    return Err(NotYourTurn);
                }
                if partner as usize >= self.n() || !self.confirmable(trade, partner as usize) {
                    return Err(Illegal("that player did not accept"));
                }
                Ok(())
            }
            (Action::CancelTrade, Phase::TradeResponse | Phase::TradeConfirm) => {
                if self.trade.as_ref().unwrap().proposer != p {
                    return Err(NotYourTurn);
                }
                Ok(())
            }
            _ => Err(WrongPhase),
        }
    }

    // ------------------------------------------------------------------
    // Transitions
    // ------------------------------------------------------------------

    /// Validate and apply an action.
    pub fn apply(&mut self, p: u8, action: Action) -> Result<(), ActionError> {
        self.check(p, &action)?;
        self.apply_unchecked(p, action);
        Ok(())
    }

    /// Apply an action that is known to be legal (e.g. drawn from [`Self::legal_actions`]).
    /// Applying an illegal action through this method can corrupt the state.
    pub fn apply_unchecked(&mut self, p: u8, action: Action) {
        let pu = p as usize;
        match action {
            Action::BuildSettlement { vertex } => {
                if self.phase == Phase::SetupSettlement {
                    self.place_settlement(pu, vertex);
                    if self.setup_step >= self.config.num_players {
                        self.grant_initial_resources(pu, vertex);
                    }
                    self.phase = Phase::SetupRoad { vertex };
                } else {
                    self.player_to_bank(pu, &SETTLEMENT_COST);
                    self.place_settlement(pu, vertex);
                }
            }
            Action::BuildCity { vertex } => {
                self.player_to_bank(pu, &CITY_COST);
                let ps = &mut self.players[pu];
                ps.settlements &= !(1u64 << vertex);
                ps.cities |= 1u64 << vertex;
                ps.settlements_left += 1;
                ps.cities_left -= 1;
                self.log(Event::CityBuilt { player: p, vertex });
            }
            Action::BuildRoad { edge } => match self.phase {
                Phase::SetupRoad { .. } => {
                    self.place_road(pu, edge);
                    self.setup_step += 1;
                    if self.setup_step == 2 * self.config.num_players {
                        self.current = 0;
                        self.phase = Phase::PreRoll;
                        self.log(Event::TurnStarted { player: 0, turn: 0 });
                    } else {
                        self.current = self.setup_player(self.setup_step);
                        self.phase = Phase::SetupSettlement;
                    }
                }
                Phase::RoadBuilding { remaining } => {
                    self.place_road(pu, edge);
                    let remaining = remaining - 1;
                    if remaining == 0 || self.players[pu].roads_left == 0 || self.road_spots(pu) == 0 {
                        self.phase = Phase::Main;
                    } else {
                        self.phase = Phase::RoadBuilding { remaining };
                    }
                }
                _ => {
                    self.player_to_bank(pu, &ROAD_COST);
                    self.place_road(pu, edge);
                }
            },
            Action::RollDice => self.roll_dice(),
            Action::Discard { resource } => {
                let mut h = EMPTY_HAND;
                h[resource as usize] = 1;
                self.player_to_bank(pu, &h);
                self.players[pu].discard_pending -= 1;
                self.log(Event::Discarded { player: p, resource });
                if self.players[..self.n()].iter().all(|ps| ps.discard_pending == 0) {
                    self.phase = Phase::MoveRobber;
                }
            }
            Action::MoveRobber { hex, victim } => {
                self.robber = hex;
                self.log(Event::RobberMoved { player: p, hex });
                if let Some(v) = victim {
                    let weights: [u32; NUM_RESOURCES] =
                        std::array::from_fn(|r| self.players[v as usize].resources[r] as u32);
                    if let Some(r) = self.rng.weighted_index(&weights) {
                        self.hidden_steal(v as usize, pu, r);
                        self.log(Event::Stolen {
                            thief: p,
                            victim: v,
                            resource: Some(r as u8),
                        });
                    }
                }
                self.phase = if self.dice_rolled { Phase::Main } else { Phase::PreRoll };
            }
            Action::EndTurn => self.end_turn(),
            Action::BuyDevCard => {
                self.player_to_bank(pu, &DEV_CARD_COST);
                let weights: [u32; NUM_DEV_TYPES] = std::array::from_fn(|i| self.dev_deck[i] as u32);
                let card = self.rng.weighted_index(&weights).expect("deck checked non-empty");
                self.dev_deck[card] -= 1;
                self.players[pu].new_dev_cards[card] += 1;
                self.log(Event::DevCardBought {
                    player: p,
                    card: Some(card as u8),
                });
            }
            Action::PlayKnight => {
                self.use_dev(pu, DevCard::Knight);
                self.players[pu].knights_played += 1;
                self.update_largest_army(pu);
                self.phase = Phase::MoveRobber;
            }
            Action::PlayRoadBuilding => {
                self.use_dev(pu, DevCard::RoadBuilding);
                self.phase = Phase::RoadBuilding {
                    remaining: 2.min(self.players[pu].roads_left),
                };
            }
            Action::PlayYearOfPlenty => {
                self.use_dev(pu, DevCard::YearOfPlenty);
                self.phase = Phase::YearOfPlenty { remaining: 2 };
            }
            Action::PlayMonopoly => {
                self.use_dev(pu, DevCard::Monopoly);
                self.phase = Phase::Monopoly;
            }
            Action::ChooseResource { resource } => {
                let r = resource as usize;
                match self.phase {
                    Phase::YearOfPlenty { remaining } => {
                        let mut h = EMPTY_HAND;
                        h[r] = 1;
                        self.bank_to_player(pu, &h);
                        self.log(Event::YearOfPlentyTaken { player: p, resource });
                        let remaining = remaining - 1;
                        self.phase = if remaining == 0 || self.bank.iter().all(|&b| b == 0) {
                            Phase::Main
                        } else {
                            Phase::YearOfPlenty { remaining }
                        };
                    }
                    Phase::Monopoly => {
                        let mut total = 0u8;
                        for q in 0..self.n() {
                            if q == pu {
                                continue;
                            }
                            let amt = self.players[q].resources[r];
                            if amt > 0 {
                                let mut h = EMPTY_HAND;
                                h[r] = amt;
                                self.player_to_player(q, pu, &h);
                                total += amt;
                            }
                            // Everyone now knows q holds none of this resource.
                            self.players[q].belief[r] = 0.0;
                            self.normalize_belief(q);
                        }
                        self.log(Event::MonopolyTaken {
                            player: p,
                            resource,
                            amount: total,
                        });
                        self.phase = Phase::Main;
                    }
                    _ => unreachable!("choose resource outside of a card effect"),
                }
            }
            Action::MaritimeTrade { give, get } => {
                let mut g = EMPTY_HAND;
                g[give as usize] = self.players[pu].trade_ratio(give as usize);
                let mut w = EMPTY_HAND;
                w[get as usize] = 1;
                self.player_to_bank(pu, &g);
                self.bank_to_player(pu, &w);
                self.log(Event::MaritimeTraded {
                    player: p,
                    give: g,
                    get: w,
                });
            }
            Action::OfferTrade { give, want } => {
                if self.phase == Phase::Main {
                    let mut responses = [Response::NotInvolved; MAX_PLAYERS];
                    for (q, r) in responses.iter_mut().enumerate().take(self.n()) {
                        if q != pu {
                            *r = Response::Pending;
                        }
                    }
                    self.trade = Some(TradeOffer {
                        proposer: p,
                        give,
                        want,
                        responses,
                    });
                    self.trade_offers_this_turn += 1;
                    self.phase = Phase::TradeResponse;
                    self.log(Event::TradeOffered {
                        proposer: p,
                        give,
                        want,
                    });
                } else {
                    // Counter-offer: store in the proposer's orientation.
                    let trade = self.trade.as_mut().unwrap();
                    trade.responses[pu] = Response::Counter { give: want, want: give };
                    self.log(Event::TradeResponded {
                        player: p,
                        response: TradeResponseKind::Counter { give: want, want: give },
                    });
                    self.advance_trade();
                }
            }
            Action::AcceptTrade => {
                self.trade.as_mut().unwrap().responses[pu] = Response::Accept;
                self.log(Event::TradeResponded {
                    player: p,
                    response: TradeResponseKind::Accept,
                });
                self.advance_trade();
            }
            Action::RejectTrade => {
                self.trade.as_mut().unwrap().responses[pu] = Response::Reject;
                self.log(Event::TradeResponded {
                    player: p,
                    response: TradeResponseKind::Reject,
                });
                self.advance_trade();
            }
            Action::ConfirmTrade { partner } => {
                let trade = self.trade.take().unwrap();
                let (give, want) = match trade.responses[partner as usize] {
                    Response::Counter { give, want } => (give, want),
                    _ => (trade.give, trade.want),
                };
                self.player_to_player(pu, partner as usize, &give);
                self.player_to_player(partner as usize, pu, &want);
                self.log(Event::TradeExecuted {
                    proposer: p,
                    partner,
                    give,
                    want,
                });
                self.phase = Phase::Main;
            }
            Action::CancelTrade => {
                self.trade = None;
                self.phase = Phase::Main;
                self.log(Event::TradeCancelled { proposer: p });
            }
        }

        // A player can only win on their own turn.
        if self.phase != Phase::GameOver && self.total_vp(self.current as usize) >= self.config.vp_to_win {
            self.declare_winner(self.current);
        }
    }

    fn declare_winner(&mut self, p: u8) {
        self.winner = Some(p);
        self.phase = Phase::GameOver;
        self.trade = None;
        self.log(Event::GameWon { player: p });
    }

    fn advance_trade(&mut self) {
        let trade = self.trade.as_ref().unwrap();
        if trade.responses.contains(&Response::Pending) {
            return;
        }
        let any_taker = (0..self.n()).any(|q| self.confirmable(trade, q));
        if any_taker {
            self.phase = Phase::TradeConfirm;
        } else {
            let proposer = trade.proposer;
            self.trade = None;
            self.phase = Phase::Main;
            self.log(Event::TradeCancelled { proposer });
        }
    }

    fn use_dev(&mut self, p: usize, card: DevCard) {
        self.players[p].dev_cards[card as usize] -= 1;
        self.dev_played_this_turn = true;
        self.log(Event::DevCardPlayed {
            player: p as u8,
            card: card as u8,
        });
    }

    fn roll_dice(&mut self) {
        let dice = [self.rng.roll_die(), self.rng.roll_die()];
        self.apply_roll(dice);
    }

    /// Resolve a specific dice result for the current player (used by `roll_dice` and by
    /// tests / UIs that want to force a roll).
    pub fn apply_roll(&mut self, dice: [u8; 2]) {
        let p = self.current;
        self.last_roll = Some(dice);
        self.dice_rolled = true;
        self.log(Event::DiceRolled { player: p, dice });
        let total = dice[0] + dice[1];
        if total == 7 {
            let mut any = false;
            for q in 0..self.n() {
                let c = self.players[q].num_resources();
                if c > self.config.discard_limit as u32 {
                    self.players[q].discard_pending = (c / 2) as u8;
                    any = true;
                }
            }
            self.phase = if any { Phase::Discard } else { Phase::MoveRobber };
        } else {
            self.produce(total);
            self.phase = Phase::Main;
        }
    }

    fn produce(&mut self, total: u8) {
        let t = topo();
        let mut gains = [[0u8; NUM_RESOURCES]; MAX_PLAYERS];
        for h in bits64(self.board.number_hexes[total as usize] as u64) {
            if h == self.robber {
                continue;
            }
            let Some(res) = self.board.hex_resource(h as usize) else {
                continue;
            };
            for &v in &t.hex_vertices[h as usize] {
                let o = self.vertex_owner[v as usize];
                if o != NONE {
                    let amt = if self.players[o as usize].cities & (1u64 << v) != 0 {
                        2
                    } else {
                        1
                    };
                    gains[o as usize][res as usize] += amt;
                }
            }
        }
        // Bank shortage: if the bank cannot pay everyone, nobody receives that resource,
        // unless only a single player is owed it (they get what is left).
        for r in 0..NUM_RESOURCES {
            let demand: u32 = gains.iter().map(|g| g[r] as u32).sum();
            if demand > self.bank[r] as u32 {
                let recipients = gains.iter().filter(|g| g[r] > 0).count();
                for g in gains.iter_mut() {
                    if g[r] > 0 {
                        g[r] = if recipients == 1 { self.bank[r] } else { 0 };
                    }
                }
            }
        }
        for q in 0..self.n() {
            if gains[q].iter().any(|&x| x > 0) {
                self.bank_to_player(q, &gains[q]);
                self.log(Event::Produced {
                    player: q as u8,
                    resources: gains[q],
                });
            }
        }
    }

    fn grant_initial_resources(&mut self, p: usize, vertex: u8) {
        let t = topo();
        let mut h = EMPTY_HAND;
        for &hex in &t.vertex_hexes[vertex as usize] {
            if hex == NONE {
                continue;
            }
            if let Some(r) = self.board.hex_resource(hex as usize) {
                h[r as usize] += 1;
            }
        }
        self.bank_to_player(p, &h);
        self.log(Event::Produced {
            player: p as u8,
            resources: h,
        });
    }

    fn place_settlement(&mut self, p: usize, v: u8) {
        let t = topo();
        let vi = v as usize;
        let ps = &mut self.players[p];
        ps.settlements |= 1u64 << v;
        ps.settlements_left -= 1;
        if let Some(port) = self.board.vertex_port[vi] {
            ps.ports |= port.bit();
        }
        self.vertex_owner[vi] = p as u8;
        self.blocked |= t.vertex_footprint[vi];
        self.log(Event::SettlementBuilt {
            player: p as u8,
            vertex: v,
        });

        // A new settlement may cut an opponent's road.
        let mut changed = false;
        for q in 0..self.n() {
            if q != p && self.players[q].road_vertices & (1u64 << v) != 0 {
                let len = self.compute_longest_road(q);
                if len != self.players[q].longest_road {
                    self.players[q].longest_road = len;
                    changed = true;
                }
            }
        }
        if changed {
            self.update_longest_road_owner();
        }
    }

    fn place_road(&mut self, p: usize, e: u8) {
        let t = topo();
        let ps = &mut self.players[p];
        ps.roads |= 1u128 << e;
        ps.road_vertices |= t.edge_vertex_mask[e as usize];
        ps.roads_left -= 1;
        self.edge_owner[e as usize] = p as u8;
        self.log(Event::RoadBuilt {
            player: p as u8,
            edge: e,
        });
        let len = self.compute_longest_road(p);
        if len != self.players[p].longest_road {
            self.players[p].longest_road = len;
            self.update_longest_road_owner();
        }
    }

    /// Longest simple path (by edges) through `p`'s road network, where paths may not
    /// pass *through* a vertex occupied by an opponent.
    pub fn compute_longest_road(&self, p: usize) -> u8 {
        let roads = self.players[p].roads;
        if roads == 0 {
            return 0;
        }
        let blocked = self.opponent_buildings(p);
        let t = topo();
        let mut best = 0u8;
        for v in bits64(self.players[p].road_vertices) {
            // Starting from vertices of degree 1 (in our network) suffices for trees, but
            // cycles need every start point; the network is tiny so we try all.
            dfs_road(t, roads, blocked, v, 0, 0, &mut best);
        }
        best
    }

    fn update_longest_road_owner(&mut self) {
        let n = self.n();
        let max = (0..n).map(|q| self.players[q].longest_road).max().unwrap_or(0);
        let old = self.longest_road_owner;
        let new = match old {
            Some(o) if self.players[o as usize].longest_road == max && max >= 5 => Some(o),
            _ => {
                let holders: Vec<usize> = (0..n).filter(|&q| self.players[q].longest_road == max).collect();
                if max >= 5 && holders.len() == 1 {
                    Some(holders[0] as u8)
                } else {
                    None
                }
            }
        };
        if new != old {
            self.longest_road_owner = new;
            self.log(Event::LongestRoadChanged {
                player: new,
                length: max,
            });
        }
    }

    fn update_largest_army(&mut self, p: usize) {
        let k = self.players[p].knights_played;
        if k < 3 || self.largest_army_owner == Some(p as u8) {
            return;
        }
        let beats = match self.largest_army_owner {
            None => true,
            Some(o) => k > self.players[o as usize].knights_played,
        };
        if beats {
            self.largest_army_owner = Some(p as u8);
            self.log(Event::LargestArmyChanged {
                player: Some(p as u8),
                knights: k,
            });
        }
    }

    fn end_turn(&mut self) {
        let p = self.current as usize;
        let ps = &mut self.players[p];
        for i in 0..NUM_DEV_TYPES {
            ps.dev_cards[i] += ps.new_dev_cards[i];
            ps.new_dev_cards[i] = 0;
        }
        self.current = (self.current + 1) % self.config.num_players;
        self.turn += 1;
        self.dice_rolled = false;
        self.dev_played_this_turn = false;
        self.trade_offers_this_turn = 0;
        self.phase = Phase::PreRoll;
        if self.config.max_turns > 0 && self.turn >= self.config.max_turns {
            self.phase = Phase::GameOver;
            self.log(Event::GameTruncated);
            return;
        }
        self.log(Event::TurnStarted {
            player: self.current,
            turn: self.turn,
        });
    }
}

fn dfs_road(t: &crate::topology::Topology, roads: u128, blocked: u64, v: u8, used: u128, len: u8, best: &mut u8) {
    if len > *best {
        *best = len;
    }
    if len > 0 && blocked & (1u64 << v) != 0 {
        return;
    }
    for &e in &t.vertex_edges[v as usize] {
        if e == NONE {
            continue;
        }
        let bit = 1u128 << e;
        if roads & bit != 0 && used & bit == 0 {
            dfs_road(t, roads, blocked, t.edge_other(e, v), used | bit, len + 1, best);
        }
    }
}
