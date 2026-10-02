#![allow(clippy::needless_range_loop)]

use catan_core::bots::{play_game, Bot, HeuristicBot, RandomBot};
use catan_core::encode::{action_to_index, index_to_action, legal_mask, observe, ACTION_SIZE, OBS_SIZE};
use catan_core::topology::{bits64, topo};
use catan_core::*;

fn config() -> GameConfig {
    GameConfig {
        max_turns: 400,
        record_events: true,
        ..GameConfig::default()
    }
}

/// Cross-cutting invariants that must hold after every action.
fn check_invariants(s: &GameState) {
    let n = s.n();
    for r in 0..NUM_RESOURCES {
        let total: u32 = s.bank[r] as u32 + (0..n).map(|p| s.players[p].resources[r] as u32).sum::<u32>();
        assert_eq!(total, BANK_START as u32, "resource {r} not conserved");
    }
    for d in 0..NUM_DEV_TYPES {
        let held: u32 = (0..n)
            .map(|p| (s.players[p].dev_cards[d] + s.players[p].new_dev_cards[d]) as u32)
            .sum();
        assert!(held + s.dev_deck[d] as u32 <= DEV_DECK[d] as u32);
    }
    for p in 0..n {
        let ps = &s.players[p];
        assert_eq!(ps.settlements.count_ones() as u8 + ps.settlements_left, MAX_SETTLEMENTS);
        assert_eq!(ps.cities.count_ones() as u8 + ps.cities_left, MAX_CITIES);
        assert_eq!(ps.roads.count_ones() as u8 + ps.roads_left, MAX_ROADS);
        let b: f32 = ps.belief.iter().sum();
        assert!(
            (b - ps.num_resources() as f32).abs() < 1e-2,
            "belief {:?} vs {:?}",
            ps.belief,
            ps.resources
        );
        assert!(ps.belief.iter().all(|&x| x >= 0.0));
        assert_eq!(ps.longest_road, s.compute_longest_road(p));
    }
    let t = topo();
    let all: u64 = (0..n).fold(0, |a, p| a | s.players[p].buildings());
    for v in bits64(all) {
        assert_eq!(
            t.vertex_neighbor_mask[v as usize] & all,
            0,
            "distance rule violated at {v}"
        );
    }
}

fn random_bots(seed: u64, n: u64) -> Vec<Box<dyn Bot>> {
    (0..n)
        .map(|i| Box::new(RandomBot::new(seed * 10 + i)) as Box<dyn Bot>)
        .collect()
}

fn heuristic_bots(seed: u64, n: u64) -> Vec<Box<dyn Bot>> {
    (0..n)
        .map(|i| Box::new(HeuristicBot::new(seed * 10 + i)) as Box<dyn Bot>)
        .collect()
}

#[test]
fn random_games_preserve_invariants_and_generated_moves_validate() {
    let mut buf = Vec::new();
    let mut other = Vec::new();
    for seed in 0..60u64 {
        let mut s = GameState::new(config(), seed);
        let mut bots = random_bots(seed, 4);
        let mut steps = 0;
        while let Some(p) = s.next_actor() {
            s.legal_actions(p, &mut buf);
            for a in &buf {
                assert_eq!(s.check(p, a), Ok(()), "generated {a:?} fails check in {:?}", s.phase);
            }
            // Non-actors have nothing to do (except a proposer withdrawing an offer).
            for q in 0..4u8 {
                if s.actors() & (1 << q) == 0 {
                    s.legal_actions(q, &mut other);
                    assert!(
                        other.iter().all(|a| *a == Action::CancelTrade),
                        "{q} has {other:?} in {:?}",
                        s.phase
                    );
                }
            }
            let a = bots[p as usize].choose(&s, p);
            s.apply(p, a).unwrap();
            check_invariants(&s);
            steps += 1;
            assert!(steps < 100_000);
        }
    }
}

#[test]
fn heuristic_bots_finish_games() {
    let mut finished = 0;
    for seed in 0..40u64 {
        let mut s = GameState::new(config(), seed);
        let mut bots = heuristic_bots(seed, 4);
        while let Some(p) = s.next_actor() {
            let a = bots[p as usize].choose(&s, p);
            s.apply(p, a).unwrap();
            check_invariants(&s);
        }
        if let Some(w) = s.winner {
            assert!(s.total_vp(w as usize) >= 10);
            finished += 1;
        }
    }
    assert!(finished >= 35, "only {finished}/40 heuristic games finished");
}

#[test]
fn heuristic_beats_random() {
    let mut wins = 0;
    let games = 40;
    for seed in 0..games {
        let mut s = GameState::new(config(), 1000 + seed);
        let mut bots: Vec<Box<dyn Bot>> = vec![
            Box::new(HeuristicBot::new(seed)),
            Box::new(RandomBot::new(seed + 1)),
            Box::new(RandomBot::new(seed + 2)),
            Box::new(RandomBot::new(seed + 3)),
        ];
        play_game(&mut s, &mut bots);
        if s.winner == Some(0) {
            wins += 1;
        }
    }
    assert!(wins > games / 2, "heuristic won only {wins}/{games}");
}

#[test]
fn mask_matches_legal_actions_and_encoding_roundtrips() {
    let mut mask = vec![false; ACTION_SIZE];
    let mut scratch = Vec::new();
    let mut obs = vec![0f32; OBS_SIZE];
    for seed in 0..10u64 {
        let mut s = GameState::new(config(), seed);
        let mut bot = RandomBot::new(seed);
        while let Some(p) = s.next_actor() {
            let count = legal_mask(&s, p, &mut mask, &mut scratch);
            assert!(count > 0);
            for (i, &legal) in mask.iter().enumerate() {
                if legal {
                    let a = index_to_action(&s, p, i).unwrap();
                    assert_eq!(s.check(p, &a), Ok(()));
                    assert_eq!(action_to_index(&s, p, &a), Some(i));
                }
            }
            observe(&s, p, &mut obs);
            assert!(obs.iter().all(|x| x.is_finite()));
            let a = bot.choose(&s, p);
            s.apply_unchecked(p, a);
        }
    }
}

#[test]
fn snake_setup_order_and_second_settlement_resources() {
    let mut s = GameState::new(GameConfig::default(), 5);
    let mut order = Vec::new();
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let p = s.current;
        if s.phase == Phase::SetupSettlement {
            order.push(p);
        }
        s.legal_actions(p, &mut buf);
        s.apply(p, buf[0]).unwrap();
    }
    assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.current, 0);
    let total: u32 = (0..4).map(|p| s.players[p].num_resources()).sum();
    assert!(total >= 4, "second settlements should grant resources");
}

/// Run setup with the first legal action each time; returns the state at PreRoll.
fn after_setup(seed: u64) -> GameState {
    let mut s = GameState::new(config(), seed);
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let p = s.current;
        s.legal_actions(p, &mut buf);
        s.apply(p, buf[0]).unwrap();
    }
    s
}

/// Force a non-7 roll so the current player reaches the main phase.
fn to_main(s: &mut GameState) {
    s.apply_roll([2, 3]);
    assert_eq!(s.phase, Phase::Main);
}

/// Overwrite all hands, rebalancing the bank so resources stay conserved.
fn set_hands(s: &mut GameState, hands: [Hand; 4]) {
    s.bank = [BANK_START; NUM_RESOURCES];
    for (p, h) in hands.iter().enumerate() {
        s.players[p].resources = *h;
        s.players[p].belief = h.map(|x| x as f32);
        for r in 0..NUM_RESOURCES {
            s.bank[r] -= h[r];
        }
    }
}

#[test]
fn trade_offer_accept_counter_confirm_flow() {
    let mut s = after_setup(11);
    to_main(&mut s);
    set_hands(&mut s, [[2, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0; 5]]);
    s.apply(
        0,
        Action::OfferTrade {
            give: [1, 0, 0, 0, 0],
            want: [0, 0, 0, 1, 0],
        },
    )
    .unwrap();
    assert_eq!(s.phase, Phase::TradeResponse);
    assert_eq!(s.actors(), 0b1110);
    // Player 3 cannot afford to accept.
    assert!(s.check(3, &Action::AcceptTrade).is_err());
    s.apply(1, Action::AcceptTrade).unwrap();
    // Player 2 counters: gives 1 wheat, wants 2 wood.
    s.apply(
        2,
        Action::OfferTrade {
            give: [0, 0, 0, 1, 0],
            want: [2, 0, 0, 0, 0],
        },
    )
    .unwrap();
    s.apply(3, Action::RejectTrade).unwrap();
    assert_eq!(s.phase, Phase::TradeConfirm);
    let mut buf = Vec::new();
    s.legal_actions(0, &mut buf);
    assert!(buf.contains(&Action::ConfirmTrade { partner: 1 }));
    assert!(buf.contains(&Action::ConfirmTrade { partner: 2 }));
    assert!(!buf.contains(&Action::ConfirmTrade { partner: 3 }));
    s.apply(0, Action::ConfirmTrade { partner: 2 }).unwrap();
    assert_eq!(s.players[0].resources, [0, 0, 0, 1, 0]);
    assert_eq!(s.players[2].resources, [2, 0, 0, 0, 0]);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.trade_offers_this_turn, 1);
    check_invariants(&s);
}

#[test]
fn trade_with_no_takers_returns_to_main_and_offers_are_capped() {
    let mut s = after_setup(12);
    to_main(&mut s);
    set_hands(&mut s, [[5, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    let give = [1, 0, 0, 0, 0];
    let want = [0, 0, 0, 0, 1];
    for _ in 0..s.config.max_trade_offers_per_turn {
        s.apply(0, Action::OfferTrade { give, want }).unwrap();
        for q in 1..4 {
            s.apply(q, Action::RejectTrade).unwrap();
        }
        assert_eq!(s.phase, Phase::Main);
    }
    assert!(s.apply(0, Action::OfferTrade { give, want }).is_err());
}

#[test]
fn proposer_can_cancel_while_waiting() {
    let mut s = after_setup(13);
    to_main(&mut s);
    set_hands(&mut s, [[1, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    s.apply(
        0,
        Action::OfferTrade {
            give: [1, 0, 0, 0, 0],
            want: [0, 1, 0, 0, 0],
        },
    )
    .unwrap();
    s.apply(0, Action::CancelTrade).unwrap();
    assert_eq!(s.phase, Phase::Main);
    assert!(s.trade.is_none());
}

#[test]
fn malformed_trades_rejected_but_arbitrary_trades_allowed() {
    let mut s = after_setup(14);
    to_main(&mut s);
    set_hands(&mut s, [[3, 0, 0, 0, 0], [0; 5], [0; 5], [0; 5]]);
    assert!(s
        .check(
            0,
            &Action::OfferTrade {
                give: [1, 0, 0, 0, 0],
                want: [1, 0, 0, 0, 0]
            }
        )
        .is_err());
    assert!(s
        .check(
            0,
            &Action::OfferTrade {
                give: [0; 5],
                want: [1, 0, 0, 0, 0]
            }
        )
        .is_err());
    assert!(s
        .check(
            0,
            &Action::OfferTrade {
                give: [4, 0, 0, 0, 0],
                want: [0, 1, 0, 0, 0]
            }
        )
        .is_err());
    assert!(s
        .check(
            0,
            &Action::OfferTrade {
                give: [3, 0, 0, 0, 0],
                want: [0, 1, 1, 0, 0]
            }
        )
        .is_ok());
}

#[test]
fn seven_forces_discards_then_robber() {
    let mut s = after_setup(21);
    set_hands(&mut s, [[0; 5], [3, 3, 3, 0, 0], [0; 5], [0; 5]]);
    s.apply_roll([3, 4]);
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(s.actors(), 0b0010);
    assert_eq!(s.players[1].discard_pending, 4);
    while s.phase == Phase::Discard {
        let p = s.next_actor().unwrap();
        let r = (0..5).find(|&r| s.players[p as usize].resources[r] > 0).unwrap();
        s.apply(p, Action::Discard { resource: r as u8 }).unwrap();
    }
    assert_eq!(s.players[1].num_resources(), 5);
    assert_eq!(s.phase, Phase::MoveRobber);
    check_invariants(&s);
}

#[test]
fn robber_cannot_stay_and_must_rob_when_possible() {
    let mut s = after_setup(22);
    set_hands(&mut s, [[0; 5], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]);
    s.apply_roll([3, 4]);
    assert_eq!(s.phase, Phase::MoveRobber);
    let robber = s.robber;
    assert!(s
        .check(
            0,
            &Action::MoveRobber {
                hex: robber,
                victim: None
            }
        )
        .is_err());
    let mut buf = Vec::new();
    s.legal_actions(0, &mut buf);
    for a in &buf {
        if let Action::MoveRobber { hex, victim: None } = a {
            assert_eq!(s.robbable(0, *hex as usize), 0);
        }
    }
    let rob = buf
        .iter()
        .find(|a| matches!(a, Action::MoveRobber { victim: Some(_), .. }))
        .copied()
        .unwrap();
    s.apply(0, rob).unwrap();
    assert_eq!(s.players[0].num_resources(), 1);
    check_invariants(&s);
}

#[test]
fn dev_card_cannot_be_played_the_turn_it_is_bought() {
    let mut s = after_setup(31);
    to_main(&mut s);
    set_hands(&mut s, [[0, 0, 1, 1, 1], [0; 5], [0; 5], [0; 5]]);
    s.dev_deck = [1, 0, 0, 0, 0];
    s.apply(0, Action::BuyDevCard).unwrap();
    assert_eq!(s.players[0].new_dev_cards[0], 1);
    assert!(s.check(0, &Action::PlayKnight).is_err());
    s.apply(0, Action::EndTurn).unwrap();
    assert_eq!(s.players[0].dev_cards[0], 1);
}

#[test]
fn only_one_dev_card_per_turn() {
    let mut s = after_setup(32);
    s.players[0].dev_cards = [2, 0, 0, 0, 1];
    s.apply(0, Action::PlayKnight).unwrap();
    assert_eq!(s.phase, Phase::MoveRobber);
    let mut buf = Vec::new();
    s.legal_actions(0, &mut buf);
    s.apply(0, buf[0]).unwrap();
    assert_eq!(s.phase, Phase::PreRoll);
    to_main(&mut s);
    assert!(s.check(0, &Action::PlayKnight).is_err());
    assert!(s.check(0, &Action::PlayMonopoly).is_err());
}

#[test]
fn monopoly_collects_everything() {
    let mut s = after_setup(33);
    to_main(&mut s);
    set_hands(&mut s, [[0; 5], [0, 0, 0, 2, 0], [0, 0, 0, 3, 1], [1, 0, 0, 0, 0]]);
    s.players[0].dev_cards[DevCard::Monopoly as usize] = 1;
    s.apply(0, Action::PlayMonopoly).unwrap();
    s.apply(0, Action::ChooseResource { resource: 3 }).unwrap();
    assert_eq!(s.players[0].resources[3], 5);
    assert_eq!(s.players[1].resources[3], 0);
    assert_eq!(s.players[2].belief[3], 0.0);
    check_invariants(&s);
}

#[test]
fn year_of_plenty_takes_two() {
    let mut s = after_setup(34);
    to_main(&mut s);
    set_hands(&mut s, [[0; 5], [0; 5], [0; 5], [0; 5]]);
    s.players[0].dev_cards[DevCard::YearOfPlenty as usize] = 1;
    s.apply(0, Action::PlayYearOfPlenty).unwrap();
    s.apply(0, Action::ChooseResource { resource: 4 }).unwrap();
    s.apply(0, Action::ChooseResource { resource: 4 }).unwrap();
    assert_eq!(s.players[0].resources, [0, 0, 0, 0, 2]);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn bank_shortage_blocks_contested_production() {
    let mut s = after_setup(35);
    // Pick a number that pays two different players the same resource, if one exists.
    let t = topo();
    for h in 0..NUM_HEXES {
        let Some(res) = s.board.hex_resource(h) else { continue };
        if h == s.robber as usize {
            continue;
        }
        let owners: Vec<u8> = t.hex_vertices[h]
            .iter()
            .map(|&v| s.vertex_owner[v as usize])
            .filter(|&o| o != u8::MAX)
            .collect();
        let mut distinct = owners.clone();
        distinct.sort();
        distinct.dedup();
        if distinct.len() < 2 {
            continue;
        }
        let n = s.board.numbers[h];
        let r = res as usize;
        // Leave only one card of that resource in the bank.
        let mut hands = [[0u8; 5]; 4];
        hands[3][r] = BANK_START - 1;
        set_hands(&mut s, hands);
        s.apply_roll([1, n - 1]);
        assert_eq!(s.bank[r], 1, "contested shortage should pay nobody");
        return;
    }
}

#[test]
fn longest_road_awarded_and_broken() {
    let t = topo();
    let mut s = GameState::new(GameConfig::default(), 1);
    let mut path = vec![0u8];
    while path.len() < 6 {
        let v = *path.last().unwrap();
        let next = t.vertex_neighbors[v as usize]
            .iter()
            .copied()
            .find(|&u| u != u8::MAX && !path.contains(&u))
            .unwrap();
        path.push(next);
    }
    s.phase = Phase::Main;
    s.dice_rolled = true;
    s.players[0].settlements |= 1u64 << path[0];
    s.players[0].settlements_left -= 1;
    s.vertex_owner[path[0] as usize] = 0;
    s.blocked |= t.vertex_footprint[path[0] as usize];
    for w in path.windows(2) {
        let e = (0..NUM_EDGES as u8)
            .find(|&e| {
                let [a, b] = t.edge_vertices[e as usize];
                (a == w[0] && b == w[1]) || (a == w[1] && b == w[0])
            })
            .unwrap();
        set_hands(&mut s, [ROAD_COST, [0; 5], [0; 5], [0; 5]]);
        s.apply(0, Action::BuildRoad { edge: e }).unwrap();
    }
    assert_eq!(s.players[0].longest_road, 5);
    assert_eq!(s.longest_road_owner, Some(0));
    assert_eq!(s.public_vp(0), 3);

    // Player 1 settles in the middle of the path, splitting it into 3 + 2.
    s.current = 1;
    s.phase = Phase::SetupSettlement;
    s.blocked = 0;
    s.apply(1, Action::BuildSettlement { vertex: path[3] }).unwrap();
    assert_eq!(s.players[0].longest_road, 3);
    assert_eq!(s.longest_road_owner, None);
}

#[test]
fn serde_roundtrip() {
    let mut s = after_setup(41);
    to_main(&mut s);
    let json = serde_json::to_string(&s).unwrap();
    let back: GameState = serde_json::from_str(&json).unwrap();
    assert_eq!(s, back);
}

#[test]
fn deterministic_given_seed() {
    let run = |seed| {
        let mut s = GameState::new(config(), seed);
        let mut bots = random_bots(7, 4);
        play_game(&mut s, &mut bots);
        (s.winner, s.turn)
    };
    assert_eq!(run(99), run(99));
}

#[test]
fn two_and_three_player_games_work() {
    for n in [2u8, 3] {
        for seed in 0..10 {
            let mut s = GameState::new(
                GameConfig {
                    num_players: n,
                    ..config()
                },
                seed,
            );
            let mut bots = heuristic_bots(seed, n as u64);
            while let Some(p) = s.next_actor() {
                let a = bots[p as usize].choose(&s, p);
                s.apply(p, a).unwrap();
                check_invariants(&s);
            }
        }
    }
}

#[test]
fn events_are_recorded_and_redacted() {
    let mut s = GameState::new(config(), 3);
    let mut bots = random_bots(3, 4);
    play_game(&mut s, &mut bots);
    let events = s.take_events();
    assert!(events.iter().any(|e| matches!(e, Event::DiceRolled { .. })));
    for e in &events {
        if let Event::Stolen { thief, victim, .. } = e {
            let other = (0..4).find(|q| q != thief && q != victim).unwrap();
            assert!(matches!(
                e.redacted_for(Some(other)),
                Event::Stolen { resource: None, .. }
            ));
            assert!(matches!(
                e.redacted_for(Some(*thief)),
                Event::Stolen { resource: Some(_), .. }
            ));
        }
    }
}

#[test]
fn heuristic_bots_trade_with_each_other() {
    use catan_core::event::TradeResponseKind;
    let (mut executed, mut counters) = (0, 0);
    for seed in 0..20u64 {
        let mut s = GameState::new(config(), 500 + seed);
        let mut bots = heuristic_bots(seed, 4);
        play_game(&mut s, &mut bots);
        for e in s.take_events() {
            match e {
                Event::TradeExecuted { .. } => executed += 1,
                Event::TradeResponded {
                    response: TradeResponseKind::Counter { .. },
                    ..
                } => counters += 1,
                _ => {}
            }
        }
    }
    assert!(executed > 20, "only {executed} player trades executed");
    assert!(counters > 10, "only {counters} counter-offers made");
}
