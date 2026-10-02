//! Throughput benchmark: plays full games with scripted bots and reports games/s and
//! actions/s.
//!
//! ```text
//! cargo run --release -p catan-core --example bench -- [games] [random|heuristic] [--encode]
//! ```

use catan_core::bots::{Bot, HeuristicBot, RandomBot};
use catan_core::encode::{legal_mask, observe, ACTION_SIZE, OBS_SIZE};
use catan_core::{GameConfig, GameState};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let bot = args.get(2).cloned().unwrap_or_else(|| "random".into());
    let encode = args.iter().any(|a| a == "--encode");
    let config = GameConfig {
        max_turns: 500,
        ..GameConfig::default()
    };

    let start = Instant::now();
    let mut actions = 0u64;
    let mut wins = [0u64; 5];
    let mut obs = vec![0f32; OBS_SIZE];
    let mut mask = vec![false; ACTION_SIZE];
    let mut scratch = Vec::new();
    for g in 0..games {
        let mut s = GameState::new(config.clone(), g);
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|i| -> Box<dyn Bot> {
                if bot == "heuristic" {
                    Box::new(HeuristicBot::new(g * 4 + i))
                } else {
                    Box::new(RandomBot::new(g * 4 + i))
                }
            })
            .collect();
        while let Some(p) = s.next_actor() {
            if encode {
                observe(&s, p, &mut obs);
                legal_mask(&s, p, &mut mask, &mut scratch);
            }
            let a = bots[p as usize].choose(&s, p);
            s.apply_unchecked(p, a);
            actions += 1;
        }
        wins[s.winner.map_or(4, |w| w as usize)] += 1;
    }
    let dt = start.elapsed().as_secs_f64();
    println!(
        "{games} games ({bot}{}) in {dt:.2}s: {:.0} games/s, {:.0} actions/s, {:.1} actions/game",
        if encode { " + encode" } else { "" },
        games as f64 / dt,
        actions as f64 / dt,
        actions as f64 / games as f64
    );
    println!("wins by seat {wins:?} (last = truncated)");
}
