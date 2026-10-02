//! Trade templates: the finite set of player-to-player offers exposed to AI agents.
//!
//! The engine accepts *any* well-formed offer (humans can propose arbitrary trades),
//! but learning agents choose from this fixed catalogue so the action space stays
//! discrete. The catalogue covers the shapes that make up the vast majority of
//! real-game trades:
//!
//! | shape          | count |
//! |----------------|-------|
//! | 1 X  for 1 Y   | 20    |
//! | 2 X  for 1 Y   | 20    |
//! | 1 X  for 2 Y   | 20    |
//! | 1 X + 1 Y for 1 Z | 30 |
//! | 1 Z for 1 X + 1 Y | 30 |
//!
//! for a total of [`NUM_TRADE_TEMPLATES`] = 120.

use crate::types::{Hand, EMPTY_HAND, NUM_RESOURCES};
use std::sync::OnceLock;

pub const NUM_TRADE_TEMPLATES: usize = 120;

/// `(give, want)` from the offering player's point of view.
pub type TradeTemplate = (Hand, Hand);

fn one(r: usize, n: u8) -> Hand {
    let mut h = EMPTY_HAND;
    h[r] = n;
    h
}

fn build() -> Vec<TradeTemplate> {
    let mut out = Vec::with_capacity(NUM_TRADE_TEMPLATES);
    for (gn, wn) in [(1u8, 1u8), (2, 1), (1, 2)] {
        for g in 0..NUM_RESOURCES {
            for w in 0..NUM_RESOURCES {
                if g != w {
                    out.push((one(g, gn), one(w, wn)));
                }
            }
        }
    }
    // Two different resources for one other resource, and the reverse.
    for reverse in [false, true] {
        for a in 0..NUM_RESOURCES {
            for b in (a + 1)..NUM_RESOURCES {
                for z in 0..NUM_RESOURCES {
                    if z == a || z == b {
                        continue;
                    }
                    let mut pair = EMPTY_HAND;
                    pair[a] = 1;
                    pair[b] = 1;
                    let single = one(z, 1);
                    out.push(if reverse { (single, pair) } else { (pair, single) });
                }
            }
        }
    }
    assert_eq!(out.len(), NUM_TRADE_TEMPLATES);
    out
}

static TEMPLATES: OnceLock<Vec<TradeTemplate>> = OnceLock::new();
static LOOKUP: OnceLock<Vec<u8>> = OnceLock::new();

#[inline]
pub fn templates() -> &'static [TradeTemplate] {
    TEMPLATES.get_or_init(build)
}

/// Base-3 key of a hand whose entries are all <= 2.
#[inline]
fn key3(h: &Hand) -> Option<usize> {
    let mut k = 0usize;
    for &x in h.iter() {
        if x > 2 {
            return None;
        }
        k = k * 3 + x as usize;
    }
    Some(k)
}

/// Index of a trade in the template catalogue, if it is representable. O(1).
#[inline]
pub fn template_index(give: &Hand, want: &Hand) -> Option<usize> {
    let table = LOOKUP.get_or_init(|| {
        let mut t = vec![u8::MAX; 243 * 243];
        for (i, (g, w)) in templates().iter().enumerate() {
            t[key3(g).unwrap() * 243 + key3(w).unwrap()] = i as u8;
        }
        t
    });
    let idx = table[key3(give)? * 243 + key3(want)?];
    (idx != u8::MAX).then_some(idx as usize)
}

/// A trade is well-formed if both sides are non-empty, no resource appears on both
/// sides, and neither side exceeds `max_cards`.
pub fn is_well_formed(give: &Hand, want: &Hand, max_cards: u8) -> bool {
    let g: u32 = give.iter().map(|&x| x as u32).sum();
    let w: u32 = want.iter().map(|&x| x as u32).sum();
    if g == 0 || w == 0 || g > max_cards as u32 || w > max_cards as u32 {
        return false;
    }
    (0..NUM_RESOURCES).all(|r| give[r] == 0 || want[r] == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_unique_and_well_formed() {
        let t = templates();
        for (i, (g, w)) in t.iter().enumerate() {
            assert!(is_well_formed(g, w, 6));
            assert_eq!(template_index(g, w), Some(i));
        }
    }
}
