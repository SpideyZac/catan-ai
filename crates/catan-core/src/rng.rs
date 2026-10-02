//! Small, fast, serializable PRNG (xoshiro256++ seeded through SplitMix64).
//!
//! We avoid the `rand` crate so that the full game state (including the RNG)
//! can be cloned and serialized cheaply and deterministically.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` (Lemire's multiply-shift; bias is negligible for small `n`).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    pub fn roll_die(&mut self) -> u8 {
        self.below(6) as u8 + 1
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }

    /// Pick an index with probability proportional to integer weights. Returns `None` if all are zero.
    pub fn weighted_index(&mut self, weights: &[u32]) -> Option<usize> {
        let total: u32 = weights.iter().sum();
        if total == 0 {
            return None;
        }
        let mut x = self.below(total);
        for (i, &w) in weights.iter().enumerate() {
            if x < w {
                return Some(i);
            }
            x -= w;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn dice_in_range_and_roughly_uniform() {
        let mut r = Rng::new(7);
        let mut counts = [0u32; 7];
        for _ in 0..60_000 {
            counts[r.roll_die() as usize] += 1;
        }
        assert_eq!(counts[0], 0);
        for &c in &counts[1..] {
            assert!((9_000..11_000).contains(&c), "{counts:?}");
        }
    }
}
