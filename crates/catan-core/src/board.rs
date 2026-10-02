//! Board layout: terrain, number tokens and harbors.

use crate::rng::Rng;
use crate::topology::{topo, NONE};
use crate::types::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    pub terrain: [Terrain; NUM_HEXES],
    /// Number token on each hex (0 for the desert).
    pub numbers: [u8; NUM_HEXES],
    /// Harbor kind at each of the standard harbor edges (see [`crate::topology::Topology::port_edges`]).
    pub ports: [PortKind; NUM_PORTS],
    /// Harbor reachable from each vertex, if any.
    #[serde(with = "crate::serde_arr")]
    pub vertex_port: [Option<PortKind>; NUM_VERTICES],
    /// For each dice total (index 2..=12), bitmask of hexes carrying that number.
    pub number_hexes: [u32; 13],
}

impl Board {
    /// Randomized board following the base-game distribution. Red numbers (6 and 8)
    /// are never placed on adjacent hexes.
    pub fn random(rng: &mut Rng) -> Board {
        let mut terrains: Vec<Terrain> = Vec::with_capacity(NUM_HEXES);
        for &(t, n) in TERRAIN_COUNTS.iter() {
            for _ in 0..n {
                terrains.push(t);
            }
        }
        rng.shuffle(&mut terrains);
        let mut terrain = [Terrain::Desert; NUM_HEXES];
        terrain.copy_from_slice(&terrains);

        let t = topo();
        let mut numbers = [0u8; NUM_HEXES];
        let mut tokens = NUMBER_TOKENS;
        loop {
            rng.shuffle(&mut tokens);
            let mut k = 0;
            for h in 0..NUM_HEXES {
                if terrain[h] == Terrain::Desert {
                    numbers[h] = 0;
                } else {
                    numbers[h] = tokens[k];
                    k += 1;
                }
            }
            let red = |n: u8| n == 6 || n == 8;
            let ok = (0..NUM_HEXES).all(|h| {
                !red(numbers[h])
                    || t.hex_neighbors[h]
                        .iter()
                        .filter(|&&n| n != NONE)
                        .all(|&n| !red(numbers[n as usize]))
            });
            if ok {
                break;
            }
        }

        let mut ports = PortKind::STANDARD_SET;
        rng.shuffle(&mut ports);
        Board::from_parts(terrain, numbers, ports)
    }

    /// Build a board from explicit parts (used for fixed layouts and deserialization helpers).
    pub fn from_parts(terrain: [Terrain; NUM_HEXES], numbers: [u8; NUM_HEXES], ports: [PortKind; NUM_PORTS]) -> Board {
        let t = topo();
        let mut vertex_port = [None; NUM_VERTICES];
        for (i, &e) in t.port_edges.iter().enumerate() {
            for &v in &t.edge_vertices[e as usize] {
                vertex_port[v as usize] = Some(ports[i]);
            }
        }
        let mut number_hexes = [0u32; 13];
        for h in 0..NUM_HEXES {
            if numbers[h] != 0 {
                number_hexes[numbers[h] as usize] |= 1 << h;
            }
        }
        Board {
            terrain,
            numbers,
            ports,
            vertex_port,
            number_hexes,
        }
    }

    /// The suggested "beginner" layout from the base-game rulebook (fixed terrain/numbers/harbors).
    pub fn beginner() -> Board {
        use Terrain::*;
        // Row by row: 3-4-5-4-3.
        let terrain = [
            Mountains, Pasture, Forest, //
            Fields, Hills, Pasture, Hills, //
            Fields, Forest, Desert, Forest, Mountains, //
            Forest, Mountains, Fields, Pasture, //
            Hills, Fields, Pasture,
        ];
        let numbers = [
            10, 2, 9, //
            12, 6, 4, 10, //
            9, 11, 0, 3, 8, //
            8, 3, 4, 5, //
            5, 6, 11,
        ];
        let ports = [
            PortKind::Generic,
            PortKind::Sheep,
            PortKind::Generic,
            PortKind::Generic,
            PortKind::Brick,
            PortKind::Wood,
            PortKind::Generic,
            PortKind::Wheat,
            PortKind::Ore,
        ];
        Board::from_parts(terrain, numbers, ports)
    }

    #[inline]
    pub fn desert(&self) -> u8 {
        self.terrain.iter().position(|&t| t == Terrain::Desert).unwrap_or(0) as u8
    }

    /// Resource produced by a hex (None for the desert).
    #[inline]
    pub fn hex_resource(&self, h: usize) -> Option<Resource> {
        self.terrain[h].resource()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_board_is_valid() {
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let b = Board::random(&mut rng);
            assert_eq!(b.terrain.iter().filter(|&&t| t == Terrain::Desert).count(), 1);
            let mut nums: Vec<u8> = b.numbers.iter().copied().filter(|&n| n != 0).collect();
            nums.sort();
            assert_eq!(nums, NUMBER_TOKENS.to_vec());
            assert_eq!(b.vertex_port.iter().filter(|p| p.is_some()).count(), 18);
        }
    }

    #[test]
    fn beginner_board_has_standard_counts() {
        let b = Board::beginner();
        let mut nums: Vec<u8> = b.numbers.iter().copied().filter(|&n| n != 0).collect();
        nums.sort();
        assert_eq!(nums, NUMBER_TOKENS.to_vec());
        for &(t, n) in TERRAIN_COUNTS.iter() {
            assert_eq!(b.terrain.iter().filter(|&&x| x == t).count(), n as usize);
        }
    }
}
