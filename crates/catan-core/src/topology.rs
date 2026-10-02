//! Static board topology for the standard 19-hex board.
//!
//! Hexes are pointy-topped and use axial coordinates `(q, r)` with radius 2.
//! Hex ids are assigned row by row (top row first, left to right), which gives
//! the familiar 3-4-5-4-3 layout. Vertices (54) and edges (72) are derived from
//! hex corners, deduplicated, and sorted top-to-bottom / left-to-right so that ids
//! are stable and human-readable.
//!
//! All adjacency information is precomputed once into plain arrays and bitmasks
//! (`u64` for vertices, `u128` for edges) so that the hot paths of the engine are
//! branch-light lookups.

use crate::types::{NUM_EDGES, NUM_HEXES, NUM_PORTS, NUM_VERTICES};
use std::sync::OnceLock;

/// Sentinel for "no entry" in fixed-size adjacency arrays.
pub const NONE: u8 = u8::MAX;

#[derive(Debug)]
pub struct Topology {
    /// Axial coordinates of each hex.
    pub hex_coords: [(i8, i8); NUM_HEXES],
    /// Pixel center of each hex for a hex of circumradius 1 (y grows downward).
    pub hex_center: [(f32, f32); NUM_HEXES],
    /// Vertices of each hex, clockwise starting from the top corner.
    pub hex_vertices: [[u8; 6]; NUM_HEXES],
    /// Edges of each hex; edge `i` joins corner `i` and corner `i+1`.
    pub hex_edges: [[u8; 6]; NUM_HEXES],
    /// Neighboring hexes (`NONE` padded).
    pub hex_neighbors: [[u8; 6]; NUM_HEXES],

    pub vertex_pos: [(f32, f32); NUM_VERTICES],
    /// Hexes touching each vertex (`NONE` padded, 1..=3 entries).
    pub vertex_hexes: [[u8; 3]; NUM_VERTICES],
    /// Edges touching each vertex (`NONE` padded, 2..=3 entries).
    pub vertex_edges: [[u8; 3]; NUM_VERTICES],
    /// Adjacent vertices (`NONE` padded, 2..=3 entries).
    pub vertex_neighbors: [[u8; 3]; NUM_VERTICES],
    /// Bitmask of adjacent vertices.
    pub vertex_neighbor_mask: [u64; NUM_VERTICES],
    /// Bitmask of the vertex itself plus its neighbors (the distance-rule footprint).
    pub vertex_footprint: [u64; NUM_VERTICES],
    /// Bitmask of edges touching each vertex.
    pub vertex_edge_mask: [u128; NUM_VERTICES],

    pub edge_vertices: [[u8; 2]; NUM_EDGES],
    pub edge_vertex_mask: [u64; NUM_EDGES],

    /// Coastal edges ordered clockwise starting from the top of the board.
    pub coastal_edges: Vec<u8>,
    /// Coastal edges that host harbors (standard spacing).
    pub port_edges: [u8; NUM_PORTS],
}

impl Topology {
    #[inline]
    pub fn edge_other(&self, e: u8, v: u8) -> u8 {
        let [a, b] = self.edge_vertices[e as usize];
        if a == v {
            b
        } else {
            a
        }
    }

    fn build() -> Topology {
        let sqrt3 = 3f64.sqrt();

        // Hex coordinates, row by row.
        let mut hex_coords = Vec::with_capacity(NUM_HEXES);
        for r in -2i8..=2 {
            let q_min = (-2).max(-2 - r);
            let q_max = 2.min(2 - r);
            for q in q_min..=q_max {
                hex_coords.push((q, r));
            }
        }
        assert_eq!(hex_coords.len(), NUM_HEXES);

        let centers: Vec<(f64, f64)> = hex_coords
            .iter()
            .map(|&(q, r)| (sqrt3 * (q as f64 + r as f64 / 2.0), 1.5 * r as f64))
            .collect();

        let key = |x: f64, y: f64| -> (i64, i64) { ((y * 1000.0).round() as i64, (x * 1000.0).round() as i64) };

        // Collect unique corners.
        let mut corner_keys: Vec<(i64, i64)> = Vec::new();
        let mut corner_pos: Vec<(f64, f64)> = Vec::new();
        let mut hex_corner_keys = vec![[(0i64, 0i64); 6]; NUM_HEXES];
        for (h, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..6 {
                let ang = (60.0 * i as f64 - 90.0).to_radians();
                let (x, y) = (cx + ang.cos(), cy + ang.sin());
                let k = key(x, y);
                hex_corner_keys[h][i] = k;
                if !corner_keys.contains(&k) {
                    corner_keys.push(k);
                    corner_pos.push((x, y));
                }
            }
        }
        assert_eq!(corner_keys.len(), NUM_VERTICES);

        // Sort vertices top-to-bottom, left-to-right.
        let mut order: Vec<usize> = (0..NUM_VERTICES).collect();
        order.sort_by_key(|&i| corner_keys[i]);
        let sorted_keys: Vec<(i64, i64)> = order.iter().map(|&i| corner_keys[i]).collect();
        let vertex_id = |k: (i64, i64)| -> u8 { sorted_keys.iter().position(|&x| x == k).unwrap() as u8 };

        let mut vertex_pos = [(0f32, 0f32); NUM_VERTICES];
        for (new_id, &old) in order.iter().enumerate() {
            vertex_pos[new_id] = (corner_pos[old].0 as f32, corner_pos[old].1 as f32);
        }

        let mut hex_vertices = [[0u8; 6]; NUM_HEXES];
        for h in 0..NUM_HEXES {
            for i in 0..6 {
                hex_vertices[h][i] = vertex_id(hex_corner_keys[h][i]);
            }
        }

        // Edges as sorted vertex pairs.
        let mut edge_pairs: Vec<(u8, u8)> = Vec::new();
        for hv in hex_vertices.iter() {
            for i in 0..6 {
                let (a, b) = (hv[i], hv[(i + 1) % 6]);
                let p = (a.min(b), a.max(b));
                if !edge_pairs.contains(&p) {
                    edge_pairs.push(p);
                }
            }
        }
        assert_eq!(edge_pairs.len(), NUM_EDGES);
        let midpoint_key = |p: &(u8, u8)| {
            let (ax, ay) = vertex_pos[p.0 as usize];
            let (bx, by) = vertex_pos[p.1 as usize];
            key(((ax + bx) / 2.0) as f64, ((ay + by) / 2.0) as f64)
        };
        edge_pairs.sort_by_key(midpoint_key);
        let edge_id = |a: u8, b: u8| -> u8 {
            let p = (a.min(b), a.max(b));
            edge_pairs.iter().position(|&x| x == p).unwrap() as u8
        };

        let mut hex_edges = [[0u8; 6]; NUM_HEXES];
        for h in 0..NUM_HEXES {
            for i in 0..6 {
                hex_edges[h][i] = edge_id(hex_vertices[h][i], hex_vertices[h][(i + 1) % 6]);
            }
        }

        let mut edge_vertices = [[0u8; 2]; NUM_EDGES];
        let mut edge_vertex_mask = [0u64; NUM_EDGES];
        for (e, &(a, b)) in edge_pairs.iter().enumerate() {
            edge_vertices[e] = [a, b];
            edge_vertex_mask[e] = (1u64 << a) | (1u64 << b);
        }

        let mut vertex_hexes = [[NONE; 3]; NUM_VERTICES];
        for h in 0..NUM_HEXES {
            for &v in &hex_vertices[h] {
                let slot = vertex_hexes[v as usize].iter().position(|&x| x == NONE).unwrap();
                vertex_hexes[v as usize][slot] = h as u8;
            }
        }

        let mut vertex_edges = [[NONE; 3]; NUM_VERTICES];
        let mut vertex_neighbors = [[NONE; 3]; NUM_VERTICES];
        let mut vertex_neighbor_mask = [0u64; NUM_VERTICES];
        let mut vertex_edge_mask = [0u128; NUM_VERTICES];
        for (e, &[a, b]) in edge_vertices.iter().enumerate() {
            for (v, o) in [(a, b), (b, a)] {
                let vi = v as usize;
                let slot = vertex_edges[vi].iter().position(|&x| x == NONE).unwrap();
                vertex_edges[vi][slot] = e as u8;
                vertex_neighbors[vi][slot] = o;
                vertex_neighbor_mask[vi] |= 1u64 << o;
                vertex_edge_mask[vi] |= 1u128 << e;
            }
        }
        let mut vertex_footprint = [0u64; NUM_VERTICES];
        for v in 0..NUM_VERTICES {
            vertex_footprint[v] = vertex_neighbor_mask[v] | (1u64 << v);
        }

        let mut hex_neighbors = [[NONE; 6]; NUM_HEXES];
        for h in 0..NUM_HEXES {
            let (q, r) = hex_coords[h];
            let mut n = 0;
            for (dq, dr) in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)] {
                if let Some(o) = hex_coords.iter().position(|&c| c == (q + dq, r + dr)) {
                    hex_neighbors[h][n] = o as u8;
                    n += 1;
                }
            }
        }

        // Coastal edges belong to exactly one hex; order them clockwise from the top.
        let mut edge_hex_count = [0u8; NUM_EDGES];
        for he in hex_edges.iter() {
            for &e in he {
                edge_hex_count[e as usize] += 1;
            }
        }
        let mut coastal: Vec<u8> = (0..NUM_EDGES as u8)
            .filter(|&e| edge_hex_count[e as usize] == 1)
            .collect();
        assert_eq!(coastal.len(), 30);
        let angle = |e: u8| -> f64 {
            let [a, b] = edge_vertices[e as usize];
            let (ax, ay) = vertex_pos[a as usize];
            let (bx, by) = vertex_pos[b as usize];
            let (mx, my) = (((ax + bx) / 2.0) as f64, ((ay + by) / 2.0) as f64);
            // Clockwise from straight up (screen coordinates: y grows downward).
            let a = mx.atan2(-my);
            if a < -1e-9 {
                a + std::f64::consts::TAU
            } else {
                a
            }
        };
        coastal.sort_by(|&a, &b| angle(a).partial_cmp(&angle(b)).unwrap());

        // Standard harbor spacing around the 30 coastal edges.
        let gaps = [3usize, 3, 4, 3, 3, 4, 3, 3, 4];
        let mut port_edges = [0u8; NUM_PORTS];
        let mut idx = 0usize;
        for (i, g) in gaps.iter().enumerate() {
            port_edges[i] = coastal[idx];
            idx += g;
        }

        let hex_center_f32 = {
            let mut out = [(0f32, 0f32); NUM_HEXES];
            for (i, c) in centers.iter().enumerate() {
                out[i] = (c.0 as f32, c.1 as f32);
            }
            out
        };
        let hex_coords_arr = {
            let mut out = [(0i8, 0i8); NUM_HEXES];
            out.copy_from_slice(&hex_coords);
            out
        };

        Topology {
            hex_coords: hex_coords_arr,
            hex_center: hex_center_f32,
            hex_vertices,
            hex_edges,
            hex_neighbors,
            vertex_pos,
            vertex_hexes,
            vertex_edges,
            vertex_neighbors,
            vertex_neighbor_mask,
            vertex_footprint,
            vertex_edge_mask,
            edge_vertices,
            edge_vertex_mask,
            coastal_edges: coastal,
            port_edges,
        }
    }
}

static TOPOLOGY: OnceLock<Topology> = OnceLock::new();

/// Global, lazily-built board topology.
#[inline]
pub fn topo() -> &'static Topology {
    TOPOLOGY.get_or_init(Topology::build)
}

/// Iterate the indices of set bits in a `u64`.
#[inline]
pub fn bits64(mut m: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let i = m.trailing_zeros() as u8;
            m &= m - 1;
            Some(i)
        }
    })
}

/// Iterate the indices of set bits in a `u128`.
#[inline]
pub fn bits128(mut m: u128) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let i = m.trailing_zeros() as u8;
            m &= m - 1;
            Some(i)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_degrees() {
        let t = topo();
        let mut deg2 = 0;
        let mut deg3 = 0;
        for v in 0..NUM_VERTICES {
            let d = t.vertex_edges[v].iter().filter(|&&e| e != NONE).count();
            match d {
                2 => deg2 += 1,
                3 => deg3 += 1,
                _ => panic!("vertex {v} has degree {d}"),
            }
            assert_eq!(t.vertex_neighbor_mask[v].count_ones() as usize, d);
            assert_eq!(t.vertex_edge_mask[v].count_ones() as usize, d);
        }
        // 30 coastal vertices with 2 edges... (18 coast vertices touch one hex and have degree 2)
        assert_eq!(deg2 + deg3, NUM_VERTICES);
        assert_eq!(deg2, 18);
        let hex_per_vertex: usize = (0..NUM_VERTICES)
            .map(|v| t.vertex_hexes[v].iter().filter(|&&h| h != NONE).count())
            .sum();
        assert_eq!(hex_per_vertex, NUM_HEXES * 6);
    }

    #[test]
    fn ports_do_not_share_vertices() {
        let t = topo();
        let mut seen = 0u64;
        for &e in &t.port_edges {
            let m = t.edge_vertex_mask[e as usize];
            assert_eq!(seen & m, 0);
            seen |= m;
        }
    }

    #[test]
    fn hex_neighbors_symmetric() {
        let t = topo();
        for h in 0..NUM_HEXES {
            for &n in t.hex_neighbors[h].iter().filter(|&&n| n != NONE) {
                assert!(t.hex_neighbors[n as usize].contains(&(h as u8)));
            }
        }
        // The center hex has 6 neighbors.
        assert_eq!(t.hex_neighbors[9].iter().filter(|&&n| n != NONE).count(), 6);
    }
}
