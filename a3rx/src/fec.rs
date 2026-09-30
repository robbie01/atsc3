// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demapping, LDPC and the BCH check for one FEC block.
use crate::tables::Tables;
use num_complex::Complex32 as C32;

/// Max-log demapper as a lookup table over the I/Q plane: per grid point and bit, the difference of the
/// squared distances to the nearest constellation point with that bit 1 and with that bit 0.
pub struct Demap {
    bits: usize,
    grid: usize,
    range: f32,
    lut: Vec<f32>,
}

impl Demap {
    pub fn new(points: &[C32], bits: usize) -> Self {
        let grid = 512usize;
        let range = points.iter().map(|p| p.re.abs().max(p.im.abs())).fold(0f32, f32::max) * 1.25;
        let mut lut = vec![0f32; grid * grid * bits];
        for gy in 0..grid {
            for gx in 0..grid {
                let x = (gx as f32 + 0.5) / grid as f32 * 2.0 * range - range;
                let y = (gy as f32 + 0.5) / grid as f32 * 2.0 * range - range;
                let mut m0 = [f32::MAX; 12];
                let mut m1 = [f32::MAX; 12];
                for (i, p) in points.iter().enumerate() {
                    let d = (x - p.re) * (x - p.re) + (y - p.im) * (y - p.im);
                    for b in 0..bits {
                        if (i >> (bits - 1 - b)) & 1 == 1 {
                            if d < m1[b] { m1[b] = d }
                        } else if d < m0[b] {
                            m0[b] = d
                        }
                    }
                }
                let o = (gy * grid + gx) * bits;
                for b in 0..bits {
                    lut[o + b] = m1[b] - m0[b];
                }
            }
        }
        Demap { bits, grid, range, lut }
    }

    #[inline]
    pub fn llr(&self, cell: C32, inv_nv: f32, out: &mut [f32]) {
        let s = self.grid as f32 / (2.0 * self.range);
        let gx = (((cell.re + self.range) * s) as isize).clamp(0, self.grid as isize - 1) as usize;
        let gy = (((cell.im + self.range) * s) as isize).clamp(0, self.grid as isize - 1) as usize;
        let o = (gy * self.grid + gx) * self.bits;
        for b in 0..self.bits {
            out[b] = self.lut[o + b] * inv_nv;
        }
    }
}

pub struct Ldpc<'a> {
    row_ptr: &'a [u32],
    col: &'a [u32],
    n: usize,
}

pub struct Work {
    pub l: Vec<f32>,
    pub chan: Vec<f32>,
    c2v: Vec<f32>,
}
impl Work {
    pub fn new(t: &Tables) -> Self {
        Work { l: vec![0.0; t.n], chan: vec![0.0; t.n], c2v: vec![0.0; t.col.len()] }
    }
}

impl<'a> Ldpc<'a> {
    pub fn new(t: &'a Tables) -> Self {
        Ldpc { row_ptr: &t.row_ptr, col: &t.col, n: t.n }
    }

    fn satisfied(&self, l: &[f32]) -> bool {
        for r in 0..self.row_ptr.len() - 1 {
            let mut p = 0u32;
            for e in self.row_ptr[r] as usize..self.row_ptr[r + 1] as usize {
                p ^= (l[self.col[e] as usize] < 0.0) as u32;
            }
            if p != 0 {
                return false;
            }
        }
        true
    }

    /// Layered normalised min-sum.  `w.l` holds the channel LLRs on entry (positive = 0) and the
    /// posterior on exit.  Returns (converged, iterations).
    pub fn decode(&self, w: &mut Work, max_iter: usize, alpha: f32) -> (bool, usize) {
        debug_assert_eq!(w.l.len(), self.n);
        #[allow(non_snake_case)]
        let ALPHA = alpha;
        const LIM: f32 = 48.0;
        if self.satisfied(&w.l) {
            return (true, 0);
        }
        w.c2v.iter_mut().for_each(|v| *v = 0.0);
        let mut tmp = [0f32; 64];
        for it in 1..=max_iter {
            for r in 0..self.row_ptr.len() - 1 {
                let (a, b) = (self.row_ptr[r] as usize, self.row_ptr[r + 1] as usize);
                let d = b - a;
                let (mut m1, mut m2, mut at, mut neg) = (f32::MAX, f32::MAX, 0usize, false);
                for i in 0..d {
                    let v = w.l[self.col[a + i] as usize] - w.c2v[a + i];
                    tmp[i] = v;
                    let m = v.abs();
                    if m < m1 {
                        m2 = m1;
                        m1 = m;
                        at = i;
                    } else if m < m2 {
                        m2 = m;
                    }
                    neg ^= v < 0.0;
                }
                for i in 0..d {
                    let mag = ALPHA * if i == at { m2 } else { m1 };
                    let s = neg ^ (tmp[i] < 0.0);
                    let m = if s { -mag } else { mag };
                    w.c2v[a + i] = m;
                    w.l[self.col[a + i] as usize] = (tmp[i] + m).clamp(-LIM, LIM);
                }
            }
            if self.satisfied(&w.l) {
                return (true, it);
            }
        }
        (false, max_iter)
    }
}

/// Remainder of the first `k` hard decisions modulo the BCH generator; zero for a valid codeword.
pub fn bch_ok(bits: &[u8], gen: &[u8]) -> bool {
    let deg = gen.len() - 1;
    let words = deg / 64 + 1;
    let mut g = vec![0u64; words];
    for (i, &v) in gen.iter().enumerate() {
        if v != 0 {
            g[i / 64] |= 1 << (i % 64);
        }
    }
    let mut reg = vec![0u64; words];
    let (tw, tb) = (deg / 64, deg % 64);
    for &b in bits {
        let mut carry = b as u64 & 1;
        for w in reg.iter_mut() {
            let n = *w >> 63;
            *w = (*w << 1) | carry;
            carry = n;
        }
        if (reg[tw] >> tb) & 1 == 1 {
            for (w, gw) in reg.iter_mut().zip(&g) {
                *w ^= gw;
            }
        }
    }
    reg.iter().all(|&w| w == 0)
}
