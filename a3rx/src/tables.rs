// SPDX-License-Identifier: AGPL-3.0-or-later
//! The flat tables written by tv3b/export_tables.py.
use num_complex::Complex32 as C32;
use serde_json::Value;
use std::path::Path;

fn bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bin"))).unwrap_or_else(|e| panic!("table {name}: {e}"))
}
pub fn u32s(dir: &Path, name: &str) -> Vec<u32> {
    bytes(dir, name).chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
pub fn u16s(dir: &Path, name: &str) -> Vec<u16> {
    bytes(dir, name).chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}
pub fn f32s(dir: &Path, name: &str) -> Vec<f32> {
    bytes(dir, name).chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
pub fn c32s(dir: &Path, name: &str) -> Vec<C32> {
    f32s(dir, name).chunks_exact(2).map(|v| C32::new(v[0], v[1])).collect()
}
pub fn u8s(dir: &Path, name: &str) -> Vec<u8> {
    bytes(dir, name)
}

pub struct Plp {
    pub id: u8,
    pub bits: usize,
    pub blocks: usize,
    pub cells: usize,
    pub src: Vec<u32>,
    pub u2cw: Vec<u32>,
    pub constellation: Vec<C32>,
}

pub struct Pattern {
    pub kk: Vec<u16>,
    /// taps x pilots, row major
    pub p: Vec<C32>,
}

pub struct Tables {
    pub frame_len: usize,
    pub bs_len: usize,
    pub nfft: usize,
    pub gi: usize,
    pub sym: usize,
    pub noc: usize,
    pub nsym: usize,
    pub backoff: usize,
    pub pil_ptr: Vec<u32>,
    pub pil_car: Vec<u16>,
    pub pil_ref: Vec<f32>,
    pub pil_amp: Vec<f32>,
    pub group_first: Vec<u32>,
    pub group_n: Vec<u32>,
    pub group_pat: Vec<u32>,
    pub patterns: Vec<Pattern>,
    pub taus: Vec<f32>,
    pub plps: Vec<Plp>,
    pub row_ptr: Vec<u32>,
    pub col: Vec<u32>,
    pub n: usize,
    pub k: usize,
    pub kbch: usize,
    pub bch_gen: Vec<u8>,
    pub scramble: Vec<u8>,
    pub bootstrap: Vec<C32>,
}

impl Tables {
    pub fn load(dir: &Path) -> Self {
        let man: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest.json")).unwrap();
        let fr = &man["frame"];
        let g = |k: &str| fr[k].as_u64().unwrap() as usize;
        let patterns = (0..man["patterns"].as_array().unwrap().len())
            .map(|i| Pattern { kk: u16s(dir, &format!("pat{i}_kk")), p: c32s(dir, &format!("pat{i}_P")) })
            .collect();
        let plps = man["plps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let id = p["id"].as_u64().unwrap();
                Plp {
                    id: id as u8,
                    bits: p["mod_bits"].as_u64().unwrap() as usize,
                    blocks: p["fec_blocks"].as_u64().unwrap() as usize,
                    cells: p["fec_cells"].as_u64().unwrap() as usize,
                    src: u32s(dir, &format!("plp{id}_src")),
                    u2cw: u32s(dir, &format!("plp{id}_u2cw")),
                    constellation: c32s(dir, &format!("plp{id}_const")),
                }
            })
            .collect();
        Tables {
            frame_len: g("frame_len"),
            bs_len: g("bs_len"),
            nfft: g("nfft"),
            gi: g("gi"),
            sym: g("sym"),
            noc: g("noc"),
            nsym: g("nsym"),
            backoff: g("backoff"),
            pil_ptr: u32s(dir, "pil_ptr"),
            pil_car: u16s(dir, "pil_car"),
            pil_ref: f32s(dir, "pil_ref"),
            pil_amp: f32s(dir, "pil_amp"),
            group_first: u32s(dir, "group_first_sym"),
            group_n: u32s(dir, "group_nsym"),
            group_pat: u32s(dir, "group_pattern"),
            patterns,
            taus: f32s(dir, "taus"),
            plps,
            row_ptr: u32s(dir, "ldpc_row_ptr"),
            col: u32s(dir, "ldpc_col"),
            n: man["ldpc"]["N"].as_u64().unwrap() as usize,
            k: man["ldpc"]["K"].as_u64().unwrap() as usize,
            kbch: man["plps"][0]["Kbch"].as_u64().unwrap() as usize,
            bch_gen: u8s(dir, "bch_gen"),
            scramble: u8s(dir, "bb_scramble"),
            bootstrap: c32s(dir, "bootstrap_ref"),
        }
    }
}
