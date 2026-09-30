// SPDX-License-Identifier: AGPL-3.0-or-later
//! a3rx -- real-time ATSC 3.0 receiver for one station's configuration.
//!
//!     a3rx --file capture.ci16                 decode a 6.912 MS/s recording
//!     a3rx --live --freq 527008711 --gain 34   decode from the radio
//!
//! stdout: records `magic[4] length:u32le body`.  "BBP0": frame:u32 plp:u8 block:u8 ok:u8 iters:u8 then the
//! baseband packet; "STAT": one JSON object per frame.  Everything about the waveform comes from tables/.
mod fec;
mod tables;

use fec::{bch_ok, Demap, Ldpc, Work};
use num_complex::Complex32 as C32;
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use tables::Tables;

const FS: f32 = 6_912_000.0;
const LLR_CLIP: f32 = 12.0;
const MAX_ITER: usize = 25;

struct Args {
    file: Option<String>,
    tables: PathBuf,
    freq: u64,
    rate: u32,
    gain: f64,
    max_frames: usize,
}

fn args() -> Args {
    let mut a = Args { file: None, tables: std::env::var_os("A3RX_TABLES").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tables")), freq: 527_008_711, rate: 6_912_114, gain: 34.0, max_frames: usize::MAX };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let next = |i: usize| v.get(i + 1).cloned().unwrap_or_else(|| panic!("{} needs a value", v[i]));
        match v[i].as_str() {
            "--file" => { a.file = Some(next(i)); i += 1 }
            "--tables" => { a.tables = next(i).into(); i += 1 }
            "--freq" => { a.freq = next(i).parse().unwrap(); i += 1 }
            "--rate" => { a.rate = next(i).parse().unwrap(); i += 1 }
            "--gain" => { a.gain = next(i).parse().unwrap(); i += 1 }
            "--frames" => { a.max_frames = next(i).parse().unwrap(); i += 1 }
            "--live" => {}
            o => panic!("unknown option {o}"),
        }
        i += 1;
    }
    a
}

/// Samples with an absolute index, fed by a file or by the radio.
struct Input {
    buf: Vec<C32>,
    base: u64,
    rx: Option<mpsc::Receiver<Vec<u8>>>,
    ended: bool,
    dropped: u64,
}

impl Input {
    fn push(&mut self, raw: &[u8]) {
        self.buf.extend(raw.chunks_exact(4).map(|s| C32::new(i16::from_le_bytes([s[0], s[1]]) as f32, i16::from_le_bytes([s[2], s[3]]) as f32)));
    }
    /// Make [from, from+n) available; false at the end of the input.
    fn need(&mut self, from: u64, n: usize) -> bool {
        if from > self.base {
            let cut = ((from - self.base) as usize).min(self.buf.len());
            if cut > 1 << 22 {
                self.buf.drain(..cut);
                self.base += cut as u64;
            }
        }
        while self.base + (self.buf.len() as u64) < from + n as u64 {
            if self.ended {
                return false;
            }
            match self.rx.as_ref().map(|r| r.recv()) {
                Some(Ok(b)) => self.push(&b),
                _ => {
                    self.ended = true;
                    return false;
                }
            }
        }
        true
    }
    /// Take whatever has already arrived, without waiting.
    fn drain_pending(&mut self) {
        let pending: Vec<Vec<u8>> = self.rx.as_ref().map(|rx| rx.try_iter().collect()).unwrap_or_default();
        for b in pending {
            self.push(&b);
        }
    }
    fn slice(&self, from: u64, n: usize) -> &[C32] {
        let o = (from - self.base) as usize;
        &self.buf[o..o + n]
    }
    fn end(&self) -> u64 {
        self.base + self.buf.len() as u64
    }
}

fn corr(x: &[C32], r: &[C32], er: f32) -> f32 {
    let mut acc = C32::new(0.0, 0.0);
    let mut e = 0f32;
    for (a, b) in x.iter().zip(r) {
        acc += a * b.conj();
        e += a.norm_sqr();
    }
    acc.norm_sqr() / (er * e).max(1e-20)
}

struct Group {
    hfit: Vec<C32>,
    n0: f32,
    snr_db: f32,
}

struct Rx {
    t: Tables,
    ramp: Vec<C32>,
    /// carriers x taps
    basis: Vec<C32>,
    /// per pattern: pilots x taps
    basis_k: Vec<Vec<C32>>,
    pos: Vec<Vec<i32>>,
    group_of: Vec<usize>,
    demap: Vec<Demap>,
    eref: f32,
}

impl Rx {
    fn new(t: Tables) -> Self {
        let half = (t.noc / 2) as i32;
        let ramp = (0..t.noc).map(|c| C32::from_polar(1.0, std::f32::consts::TAU * ((c as i32 - half) * t.backoff as i32) as f32 / t.nfft as f32)).collect();
        let nt = t.taus.len();
        let ph = |c: i32, tau: f32| {
            let x = -(c as f64) * tau as f64 / t.nfft as f64;
            let x = (x - x.floor()) * std::f64::consts::TAU;
            C32::new(x.cos() as f32, x.sin() as f32)
        };
        let mut basis = vec![C32::default(); t.noc * nt];
        for c in 0..t.noc {
            for (j, &tau) in t.taus.iter().enumerate() {
                basis[c * nt + j] = ph(c as i32 - half, tau);
            }
        }
        let mut basis_k = Vec::new();
        let mut pos = Vec::new();
        for p in &t.patterns {
            let mut b = vec![C32::default(); p.kk.len() * nt];
            let mut m = vec![-1i32; t.noc];
            for (i, &c) in p.kk.iter().enumerate() {
                m[c as usize] = i as i32;
                b[i * nt..(i + 1) * nt].copy_from_slice(&basis[c as usize * nt..(c as usize + 1) * nt]);
            }
            basis_k.push(b);
            pos.push(m);
        }
        let mut group_of = vec![0usize; t.nsym];
        for g in 0..t.group_first.len() {
            for s in t.group_first[g]..t.group_first[g] + t.group_n[g] {
                group_of[s as usize] = g;
            }
        }
        let demap = t.plps.iter().map(|p| Demap::new(&p.constellation, p.bits)).collect();
        let eref = t.bootstrap.iter().map(|v| v.norm_sqr()).sum();
        Rx { t, ramp, basis, basis_k, pos, group_of, demap, eref }
    }

    /// Best bootstrap position in [from, from+span), and its normalised correlation.
    fn search(&self, inp: &Input, from: u64, span: usize) -> (u64, f32) {
        let r = &self.t.bootstrap;
        let x = inp.slice(from, span + r.len());
        let best = (0..span)
            .into_par_iter()
            .map(|i| (i, corr(&x[i..i + r.len()], r, self.eref)))
            .reduce(|| (0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        (from + best.0 as u64, best.1)
    }

    fn estimate(&self, g: usize, y: &[C32], ph: &mut [C32]) -> Group {
        let t = &self.t;
        let (s0, n) = (t.group_first[g] as usize, t.group_n[g] as usize);
        let pi = t.group_pat[g] as usize;
        let pat = &t.patterns[pi];
        let (nk, nt) = (pat.kk.len(), t.taus.len());
        let pos = &self.pos[pi];
        let bk = &self.basis_k[pi];
        let obs: Vec<Vec<(usize, C32, f32)>> = (0..n)
            .map(|i| {
                let s = s0 + i;
                (t.pil_ptr[s] as usize..t.pil_ptr[s + 1] as usize)
                    .map(|j| {
                        let c = t.pil_car[j] as usize;
                        (c, y[s * t.noc + c] * (t.pil_ref[j] / t.pil_amp[j]), t.pil_amp[j] * t.pil_amp[j])
                    })
                    .collect()
            })
            .collect();
        ph[..n].iter_mut().for_each(|p| *p = C32::new(1.0, 0.0));
        let mut a = vec![C32::default(); nt];
        let mut hk = vec![C32::default(); nk];
        let rounds = if n == 1 { 1 } else { 3 };
        for _ in 0..rounds {
            let mut num = vec![C32::default(); nk];
            let mut den = vec![0f32; nk];
            for (i, o) in obs.iter().enumerate() {
                for &(c, h, w) in o {
                    let k = pos[c] as usize;
                    num[k] += h * ph[i].conj() * w;
                    den[k] += w * ph[i].norm_sqr();
                }
            }
            for k in 0..nk {
                num[k] = if den[k] > 0.0 { num[k] / den[k] } else { C32::default() };
            }
            for j in 0..nt {
                a[j] = pat.p[j * nk..(j + 1) * nk].iter().zip(&num).map(|(p, h)| p * h).sum();
            }
            for k in 0..nk {
                hk[k] = bk[k * nt..(k + 1) * nt].iter().zip(&a).map(|(b, v)| b * v).sum();
            }
            if n > 1 {
                for (i, o) in obs.iter().enumerate() {
                    let (mut nu, mut de) = (C32::default(), 0f32);
                    for &(c, h, w) in o {
                        let f = hk[pos[c] as usize];
                        nu += f.conj() * h * w;
                        de += f.norm_sqr() * w;
                    }
                    ph[i] = nu / de.max(1e-20);
                }
            }
        }
        let (mut res, mut cnt) = (0f64, 0usize);
        for (i, o) in obs.iter().enumerate() {
            for &(c, h, w) in o {
                res += ((h - ph[i] * hk[pos[c] as usize]).norm_sqr() * w) as f64;
                cnt += 1;
            }
        }
        let n0 = if n == 1 { res / (cnt as f64 - nt as f64) } else { res / cnt as f64 } as f32;
        let hfit: Vec<C32> = (0..t.noc).map(|c| self.basis[c * nt..(c + 1) * nt].iter().zip(&a).map(|(b, v)| b * v).sum()).collect();
        let p = hfit.iter().map(|h| h.norm_sqr()).sum::<f32>() / t.noc as f32;
        Group { hfit, n0: n0.max(1e-12), snr_db: 10.0 * (p / n0.max(1e-12)).log10() }
    }

    /// Decode the frame whose bootstrap starts at x[0].  Returns the records and the statistics.
    fn frame(&self, x: &[C32], frame_no: u32, out: &mut Vec<u8>) -> String {
        let t = &self.t;
        let t0 = std::time::Instant::now();
        // carrier offset from the guard intervals
        let mut acc = C32::default();
        for l in 0..t.nsym {
            let s = t.bs_len + l * t.sym;
            for i in 64..t.gi {
                acc += x[s + i].conj() * x[s + i + t.nfft];
            }
        }
        let cfo = acc.arg() / std::f32::consts::TAU * FS / t.nfft as f32;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(t.nfft);
        let half = t.noc / 2;
        let scale = 1.0 / (t.nfft as f32).sqrt();
        let step = -(cfo as f64) / FS as f64;
        let y: Vec<C32> = (0..t.nsym)
            .into_par_iter()
            .flat_map_iter(|l| {
                let s = t.bs_len + l * t.sym + t.gi - t.backoff;
                let mut b: Vec<C32> = (0..t.nfft)
                    .map(|i| {
                        let p = ((s + i) as f64 * step).fract() * std::f64::consts::TAU;
                        x[s + i] * C32::new(p.cos() as f32, p.sin() as f32)
                    })
                    .collect();
                fft.process(&mut b);
                (0..t.noc).map(|c| b[(c + t.nfft - half) % t.nfft] * self.ramp[c] * scale).collect::<Vec<_>>()
            })
            .collect();
        let ng = t.group_first.len();
        let est: Vec<(Group, Vec<C32>)> = (0..ng)
            .into_par_iter()
            .map(|g| {
                let mut ph = vec![C32::default(); t.group_n[g] as usize];
                let gr = self.estimate(g, &y, &mut ph);
                (gr, ph)
            })
            .collect();
        let ldpc = Ldpc::new(t);
        let mut snrs: Vec<f32> = est[1..].iter().map(|e| e.0.snr_db).collect();
        snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let snr_med = snrs[snrs.len() / 2];
        // no point running the decoder to its iteration limit on every block of a frame that cannot decode:
        // that takes ~10 s a frame and stalls everything else.  16QAM 9/15 gives up near 9 dB, 256QAM 9/15 near 16 dB.
        let floor = [4.0f32, 11.0];
        let jobs: Vec<(usize, usize)> = t.plps.iter().enumerate().filter(|(p, _)| snr_med >= floor[(*p).min(1)])
            .flat_map(|(p, plp)| (0..plp.blocks).map(move |b| (p, b))).collect();
        let done: Vec<(usize, usize, bool, usize, Vec<u8>)> = jobs
            .par_iter()
            .map_init(
                || Work::new(t),
                |w, &(p, b)| {
                    let plp = &t.plps[p];
                    let dm = &self.demap[p];
                    let mut l8 = [0f32; 12];
                    for c in 0..plp.cells {
                        let idx = plp.src[b * plp.cells + c] as usize;
                        let (s, car) = (idx / t.noc, idx % t.noc);
                        let g = self.group_of[s];
                        let h = est[g].1[s - t.group_first[g] as usize] * est[g].0.hfit[car];
                        let hp = h.norm_sqr().max(1e-20);
                        dm.llr(y[idx] * h.conj() / hp, hp / est[g].0.n0, &mut l8);
                        for k in 0..plp.bits {
                            w.l[plp.u2cw[c * plp.bits + k] as usize] = l8[k].clamp(-LLR_CLIP, LLR_CLIP);
                        }
                    }
                    w.chan.copy_from_slice(&w.l);
                    let (mut conv, mut it) = ldpc.decode(w, MAX_ITER, 0.8);
                    for (alpha, scale) in [(0.9f32, 1.0f32), (0.7, 0.6), (1.0, 0.5)] {
                        if conv { break }
                        // rare: start again from the channel values with a different message scaling
                        for (l, c) in w.l.iter_mut().zip(&w.chan) { *l = c * scale }
                        let (c2, i2) = ldpc.decode(w, 60, alpha);
                        conv = c2;
                        it += i2;
                    }
                    let hard: Vec<u8> = w.l[..t.k].iter().map(|&v| (v < 0.0) as u8).collect();
                    let ok = conv && bch_ok(&hard, &t.bch_gen);
                    let mut bytes = vec![0u8; t.kbch / 8];
                    if ok {
                        for i in 0..t.kbch {
                            bytes[i / 8] |= (hard[i] ^ t.scramble[i]) << (7 - i % 8);
                        }
                    }
                    (p, b, ok, it, bytes)
                },
            )
            .collect();
        let mut okc = vec![0usize; t.plps.len()];
        let mut its = 0usize;
        for (p, b, ok, it, bytes) in &done {
            its += it;
            if *ok {
                okc[*p] += 1;
                out.extend_from_slice(b"BBP0");
                out.extend_from_slice(&((8 + bytes.len()) as u32).to_le_bytes());
                out.extend_from_slice(&frame_no.to_le_bytes());
                out.extend_from_slice(&[t.plps[*p].id, *b as u8, 1, *it as u8]);
                out.extend_from_slice(bytes);
            }
        }
        format!(
            "\"frame\":{frame_no},\"snr_db\":{:.2},\"snr_min\":{:.2},\"snr_preamble_db\":{:.2},\"cfo_hz\":{:.1},\"plp0_ok\":{},\"plp0_blocks\":{},\"plp1_ok\":{},\"plp1_blocks\":{},\"mean_iters\":{:.2},\"ms\":{}",
            snrs[snrs.len() / 2], snrs[0], est[0].0.snr_db, cfo, okc[0], t.plps[0].blocks, okc[1], t.plps[1].blocks,
            its as f32 / done.len().max(1) as f32, t0.elapsed().as_millis()
        )
    }
}

fn record(out: &mut Vec<u8>, magic: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(magic);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
}

fn main() {
    let a = args();
    // leave a core for the radio reader (and the rest of the machine)
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    rayon::ThreadPoolBuilder::new().num_threads((n - 1).max(2)).build_global().ok();
    let rx = Rx::new(Tables::load(&a.tables));
    let t = &rx.t;
    let live = a.file.is_none();
    let mut inp = Input { buf: Vec::new(), base: 0, rx: None, ended: false, dropped: 0 };
    let _radio_thread;
    if let Some(f) = &a.file {
        inp.push(&std::fs::read(f).expect("capture file"));
        inp.ended = true;
    } else {
        let (tx, rxc) = mpsc::sync_channel::<Vec<u8>>(256);
        let (freq, rate, gain) = (a.freq, a.rate, a.gain);
        _radio_thread = std::thread::spawn(move || {
            use pmpstream::{Format, Radio, StreamConfig, RX};
            // the reader must never wait behind the decoder or a busy Mac: user-interactive QoS, the highest a thread can ask for
            unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0); }
            let radio = Radio::open(None).expect("radio");
            let cfg = StreamConfig { lo_hz: freq, rate_hz: rate, bandwidth_hz: 7_000_000, gain_mdb: (gain * 1000.0) as u32, block_samples: 65536 };
            radio.configure(RX, &cfg, Format::Ci16, 0, 0).expect("configure");
            let mut r = radio.start_rx().expect("start");
            loop {
                let mut b = vec![0u8; 1 << 18];
                if r.read_exact(&mut b).is_err() || tx.send(b).is_err() {
                    break;
                }
            }
            let _ = radio.stop(RX);
        });
        inp.rx = Some(rxc);
    }
    let stdout = std::io::stdout();
    let mut so = stdout.lock();
    let need = t.frame_len + t.sym;
    let mut start: Option<u64> = None;
    let mut misses = 0;
    let mut frame_no = 0u32;
    let mut out = Vec::with_capacity(1 << 20);
    while (frame_no as usize) < a.max_frames {
        let s = match start {
            Some(s) => s,
            None => {
                // acquisition: one frame period somewhere holds a bootstrap
                let from = inp.end().max(inp.base);
                let from = if live { from } else { inp.base };
                if !inp.need(from, t.frame_len + t.bootstrap.len() + 16) {
                    break;
                }
                let (p, m) = rx.search(&inp, from, t.frame_len);
                eprintln!("a3rx: acquired bootstrap at sample {p}, correlation {m:.3}");
                if m < 0.1 {
                    if !live { break }
                    continue;
                }
                misses = 0;
                start = Some(p);
                p
            }
        };
        if !inp.need(s.saturating_sub(16), need + 64) {
            break;
        }
        let from = s.saturating_sub(12).max(inp.base);
        let (p, m) = rx.search(&inp, from, 25);
        let s = if m > 0.2 { misses = 0; p } else { misses += 1; s };
        // no bootstrap near the expected place at all (a gap in the sample stream, say): search again now, not eight frames later
        if misses > 8 || (m < 0.05 && start.is_some()) {
            eprintln!("a3rx: lost the frame timing, reacquiring");
            start = None;
            continue;
        }
        if !inp.need(s, need) {
            break;
        }
        out.clear();
        let stat = rx.frame(inp.slice(s, need), frame_no, &mut out);
        let mut next = s + t.frame_len as u64;
        if live {
            // never fall more than a few frames behind the radio
            inp.drain_pending();
            let backlog = inp.end().saturating_sub(next) as usize / t.frame_len;
            if backlog > 6 {
                next += ((backlog - 1) * t.frame_len) as u64;
                inp.dropped += (backlog - 1) as u64;
            }
        }
        let backlog_ms = inp.end().saturating_sub(next) as f32 / FS * 1000.0;
        let stat = format!("{{{stat},\"sync\":{m:.3},\"sample\":{s},\"backlog_ms\":{backlog_ms:.0},\"frames_dropped\":{}}}", inp.dropped);
        record(&mut out, b"STAT", stat.as_bytes());
        if so.write_all(&out).is_err() || so.flush().is_err() {
            break;
        }
        eprintln!("{stat}");
        start = Some(next);
        frame_no += 1;
    }
}
