"""Export everything the real-time decoder needs as flat little-endian tables, computed by the validated
Python modules (frame, sub, chan, plp, ldpc, bootstrap) for this station's configuration:
16K FFT, GI 512, preamble cred 4 + 100 subframe symbols, SP24_4, PLP0 16QAM 9/15, PLP1 256QAM 9/15."""
import numpy as np, json, os, sys
from scipy import signal
import frame as f, sub, chan, plp, bootstrap as b

OUT = sys.argv[1]
man = {}
def put(name, arr, dtype):
    a = np.ascontiguousarray(np.asarray(arr).astype(dtype))
    a.tofile(os.path.join(OUT, name + '.bin'))
    man[name] = dict(dtype=np.dtype(dtype).str, shape=list(a.shape))

NSYM = 1 + sub.NSUB
NOC = f.NOC_MAX
CHUNK = 10
TAUS = np.arange(-24, 121, 1.0)
L1CELLS = 3820 + 3708

# ---- pilots per symbol (absolute carrier index), expected value and weight
pil_ptr = [0]; pil_car = []; pil_ref = []; pil_amp = []
types = []
noc_p = f.CARRIERS[4]; shift_p = (NOC - noc_p) // 2
tp = f.carrier_map('P', 4, preamble_dx=12)
refp = 1 - 2 * f.PRBS[:noc_p].astype(float)
ap = 10 ** (4.6 / 20); ac = 10 ** (8.52 / 20)
def add(car, ref, amp):
    o = np.argsort(car); pil_car.extend(car[o]); pil_ref.extend(ref[o]); pil_amp.extend(amp[o]); pil_ptr.append(len(pil_car))
pp = np.where(tp == 'p')[0]; cp = np.where(tp == 'c')[0]
add(np.concatenate([pp, cp]) + shift_p, np.concatenate([refp[pp], refp[cp]]), np.concatenate([np.full(len(pp), ap), np.full(len(cp), ac)]))
for ls in range(sub.NSUB):
    t = sub.cmap(ls)
    ks = np.where(t == 's')[0]; kc = np.where(t == 'c')[0]
    add(np.concatenate([ks, kc]), np.concatenate([sub.REF[ks], sub.REF[kc]]), np.concatenate([np.full(len(ks), sub.A_SP), np.full(len(kc), sub.A_CP)]))
put('pil_ptr', pil_ptr, '<u4'); put('pil_car', pil_car, '<u2'); put('pil_ref', pil_ref, '<f4'); put('pil_amp', pil_amp, '<f4')

# ---- least-squares fit matrices per estimation group: group 0 = preamble symbol, then subframe chunks
groups = [(0, 1)] + [(1 + s, min(CHUNK, sub.NSUB - s)) for s in range(0, sub.NSUB, CHUNK)]
pats = {}; g_pat = []; pat_list = []
for (s0, n) in groups:
    kk = np.unique(np.concatenate([pil_car[pil_ptr[s]:pil_ptr[s + 1]] for s in range(s0, s0 + n)])).astype(int)
    if s0 == 0:
        koff = kk - shift_p - noc_p // 2          # the preamble fit is referenced to its own centre carrier, as in chan.preamble_estimate
        centre = shift_p + noc_p // 2
    else:
        koff = kk - NOC // 2; centre = NOC // 2
    key = (kk.tobytes(), centre)
    if key not in pats:
        A = np.exp(-2j * np.pi * np.outer(koff, TAUS) / f.NFFT)
        G = A.conj().T @ A
        G += 1e-2 * np.trace(G).real / len(TAUS) * np.eye(len(TAUS))
        P = np.linalg.solve(G, A.conj().T)
        pats[key] = len(pat_list); pat_list.append((kk, P, centre))
    g_pat.append(pats[key])
put('group_first_sym', [g[0] for g in groups], '<u4'); put('group_nsym', [g[1] for g in groups], '<u4'); put('group_pattern', g_pat, '<u4')
for i, (kk, P, centre) in enumerate(pat_list):
    put('pat%d_kk' % i, kk, '<u2')
    put('pat%d_P' % i, np.stack([P.real, P.imag], -1), '<f4')
man['patterns'] = [dict(centre=int(c_), n_kk=int(len(kk))) for kk, P, c_ in pat_list]
put('taus', TAUS, '<f4')

# ---- where every data cell of the frame comes from: index = symbol * NOC + absolute carrier
stream = []
d = np.where(tp == 'd')[0]
H = f.freq_interleaver_seq(0, len(d)); src = np.empty(len(d), np.int64); src[H] = 0 * NOC + d + shift_p
stream.append(src[L1CELLS:])
for ls in range(sub.NSUB):
    d = np.where(sub.cmap(ls) == 'd')[0]
    H = f.freq_interleaver_seq(ls + 1, len(d)); src = np.empty(len(d), np.int64); src[H] = (ls + 1) * NOC + d
    stream.append(src)
stream = np.concatenate(stream)
plps = []
for pid, (start, size, mod, tib, nfec) in enumerate([(0, 64800, 4, 1, 4), (64800, 1296000, 8, 10, 160)]):
    fc = 64800 // mod
    T = plp.hti_tables(fc, tib, nfec, nfec)
    blocks = np.concatenate(plp.hti_deinterleave(stream[start:start + size], T, fc))
    assert blocks.shape == (nfec, fc) and len(np.unique(blocks)) == nfec * fc
    put('plp%d_src' % pid, blocks, '<u4')
    B = plp.Bicm('9_15', mod)
    put('plp%d_u2cw' % pid, B.u2cw, '<u4')
    put('plp%d_const' % pid, np.stack([B.const.real, B.const.imag], -1), '<f4')
    plps.append(dict(id=pid, mod_bits=mod, fec_blocks=nfec, fec_cells=fc, K=B.K, Kbch=B.Kbch, N=B.N))
man['plps'] = plps

# ---- LDPC 64800 rate 9/15 (both pipes), by check
code = B.code
put('ldpc_row_ptr', np.append(code.cstart, code.E), '<u4'); put('ldpc_col', code.var, '<u4')
man['ldpc'] = dict(N=int(code.N), K=int(code.K), M=int(code.M), E=int(code.E))
put('bch_gen', plp.BCHG, 'u1'); put('bb_scramble', plp.bb_scramble_bits(B.Kbch), 'u1')

# ---- bootstrap symbol 0 at 6.912 MS/s for frame acquisition
ref0 = b.generate([0, 0, 0])[:b.SYM]
ref = signal.resample_poly(ref0, 9, 8)
put('bootstrap_ref', np.stack([ref.real, ref.imag], -1), '<f4')

man['frame'] = dict(fs=6912000, frame_len=f.FRAME_LEN, bs_len=f.BS_LEN, nfft=f.NFFT, gi=f.GI, sym=f.SYM, noc=NOC, nsym=NSYM, backoff=128,
                    l1_cells=L1CELLS, preamble_noc=noc_p, preamble_shift=shift_p)
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print('exported', len(man), 'entries;', 'patterns', man['patterns'], '; groups', len(groups), '; stream cells', len(stream))
