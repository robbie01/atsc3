"""Frame access + OFDM demodulation helpers for the ATSC 3.0 capture."""
import numpy as np, json, os
import a3common as c
import cparse

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_LEN = 1720320            # samples @ 6.912 MS/s, measured bootstrap period (symbol aligned: 13824 + 101*16896)
BS_LEN = 13824                 # bootstrap = 2 ms
NFFT = 16384; GI = 512; SYM = NFFT + GI
NOC_MAX = 13825
CARRIERS = {0: 13825, 1: 13633, 2: 13441, 3: 13249, 4: 13057}
CP16 = np.array(cparse.nested('pilotgenerator_cc_impl.cc', 'continual_pilot_table_16K'), int)

def bootstraps():
    r = json.load(open(os.path.join(HERE, 'bootstraps.json')))
    g = [x for x in r if x['mf_peak_over_median'] > 500 and min(x['peak_to_mean']) > 200]
    p = np.array([x['start_native'] for x in g])
    per = FRAME_LEN / c.FS_A3 * c.FS_TRUE
    k = np.rint((p - p[0]) / per)
    A = np.polyfit(k, p, 1)
    return k.astype(int), np.polyval(A, k), g

def prbs(n=NOC_MAX):
    # A/322 8.1.2: G(x) = 1 + x^9 + x^10 + x^12 + x^13, init 0000000011011; (LFSR form as in gr-atsc3)
    sr = 0x1b; out = np.zeros(n, np.int8)
    for i in range(n):
        b = (sr ^ (sr >> 1) ^ (sr >> 3) ^ (sr >> 4)) & 1
        out[i] = sr & 1
        sr >>= 1
        if b: sr |= 0x1000
    return out
PRBS = prbs()
assert ''.join(map(str, PRBS[:24])) == '110110000000000101000000'   # A/322 8.1.2 first 24 values

def get_frame(start_native, nsym=None, cfo=0.0, early=0):
    """resample a frame (from bootstrap start) to 6.912 MS/s. Returns samples starting `early`
    samples before the bootstrap start."""
    n = FRAME_LEN if nsym is None else BS_LEN + nsym * SYM
    step = c.FS_TRUE / c.FS_A3
    x = c.resample(start_native - early * step, n + early + 64, c.FS_A3)
    if cfo:
        x = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)) / c.FS_A3).astype(np.complex64)
    return x

def cp_cfo(x, first=BS_LEN, nsym=101):
    """CFO estimate (Hz) and fine timing check from the cyclic prefix of all symbols of the frame"""
    acc = 0
    for l in range(nsym):
        s = first + l * SYM
        a = x[s + 64:s + GI]; b = x[s + 64 + NFFT:s + GI + NFFT]
        acc += np.vdot(a, b)
    return np.angle(acc) / (2 * np.pi) * c.FS_A3 / NFFT, acc

def demod_symbol(x, start, backoff=128):
    """FFT of one OFDM symbol whose GI starts at `start`; window begins `backoff` samples before the
    end of the GI; phase slope compensated. Returns NOC_MAX carriers (absolute carrier index 0..13824)."""
    s = start + GI - backoff
    Y = np.fft.fftshift(np.fft.fft(x[s:s + NFFT])) / np.sqrt(NFFT)
    k = np.arange(-(NOC_MAX // 2), NOC_MAX // 2 + 1)
    Y = Y[NFFT // 2 + k]
    return Y * np.exp(2j * np.pi * k * backoff / NFFT)

def carrier_map(kind, cred, dx=None, dy=None, l=0, preamble_dx=12, sp_extra=None):
    """relative-index carrier types for a symbol. kind: 'P' preamble, 'S' sbs, 'D' data.
    returns (types array of chars) 'd' data, 'p' preamble pilot, 'c' continual, 's' scattered/edge"""
    noc = CARRIERS[cred]; shift = (NOC_MAX - noc) // 2
    t = np.full(noc, 'd')
    cp = CP16 - shift
    cp = cp[(cp >= 0) & (cp < noc)]
    t[cp] = 'c'
    k = np.arange(noc)
    if kind == 'P':
        t[k % preamble_dx == 0] = 'p'
    elif kind == 'S':
        t[k % dx == 0] = 's'
    else:
        t[k % (dx * dy) == dx * (l % dy)] = 's'
    if kind != 'P':
        t[0] = 's'; t[noc - 1] = 's'
        if sp_extra is not None:
            t[np.array(sp_extra, int)] = 's'
    return t

def freq_interleaver_seq(nsym_index, ncells):
    """H for OFDM symbol index (within subframe / preamble numbering as in gr-atsc3) for 16K.
    Returns array H with out[n] = in[H[n]] (transmit side)."""
    return _fi_all(nsym_index)[0 if nsym_index % 2 == 0 else 1][ncells]

_fi_cache = {}
class _Lazy(dict):
    def __init__(self, base): self.base = base
    def __missing__(self, ncells):
        v = self.base[self.base < ncells]
        self[ncells] = v
        return v

def _fi_base():
    if 'base' in _fi_cache: return _fi_cache['base']
    pn_degree = 13; pn_mask = 0x1fff; max_states = 16384
    logic = [0, 1, 4, 5, 9, 11]
    be = cparse.nested('freqinterleaver_cc_impl.cc', 'bitperm16keven')
    bo = cparse.nested('freqinterleaver_cc_impl.cc', 'bitperm16kodd')
    be = [int(v) for v in be]; bo = [int(v) for v in bo]
    ev = np.zeros(max_states, np.int64); od = np.zeros(max_states, np.int64)
    lfsr = 0
    for j in range(max_states):
        if j < 2: lfsr = 0
        elif j == 2: lfsr = 1
        else:
            r = 0
            for k in logic: r ^= (lfsr >> k) & 1
            lfsr &= pn_mask; lfsr >>= 1; lfsr |= r << (pn_degree - 1)
        e = 0; o = 0
        for n in range(pn_degree):
            bit = (lfsr >> n) & 1
            e |= bit << be[n]; o |= bit << bo[n]
        ev[j] = e + (j % 2) * (max_states // 2)
        od[j] = o + (j % 2) * (max_states // 2)
    _fi_cache['base'] = (ev, od)
    return ev, od

def symbol_offsets(nsym):
    """lfsr2 sequence (one new value per pair of symbols), 16K"""
    pn_degree = 13; pn_mask = 0x1fff
    logic2 = [0, 1, 2, 12]
    out = []; l2 = 0
    for i in range(nsym):
        if i % 2 == 0:
            if i == 0:
                l2 = (pn_mask << 1) | 1
            else:
                r = 0
                for k in logic2: r ^= (l2 >> k) & 1
                l2 &= (pn_mask << 1) | 1
                l2 >>= 1
                l2 |= r << pn_degree
        out.append(l2)
    return out

def _fi_all(i):
    key = ('sym', i)
    if key not in _fi_cache:
        ev, od = _fi_base()
        l2 = symbol_offsets(i + 1)[i]
        _fi_cache[key] = (_Lazy(ev ^ l2), _Lazy(od ^ l2))
    return _fi_cache[key]

def freq_deinterleave(cells, sym_index):
    H = freq_interleaver_seq(sym_index, len(cells))
    assert len(H) == len(cells) and len(np.unique(H)) == len(cells)
    out = np.empty_like(cells)
    out[H] = cells
    return out
