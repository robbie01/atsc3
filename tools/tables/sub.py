"""Subframe demodulation for this capture's configuration (from CRC-validated L1):
16K FFT, GI 512, 1 preamble symbol (cred 4 on preamble symbol 0) + 100 subframe symbols (cred 0),
SP24_4 boost 0 dB, SBS first+last, frequency interleaver on."""
import numpy as np
import a3common as c, frame as f, chan, rx

DX, DY = 24, 4
SP_EXTRA = [3480, 5808, 11496]       # additional continual pilots for 16K SP24_4 (A/322 Table D.1.4 via gr-atsc3)
NSUB = 100
A_CP = 10 ** (8.52 / 20)
A_SP = 1.0                           # boost index 0 -> 0 dB (A/322 Table 9.14)
REF = 1 - 2 * f.PRBS.astype(float)

def sym_kind(ls):
    return 'S' if ls in (0, NSUB - 1) else 'D'

_maps = {}
def cmap(ls):
    key = (sym_kind(ls), ls % DY)
    if key not in _maps:
        t = f.carrier_map(sym_kind(ls), 0, dx=DX, dy=DY, l=ls, sp_extra=SP_EXTRA)
        _maps[key] = t
    return _maps[key]

def demod_frame(x, nsub=NSUB, backoff=128):
    """returns Y[nsub, 13825] for subframe symbols (frame symbol index 1..)"""
    Y = np.empty((nsub, f.NOC_MAX), np.complex64)
    for ls in range(nsub):
        Y[ls] = f.demod_symbol(x, f.BS_LEN + (1 + ls) * f.SYM, backoff)
    return Y

def pilot_obs(Y):
    """raw channel observations at pilots: list of (ls, k, Hraw, amp)"""
    obs = []
    for ls in range(Y.shape[0]):
        t = cmap(ls)
        ks = np.where(t == 's')[0]; kc = np.where(t == 'c')[0]
        obs.append((ks, Y[ls, ks] * REF[ks] / A_SP, kc, Y[ls, kc] * REF[kc] / A_CP))
    return obs

def estimate(Y, Hpre=None, tau_lo=-24, tau_hi=120):
    """quasi-static channel estimate over the supplied symbols with per-symbol common phase / gain tracking.
    Returns H[nsub, 13825], N0, diagnostics"""
    nsub = Y.shape[0]
    obs = pilot_obs(Y)
    kall = np.arange(f.NOC_MAX) - f.NOC_MAX // 2
    # pass 1: reference = average over symbols of pilot observations (per carrier)
    def average(phases):
        num = np.zeros(f.NOC_MAX, complex); den = np.zeros(f.NOC_MAX)
        for ls, (ks, hs, kc, hc) in enumerate(obs):
            ph = phases[ls]
            np.add.at(num, ks, hs * np.conj(ph) * A_SP ** 2); np.add.at(den, ks, A_SP ** 2 * np.abs(ph) ** 2)
            np.add.at(num, kc, hc * np.conj(ph) * A_CP ** 2); np.add.at(den, kc, A_CP ** 2 * np.abs(ph) ** 2)
        kk = np.where(den > 0)[0]
        return kk, num[kk] / den[kk], den[kk]
    phases = np.ones(nsub, complex)
    for it in range(3):
        kk, Havg, w = average(phases)
        Hfit, res, a, taus = chan.fit(kk - f.NOC_MAX // 2, Havg, kall, tau_lo, tau_hi)
        # per symbol complex gain vs fitted channel (least squares over that symbol's pilots)
        for ls, (ks, hs, kc, hc) in enumerate(obs):
            num = np.vdot(Hfit[ks], hs) * A_SP ** 2 + np.vdot(Hfit[kc], hc) * A_CP ** 2
            den = np.sum(np.abs(Hfit[ks]) ** 2) * A_SP ** 2 + np.sum(np.abs(Hfit[kc]) ** 2) * A_CP ** 2
            phases[ls] = num / den
    # noise estimate from pilot residuals against the final model
    rs = []; 
    for ls, (ks, hs, kc, hc) in enumerate(obs):
        rs.append(np.abs(hs - phases[ls] * Hfit[ks]) ** 2 * A_SP ** 2)
        rs.append(np.abs(hc - phases[ls] * Hfit[kc]) ** 2 * A_CP ** 2)
    n0 = float(np.mean(np.concatenate(rs)))
    H = phases[:, None] * Hfit[None, :]
    return H, n0, dict(phases=phases, Hfit=Hfit, taps=a, taus=taus)

def data_cells(Y, H, n0, ls):
    """equalised, frequency de-interleaved data cells of subframe symbol ls"""
    t = cmap(ls)
    d = np.where(t == 'd')[0]
    eq = Y[ls, d] / H[ls, d]
    nv = n0 / np.abs(H[ls, d]) ** 2
    return f.freq_deinterleave(eq, ls + 1), f.freq_deinterleave(nv, ls + 1)
