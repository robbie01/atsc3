"""Pilot based channel estimation (least squares fit of a delay-limited channel) and SNR estimate."""
import numpy as np
import frame as f

_basis = {}
def basis(kabs, taus):
    key = (len(kabs), int(kabs[0]), int(kabs[-1]), len(taus), float(taus[0]), float(taus[-1]), hash(kabs.tobytes()))
    if key not in _basis:
        A = np.exp(-2j * np.pi * np.outer(kabs, taus) / f.NFFT)
        _basis[key] = A
    return _basis[key]

_solver = {}
def fit(kp, Hp, kall, tau_lo=-24, tau_hi=120, lam=1e-2, w=None):
    """kp: carrier offsets from DC of pilots; Hp: raw estimates; returns H at kall, and residual at pilots"""
    taus = np.arange(tau_lo, tau_hi + 1, 1.0)
    A = basis(kp, taus)
    key = (id(A), lam)
    if key not in _solver:
        G = A.conj().T @ A
        G += lam * np.trace(G).real / len(taus) * np.eye(len(taus))
        _solver[key] = np.linalg.solve(G, A.conj().T)
    P = _solver[key]
    a = P @ Hp
    fitp = A @ a
    Hall = basis(kall, taus) @ a
    dof = len(taus)
    return Hall, Hp - fitp, a, taus

def preamble_estimate(Yr, cred, preamble_dx=12, boost_db=4.6, **kw):
    """Yr: received carriers (relative index 0..noc-1) of a preamble symbol. Returns H (all carriers), N0, types"""
    noc = f.CARRIERS[cred]
    t = f.carrier_map('P', cred, preamble_dx=preamble_dx)
    ref = 1 - 2 * f.PRBS[:noc].astype(float)
    ap = 10 ** (boost_db / 20); ac = 10 ** (8.52 / 20)
    pp = np.where(t == 'p')[0]; cp = np.where(t == 'c')[0]
    kp = np.concatenate([pp, cp]); amp = np.concatenate([np.full(len(pp), ap), np.full(len(cp), ac)])
    o = np.argsort(kp); kp = kp[o]; amp = amp[o]
    Hraw = Yr[kp] * ref[kp] / amp
    koff = kp - (noc // 2)
    kall = np.arange(noc) - (noc // 2)
    H, res, a, taus = fit(koff, Hraw, kall, **kw)
    # noise variance: residual at preamble pilots scaled back by pilot amplitude (dof correction)
    n0 = np.sum(np.abs(res) ** 2 * amp ** 2) / (len(kp) - len(taus))
    return H, n0, t, (a, taus)
