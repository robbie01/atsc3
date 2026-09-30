# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fast SNR estimate for the 16K / GI 512 ATSC 3.0 signal from the cyclic prefix, at 6.912 MS/s.
The guard interval is a copy of the end of the symbol, so the correlation between the two is S/(S+N)."""
import numpy as np
N, GI = 16384, 512
SYM = N + GI
USE = 256            # the last part of the guard interval: the first part carries echoes of the previous symbol

def estimate(x):
    """x: complex samples at 6.912 MS/s, at least a few symbols.  Returns (snr_db, rho)"""
    x = np.asarray(x, np.complex64)
    a, b = x[:-N], x[N:]
    r = np.cumsum(a * np.conj(b)); e = np.cumsum(0.5 * (a.real ** 2 + a.imag ** 2 + b.real ** 2 + b.imag ** 2))
    r = r[USE:] - r[:-USE]; e = e[USE:] - e[:-USE]
    k = len(r) // SYM
    if k < 2: return None, None
    R = r[:k * SYM].reshape(k, SYM); E = e[:k * SYM].reshape(k, SYM)
    # symbol timing: the phase with the largest summed |correlation|
    ph = int(np.argmax(np.abs(R).sum(0)))
    rho = np.abs(R[:, ph]) / np.maximum(E[:, ph], 1e-12)
    rho = np.sort(rho)[k // 4:]                      # drop symbols that straddle a frame boundary or a bootstrap
    rh = float(np.clip(np.median(rho), 1e-6, 0.9999))
    return 10 * np.log10(rh / (1 - rh)), rh
