import os
"""Common helpers for the ATSC 3.0 analysis of tv23.ci16 (receive-only, file on disk)."""
import numpy as np
from scipy import signal

CAP = os.environ.get('A3_CAPTURE', '')   # a 6.912 MS/s ci16 recording, only for the self-tests of these modules
FS_NOM = 8.0e6
PPM = -16.5295                    # measured from the bootstrap period (step2b): frame = 1720320 samples @ 6.912 MS/s
FS_TRUE = FS_NOM * (1 + PPM * 1e-6)

_mm = None
def raw():
    global _mm
    if _mm is None:
        _mm = np.memmap(CAP, dtype='<i2', mode='r')
    return _mm

def nsamp():
    return raw().size // 2

def load(i0, n):
    """complex64 samples [i0, i0+n) at the native rate; zero padded outside the file."""
    m = raw(); N = m.size // 2
    out = np.zeros(n, np.complex64)
    a = max(i0, 0); b = min(i0 + n, N)
    if b > a:
        seg = np.asarray(m[2 * a:2 * b]).astype(np.float32)
        out[a - i0:b - i0] = seg[0::2] + 1j * seg[1::2]
    return out

# --- arbitrary ratio polyphase resampler -------------------------------------------------
_K = 64       # taps per phase
_L = 1024     # phases
_tabs = {}
def _table(fc):
    """fc: cutoff in cycles/input-sample. Returns [L+1, K] table."""
    key = round(fc, 6)
    if key not in _tabs:
        k = np.arange(_K) - (_K // 2 - 1)               # tap offsets relative to floor(p)
        fr = np.arange(_L + 1) / _L
        t = k[None, :] - fr[:, None]                    # distance from wanted point
        h = 2 * fc * np.sinc(2 * fc * t)
        w = np.i0(9.0 * np.sqrt(np.clip(1 - (t / (_K / 2)) ** 2, 0, 1))) / np.i0(9.0)
        h = h * w
        _tabs[key] = (h).astype(np.float32)
    return _tabs[key]

def resample(p0, n_out, fs_out, fs_in=FS_TRUE, fc_hz=3.15e6, chunk=1 << 20):
    """Return n_out samples at rate fs_out; output sample n is taken at input position
    p0 + n*fs_in/fs_out (input sample index, float, in units of native samples)."""
    step = fs_in / fs_out
    fc = min(fc_hz, 0.48 * fs_out) / FS_NOM
    H = _table(fc)
    out = np.empty(n_out, np.complex64)
    for c0 in range(0, n_out, chunk):
        c1 = min(n_out, c0 + chunk)
        p = p0 + np.arange(c0, c1, dtype=np.float64) * step
        ip = np.floor(p).astype(np.int64)
        ph = np.rint((p - ip) * _L).astype(np.int64)
        base = int(ip[0]) - (_K // 2 - 1)
        x = load(base, int(ip[-1]) - int(ip[0]) + _K + 1)
        rel = ip - ip[0]
        acc = np.zeros(c1 - c0, np.complex64)
        for k in range(_K):
            acc += x[rel + k] * H[ph, k]
        out[c0:c1] = acc
    return out

FS_A3 = 6.912e6     # ATSC 3.0 baseband rate for 6 MHz (bsr_coefficient 2)
FS_BS = 6.144e6     # bootstrap rate
