"""ATSC A/321 bootstrap: generator (from the standard's definition) and decoder.
Constants (ZC root 137, PN seeds Table 6.1, N_ZC 1499, C/A/B sizes) from A/321:2022-03 sect. 5, 6."""
import numpy as np

NFFT = 2048; NB = 504; NC = 520; NZC = 1499; NH = (NZC - 1) // 2; SYM = NFFT + NB + NC
FS = 6.144e6
SEEDS_MAJOR0 = [0x019D, 0x00ED, 0x01E8, 0x00E8, 0x00FB, 0x0021, 0x0054, 0x00EC]   # A/321 Table 6.1
ROOT_MAJOR0 = 137

def pn_sequence(seed, n):
    # g(x) = x^16 + x^15 + x^14 + x + 1 (A/321 5.2.2); output is r0, register clocked right.
    sr = seed; out = np.zeros(n, np.int8)
    for i in range(n):
        out[i] = sr & 1
        b = (sr ^ (sr >> 1) ^ (sr >> 14) ^ (sr >> 15)) & 1
        sr >>= 1
        if b: sr |= 0x8000
    return out

def zc(q):
    k = np.arange(NZC, dtype=np.float64)
    return np.exp(-1j * np.pi * q * k * (k + 1) / NZC)

def freq_sequences(q=ROOT_MAJOR0, seed=SEEDS_MAJOR0[0], nsym=4):
    """returns [nsym, NFFT] arrays in FFT order (bin 0 = DC), without cyclic shift or final inversion"""
    z = zc(q); p = pn_sequence(seed, (nsym + 1) * NH + 8)
    cpn = 1 - 2 * p.astype(np.float64)
    S = np.zeros((nsym, NFFT), complex)
    for n in range(nsym):
        for k in range(-NH, 0):
            S[n, k % NFFT] = z[k + NH] * cpn[(n + 1) * NH + k]
        for k in range(1, NH + 1):
            S[n, k % NFFT] = z[k + NH] * cpn[(n + 1) * NH - k]
    return S

def gray_shift(byte, nb=8):
    """relative cyclic shift from signalling bits (A/321 5.3.2). byte: b0 is the MSB."""
    b = [(byte >> (nb - 1 - i)) & 1 for i in range(nb)] + [0] * (11 - nb)
    m = 0
    for i in range(11):
        if i > 10 - nb:
            bit = sum(b[0:10 - i + 1]) % 2
        elif i == 10 - nb:
            bit = 1
        else:
            bit = 0
        m |= bit << i
    return m

def gray_demap(M, nb=8):
    """Annex A of A/321: estimated relative shift -> signalling bits"""
    m = [(M >> i) & 1 for i in range(11)] + [0]
    byte = 0
    for i in range(nb):
        bit = m[10] if i == 0 else (m[11 - i] ^ m[10 - i])
        byte = (byte << 1) | bit
    return byte

def generate(sig_bytes, q=ROOT_MAJOR0, seed=SEEDS_MAJOR0[0]):
    """time domain bootstrap, 4 symbols (CAB, BCA, BCA, BCA) at 6.144 MS/s"""
    nsym = 1 + len(sig_bytes)
    S = freq_sequences(q, seed, nsym)
    out = []
    M = 0
    t = np.arange(NFFT)
    for n in range(nsym):
        s = S[n].copy()
        if n == nsym - 1:
            s = -s
        At = np.fft.ifft(s) * NFFT / np.sqrt(NZC - 1)
        if n > 0:
            M = (M + gray_shift(sig_bytes[n - 1])) % NFFT
        A = At[(t + M) % NFFT]
        if n == 0:
            C = A[NFFT - NC:]
            tt = np.arange(NFFT + NC, SYM)
            B = A[tt - 1024] * np.exp(2j * np.pi * tt / NFFT)
            out.append(np.concatenate([C, A, B]))
        else:
            tt = np.arange(0, NB)
            B = A[tt + 1528] * np.exp(-2j * np.pi * (tt - NC) / NFFT)
            C = A[NFFT - NC:]
            out.append(np.concatenate([B, C, A]))
    return np.concatenate(out)

def parse_fields(b):
    X = (b[0] >> 2) & 31
    if X < 8: T = 50 * X + 50
    elif X < 16: T = 100 * (X - 8) + 500
    elif X < 24: T = 200 * (X - 16) + 1300
    else: T = 400 * (X - 24) + 2900
    return dict(ea_wake_up_1=b[0] >> 7, min_time_to_next=X, min_time_to_next_ms=T,
                system_bandwidth=b[0] & 3, system_bandwidth_mhz=[6, 7, 8, '>8'][b[0] & 3],
                ea_wake_up_2=b[1] >> 7, bsr_coefficient=b[1] & 127,
                baseband_sample_rate_hz=((b[1] & 127) + 16) * 384000,
                preamble_structure=b[2])

def cab_metric(x):
    """Sliding delayed-correlation detection metric for the CAB symbol; index = candidate symbol start."""
    n = len(x)
    ph = np.exp(-2j * np.pi * np.arange(n) / NFFT).astype(np.complex64)
    def slide(prod, W):
        cs = np.concatenate([[0], np.cumsum(prod.astype(np.complex128))])
        return cs[W:] - cs[:-W]
    ca = slide(x[:-NFFT] * np.conj(x[NFFT:]), NC)                 # C vs end of A, lag 2048, starts at p
    ba = slide(x[NB:] * np.conj(x[:-NB]) * ph[NB:], NB)            # B vs A, lag 504, starts at p+2064
    bc = slide(x[NFFT + NB:] * np.conj(x[:-(NFFT + NB)]) * ph[NFFT + NB:], NB)   # B vs C lag 2552, starts at p+16
    pw = slide(np.abs(x) ** 2, SYM)
    L = min(len(ca), len(ba) - 2064, len(bc) - 16, len(pw))
    m = (np.abs(ca[:L]) + np.abs(ba[2064:2064 + L]) + np.abs(bc[16:16 + L])) / (NC + 2 * NB)
    return m / (pw[:L] / SYM), ca[:L]

def decode(x, q=ROOT_MAJOR0, seed=SEEDS_MAJOR0[0], nsym=4, max_int=3):
    """x: samples at 6.144 MS/s starting at the (approximate) start of symbol 0 (C part), CFO corrected
    to within a fraction of a carrier. Returns dict with signalling bytes and quality metrics."""
    S = freq_sequences(q, seed, nsym)
    act = np.abs(S[0]) > 0
    Z = []
    # part A positions: sym 0: NC ; sym n>=1: n*SYM + NB + NC
    for n in range(nsym):
        a0 = n * SYM + (NC if n == 0 else NB + NC)
        Y = np.fft.fft(x[a0:a0 + NFFT])
        Z.append(Y)
    # integer carrier offset + timing from symbol 0
    best = None
    for io in range(-max_int, max_int + 1):
        Yr = np.roll(Z[0], -io)
        h = np.fft.ifft(Yr * np.conj(S[0]))
        pk = np.argmax(np.abs(h)); v = np.abs(h[pk]) ** 2 / np.mean(np.abs(h) ** 2)
        if best is None or v > best[0]:
            best = (v, io, pk)
    v0, io, pk0 = best
    out = dict(sym0_peak_to_mean=float(v0), int_cfo_bins=int(io), sym0_delay=int(pk0 if pk0 < NFFT // 2 else pk0 - NFFT))
    Zc = [np.roll(z, -io) * np.conj(S[n]) for n, z in enumerate(Z)]
    bytes_, shifts, pks, signs = [], [], [], []
    for n in range(1, nsym):
        d = Zc[n] * np.conj(Zc[n - 1])
        h = np.fft.fft(d)             # peak index = relative shift M~ (since A_n(t)=A~((t+M) mod N))
        k = int(np.argmax(np.abs(h)))
        shifts.append(k); pks.append(float(np.abs(h[k]) ** 2 / np.mean(np.abs(h) ** 2)))
        signs.append(float(np.real(h[k]) / np.abs(h[k])))
        bytes_.append(gray_demap(k))
    out.update(rel_shifts=shifts, peak_to_mean=pks, sign=signs, bytes=bytes_)
    # SNR estimate from symbol 0: channel impulse response energy within a window vs outside
    h = np.fft.ifft(Zc[0])
    p = np.abs(h) ** 2
    idx = (np.arange(NFFT) - pk0) % NFFT
    win = (idx < 96) | (idx > NFFT - 32)
    noise_per_tap = np.mean(p[~win])
    sig = np.sum(p[win]) - noise_per_tap * win.sum()
    # noise in active carriers only: total noise over NFFT taps corresponds to NZC-1 active bins
    out['snr_db_sym0'] = float(10 * np.log10(sig / (noise_per_tap * NFFT)))
    return out
