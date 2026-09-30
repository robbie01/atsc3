"""LDPC codes of ATSC 3.0 (A/322 6.1.3): parity check matrix construction from address tables
(tables read from the gr-atsc3 sources), encoder (for self-test) and a sum-product decoder."""
import numpy as np
import cparse

def build_type_a(table, N, K, M1, M2, Q1, Q2):
    """table rows: [count, addr...]. Returns (chk, var) edge lists. Codeword = [info K | p1 (M1) | p2 (M2)]
    with parity stored group-wise interleaved as transmitted: c[K+360t+s] = p[Q1 s + t], c[K+M1+360t+s] = p[M1+Q2 s+t]."""
    chk = []; var = []
    nrows_info = K // 360
    im = 0
    for row in table:
        cnt = int(row[0]); addrs = [int(a) for a in row[1:1 + cnt]]
        for n in range(360):
            for a in addrs:
                if a < M1:
                    cb = (a + n * Q1) % M1
                else:
                    cb = M1 + (a - M1 + n * Q2) % M2
                chk.append(cb); var.append(im)
            im += 1
    assert im == K + M1, (im, K, M1)
    # parity variable index for parity bit i (accumulator order)
    def pvar(i):
        if i < M1:
            s, t = divmod(i, Q1)
            return K + 360 * t + s
        s, t = divmod(i - M1, Q2)
        return K + M1 + 360 * t + s
    for i in range(M1):
        chk.append(i); var.append(pvar(i))
        if i > 0:
            chk.append(i); var.append(pvar(i - 1))
    for i in range(M1, M1 + M2):
        chk.append(i); var.append(pvar(i))
    return np.array(chk), np.array(var)

def build_type_b(table, N, K, Q):
    M = N - K
    chk = []; var = []
    im = 0
    for row in table:
        cnt = int(row[0]); addrs = [int(a) for a in row[1:1 + cnt]]
        for n in range(360):
            for a in addrs:
                chk.append((a + n * Q) % M); var.append(im)
            im += 1
    assert im == K
    def pvar(i):
        s, t = divmod(i, Q)
        return K + 360 * t + s
    for i in range(M):
        chk.append(i); var.append(pvar(i))
        if i > 0:
            chk.append(i); var.append(pvar(i - 1))
    return np.array(chk), np.array(var)

class Code:
    def __init__(self, chk, var, N, K):
        self.N = N; self.K = K; self.M = N - K
        o = np.lexsort((var, chk))
        self.chk = chk[o]; self.var = var[o]
        self.E = len(chk)
        self.cstart = np.searchsorted(self.chk, np.arange(self.M))
        assert len(np.unique(self.chk)) == self.M
        self.vorder = np.argsort(self.var, kind='stable')
        self.vsorted = self.var[self.vorder]
        self.vstart = np.searchsorted(self.vsorted, np.arange(self.N))
        self.vdeg = np.bincount(self.var, minlength=N)

    def syndrome(self, bits):
        b = bits[..., self.var].astype(np.int64)
        return np.add.reduceat(b, self.cstart, axis=-1) & 1

    def encode_check(self, cw):
        return not self.syndrome(cw).any()

    def decode(self, llr, iters=50, early=True):
        """llr: [B, N] (positive = bit 0). Returns (hard bits [B,N], ok [B], iterations used)"""
        llr = np.atleast_2d(llr).astype(np.float64)
        B = llr.shape[0]
        c2v = np.zeros((B, self.E))
        active = np.ones(B, bool)
        hard = (llr < 0).astype(np.uint8)
        ok = np.zeros(B, bool)
        nit = np.zeros(B, int)
        for it in range(iters):
            idx = np.where(active)[0]
            if len(idx) == 0:
                break
            m = c2v[idx]
            tot = llr[idx] + np.add.reduceat(m[:, self.vorder], self.vstart, axis=1) * (self.vdeg > 0)
            v2c = tot[:, self.var] - m
            v2c = np.clip(v2c, -30, 30)
            sg = v2c < 0
            a = np.abs(v2c)
            phi = -np.log(np.tanh(np.maximum(a, 1e-9) / 2))
            ssum = np.add.reduceat(phi, self.cstart, axis=1)
            par = np.add.reduceat(sg.astype(np.int64), self.cstart, axis=1) & 1
            cnt = np.diff(np.append(self.cstart, self.E))
            ssum_e = np.repeat(ssum, cnt, axis=1) - phi
            par_e = np.repeat(par, cnt, axis=1) ^ sg
            ssum_e = np.maximum(ssum_e, 1e-12)
            mag = -np.log(np.tanh(ssum_e / 2))
            m = np.where(par_e, -mag, mag)
            c2v[idx] = m
            tot = llr[idx] + np.add.reduceat(m[:, self.vorder], self.vstart, axis=1) * (self.vdeg > 0)
            h = (tot < 0).astype(np.uint8)
            hard[idx] = h
            nit[idx] = it + 1
            if early:
                s = self.syndrome(h).any(axis=1)
                ok[idx] = ~s
                active[idx[~s]] = False
        if not early:
            ok = ~self.syndrome(hard).any(axis=1)
        return hard, ok, nit

def encode_type_a(info, table, N, K, M1, M2, Q1, Q2):
    """direct port of the A/322 type A encoding procedure; returns the codeword (parity interleaved)."""
    info = np.asarray(info, np.uint8)
    p = np.zeros(M1 + M2, np.uint8)
    nr = K // 360
    lam = np.zeros(N, np.uint8); lam[:K] = info
    def accumulate(rows, base):
        im = base
        for row in rows:
            cnt = int(row[0]); addrs = [int(a) for a in row[1:1 + cnt]]
            for n in range(360):
                if lam[im]:
                    for a in addrs:
                        if a < M1: p[(a + n * Q1) % M1] ^= 1
                        else: p[M1 + (a - M1 + n * Q2) % M2] ^= 1
                im += 1
    accumulate(table[:nr], 0)
    for i in range(1, M1):
        p[i] ^= p[i - 1]
    for t in range(Q1):
        for s in range(360):
            lam[K + 360 * t + s] = p[Q1 * s + t]
    accumulate(table[nr:], K)
    for t in range(Q2):
        for s in range(360):
            lam[K + M1 + 360 * t + s] = p[M1 + Q2 * s + t]
    return lam

def encode_type_b(info, table, N, K, Q):
    M = N - K
    p = np.zeros(M, np.uint8)
    im = 0
    for row in table:
        cnt = int(row[0]); addrs = [int(a) for a in row[1:1 + cnt]]
        for n in range(360):
            if info[im]:
                for a in addrs:
                    p[(a + n * Q) % M] ^= 1
            im += 1
    p = np.cumsum(p) & 1
    cw = np.zeros(N, np.uint8); cw[:K] = info
    for t in range(Q):
        cw[K + 360 * t:K + 360 * t + 360] = p[Q * np.arange(360) + t]
    return cw

_codes = {}
def l1_code(kind):
    """kind 'a': 16200 rate 3/15 type A (L1-Basic, L1-Detail modes 1,2); 'b': 16200 rate 6/15 type B"""
    if kind not in _codes:
        if kind == 'a':
            tab = cparse.nested('framemapper_cc_impl.cc', 'ldpc_tab_3_15S', int)
            chk, var = build_type_a(tab, 16200, 3240, 1080, 11880, 3, 33)
            _codes[kind] = (Code(chk, var, 16200, 3240), tab, dict(N=16200, K=3240, M1=1080, M2=11880, Q1=3, Q2=33))
        else:
            tab = cparse.nested('framemapper_cc_impl.cc', 'ldpc_tab_6_15S', int)
            chk, var = build_type_b(tab, 16200, 6480, 27)
            _codes[kind] = (Code(chk, var, 16200, 6480), tab, dict(N=16200, K=6480, Q=27))
    return _codes[kind]

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for kind in 'ab':
        code, tab, p = l1_code(kind)
        info = rng.integers(0, 2, p['K']).astype(np.uint8)
        cw = encode_type_a(info, tab, **p) if kind == 'a' else encode_type_b(info, tab, **p)
        print(kind, 'edges', code.E, 'encoder output satisfies H:', code.encode_check(cw))
        for snr_db in ((-4.5, -3.5) if kind == 'a' else (-1.0, 0.0)):   # Es/N0 for QPSK
            B = 6
            s = 1 - 2.0 * cw
            nv = 10 ** (-snr_db / 10) / 2      # per-dimension noise var for unit-energy per bit dimension (QPSK: Es=2 per 2 dims)
            nv = 10 ** (-snr_db / 10) * 0.5 * 1.0   # each bit = one real dim with energy 1 => Es (2 bits) = 2, N0 = 2*sigma2 => Es/N0 = 1/sigma2
            sigma2 = 10 ** (-snr_db / 10)
            y = s[None, :] + np.sqrt(sigma2 / 1.0) * rng.standard_normal((B, len(s))) * np.sqrt(1.0)
            # Es/N0 = 2 / (2 sigma_dim^2) -> sigma_dim^2 = 1/(Es/N0)
            llr = 2 * y / sigma2
            hard, ok, nit = code.decode(llr, 50)
            print(f'  QPSK Es/N0 {snr_db} dB: ok {ok.sum()}/{B}, bit errors {(hard != cw).sum()}, iters {nit}')
