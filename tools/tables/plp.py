"""PLP BICM / time interleaver chain for 64800-bit type-B LDPC codes (as needed for this capture: PLP0 =
16QAM-NUC 9/15 64K, HTI with cell interleaver).  Transmit model for self-test + receiver.
Tables come from the gr-atsc3 sources (A/322 tables)."""
import numpy as np, re
import cparse, ldpc, l1

RATES = ['2_15', '3_15', '4_15', '5_15', '6_15', '7_15', '8_15', '9_15', '10_15', '11_15', '12_15', '13_15']
QVAL_N = {'6_15': 108, '7_15': 96, '8_15': 84, '9_15': 72, '10_15': 60, '11_15': 48, '12_15': 36, '13_15': 24}

def bch_long_generator():
    s = cparse.src('bch_bb_impl.cc')
    polys = re.findall(r'const int polyn(\d+)\[\] = \{([^}]*)\}', s)
    g = np.array([1], np.int64)
    for n, body in polys[:12]:
        g = np.convolve(g, np.array([int(v) for v in body.split(',')])) & 1
    assert len(g) == 193
    return g
BCHG = bch_long_generator()

def bch_remainder(bits, g=BCHG):
    """remainder of bits(x) (first bit = highest power) modulo g; zero for a valid BCH codeword"""
    deg = len(g) - 1
    gi = 0
    for i, v in enumerate(g): gi |= int(v) << i
    reg = 0; top = 1 << deg
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & top: reg ^= gi
    return reg

def bch_encode(msg, g=BCHG):
    deg = len(g) - 1
    r = bch_remainder(np.concatenate([msg, np.zeros(deg, np.uint8)]), g)
    par = np.array([(r >> i) & 1 for i in range(deg - 1, -1, -1)], np.uint8)
    return np.concatenate([msg, par])

def bb_scramble_bits(n):
    return l1.scramble_bits(n)

class Bicm:
    def __init__(self, rate='9_15', mod=4, N=64800):
        assert N == 64800
        self.N = N; self.mod = mod; self.rate = rate
        num = int(rate.split('_')[0])
        self.K = N * num // 15
        self.Kbch = self.K - 192
        tab = cparse.nested('ldpc_bb_impl.cc', 'ldpc_tab_%sN' % rate, int)
        self.tab = tab; self.Q = QVAL_N[rate]
        chk, var = ldpc.build_type_b(tab, N, self.K, self.Q)
        self.code = ldpc.Code(chk, var, N, self.K)
        name = {2: 'QPSK', 4: '16QAM', 6: '64QAM', 8: '256QAM'}[mod]
        self.group = np.array(cparse.nested('interleaver_bb_impl.cc', 'group_tab_%sN_%s' % (rate, name), int))
        assert sorted(self.group) == list(range(180))
        blk = self._block_type(rate, name)
        self.block_type = blk
        # u (cell bit order) index -> interleaved codeword index
        v2cw = (self.group[:, None] * 360 + np.arange(360)[None, :]).reshape(-1)      # v[i] = cw[v2cw[i]]
        if blk == 'B':
            inner = 360 * mod; outer = N // inner
            n = np.arange(outer)[:, None, None]; j = np.arange(360)[None, :, None]; k = np.arange(mod)[None, None, :]
            u2v = (n * inner + j + 360 * k).reshape(-1)
        else:
            # type A: column-wise write, row-wise read; 256QAM has a second, shorter part (A/322 6.2.3)
            nr2 = 180 if mod == 8 else 0
            rows = N // mod - nr2
            j = np.arange(rows)[:, None]; k = np.arange(mod)[None, :]
            u2v = (k * rows + j).reshape(-1)
            if nr2:
                base = N - nr2 * mod
                j2 = np.arange(nr2)[:, None]
                u2v = np.concatenate([u2v, (base + k * nr2 + j2).reshape(-1)])
            assert sorted(u2v) == list(range(N))
        self.u2cw = v2cw[u2v]
        if mod == 4:
            q = np.array(cparse.nested('modulator_bc_impl.cc', 'mod_table_16QAM', complex)[RATES.index(rate)])
            self.const = np.concatenate([q, -np.conj(q), np.conj(q), -q])
        elif mod in (6, 8):
            q = np.array(cparse.nested('modulator_bc_impl.cc', 'mod_table_%s' % name, complex)[RATES.index(rate)])
            self.const = np.concatenate([q, -np.conj(q), np.conj(q), -q])
        elif mod == 2:
            s = np.sqrt(0.5); self.const = np.array([s + 1j * s, -s + 1j * s, s - 1j * s, -s - 1j * s])
        else:
            raise NotImplementedError
        self.const = self.const / np.sqrt(np.mean(np.abs(self.const) ** 2))
        self.labels = np.array([[(i >> (mod - 1 - b)) & 1 for b in range(mod)] for i in range(len(self.const))])

    @staticmethod
    def _block_type(rate, name):
        s = cparse.src('interleaver_bb_impl.cc')
        m = re.search(r'group_table = &group_tab_%sN_%s\[0\];\s*block_type = BLOCK_TYPE_([AB]);' % (rate, name), s)
        return m.group(1)

    # ---- transmit model
    def encode(self, payload):
        """payload Kbch bits (already scrambled) -> cells"""
        info = bch_encode(payload)
        cw = ldpc.encode_type_b(info, self.tab, self.N, self.K, self.Q)
        u = cw[self.u2cw].reshape(-1, self.mod)
        idx = np.zeros(len(u), int)
        for b in range(self.mod): idx = (idx << 1) | u[:, b]
        return self.const[idx], cw

    # ---- receive
    def llr(self, cells, nvar):
        if self.mod >= 6:
            return self.llr_maxlog(cells, nvar)
        d = np.abs(cells[:, None] - self.const[None, :]) ** 2 / nvar[:, None]
        out = np.empty((len(cells), self.mod))
        for b in range(self.mod):
            m0 = -d[:, self.labels[:, b] == 0]; m1 = -d[:, self.labels[:, b] == 1]
            a0 = m0.max(1); a1 = m1.max(1)
            l0 = a0 + np.log(np.exp(m0 - a0[:, None]).sum(1)); l1_ = a1 + np.log(np.exp(m1 - a1[:, None]).sum(1))
            out[:, b] = l0 - l1_
        return out.reshape(-1)

    def llr_maxlog(self, cells, nvar):
        """max-log demapper, single precision, for the big constellations"""
        c = self.const.astype(np.complex64)
        y = np.asarray(cells, np.complex64)
        d = (y.real[:, None] - c.real[None, :]) ** 2 + (y.imag[:, None] - c.imag[None, :]) ** 2
        out = np.empty((len(y), self.mod), np.float32)
        for b in range(self.mod):
            one = self.labels[:, b] == 1
            out[:, b] = d[:, one].min(1) - d[:, ~one].min(1)
        return (out / np.asarray(nvar, np.float32)[:, None]).reshape(-1)

    def decode(self, cells, nvar, iters=50):
        """cells: [B, N/mod]. Returns list of dicts"""
        cells = np.atleast_2d(cells); nvar = np.atleast_2d(nvar)
        L = np.zeros((cells.shape[0], self.N))
        for i in range(cells.shape[0]):
            L[i, self.u2cw] = self.llr(cells[i], nvar[i])
        clip = getattr(self, 'llr_clip', None)
        if clip:
            L = np.clip(L, -clip, clip)
        hard, ok, nit = self.code.decode(L, iters)
        res = []
        for i in range(cells.shape[0]):
            info = hard[i, :self.K]
            rem = bch_remainder(info)
            # bit-level mutual information estimate of the demapper output (per coded bit)
            res.append(dict(ldpc_ok=bool(ok[i]), iters=int(nit[i]), bch_ok=(rem == 0), info=info,
                            raw_ber=float(np.mean((L[i] < 0) != hard[i])) if ok[i] else None, llr=L[i]))
        return res

# ---- hybrid time interleaver (intra-subframe), port of gr-atsc3 init_address
def hti_tables(fec_cells, ti_blocks, ti_fecblocks, ti_fecblocks_max):
    Nfec_ti_max = (ti_fecblocks_max // ti_blocks) + (1 if ti_fecblocks_max % ti_blocks else 0)
    nfec = [ti_fecblocks // ti_blocks + (0 if x < (ti_blocks - (ti_fecblocks % ti_blocks)) else 1) for x in range(ti_blocks)]
    Nd = int(fec_cells).bit_length()
    logic = {11: [0, 3], 12: [0, 2], 13: [0, 1, 4, 6], 14: [0, 1, 4, 5, 9, 11], 15: [0, 1, 2, 12]}[Nd]
    pn_degree = Nd - 1; pn_mask = (1 << pn_degree) - 1; max_states = 1 << Nd
    base = []
    lfsr = 0
    for j in range(max_states):
        if j < 2: lfsr = 0
        elif j == 2: lfsr = 1
        else:
            r = 0
            for k in logic: r ^= (lfsr >> k) & 1
            lfsr &= pn_mask; lfsr >>= 1; lfsr |= r << (pn_degree - 1)
        lfsr |= (j % 2) << pn_degree
        if lfsr < fec_cells: base.append(lfsr)
    base = np.array(base)
    assert len(base) == fec_cells
    Lr = []; TBI = []
    for x in range(ti_blocks):
        prs = []; k = 0
        for r in range(nfec[x]):
            Pr = fec_cells
            while Pr >= fec_cells:
                Pr = 0
                for j in range(Nd):
                    Pr |= (k & (1 << j)) << ((Nd + 16) - 1 - j * 2)
                Pr >>= 16
                k += 1
            prs.append(Pr)
        Lr.append([(base + pr) % fec_cells for pr in prs])
        n = np.arange(fec_cells * Nfec_ti_max)
        Ri = n % fec_cells; Ti = Ri % Nfec_ti_max; Ci = (Ti + n // fec_cells) % Nfec_ti_max
        a = fec_cells * Ci + Ri
        a = a[a >= (Nfec_ti_max - nfec[x]) * fec_cells] - (Nfec_ti_max - nfec[x]) * fec_cells
        TBI.append(a)
    return dict(nfec=nfec, Lr=Lr, TBI=TBI, Nfec_ti_max=Nfec_ti_max)

def hti_interleave(cells, T, fec_cells, cell_il=True):
    out = []; p = 0
    for x, nf in enumerate(T['nfec']):
        blk = cells[p:p + nf * fec_cells].reshape(nf, fec_cells); p += nf * fec_cells
        if cell_il:
            blk = np.stack([blk[j][T['Lr'][x][j]] for j in range(nf)])
        out.append(blk.reshape(-1)[T['TBI'][x]])
    return np.concatenate(out)

def hti_deinterleave(cells, T, fec_cells, cell_il=True):
    """returns list (per TI block) of arrays [nfec, fec_cells]"""
    res = []; p = 0
    for x, nf in enumerate(T['nfec']):
        seg = cells[p:p + nf * fec_cells]; p += nf * fec_cells
        tmp = np.empty_like(seg); tmp[T['TBI'][x]] = seg
        blk = tmp.reshape(nf, fec_cells)
        if cell_il:
            o = np.empty_like(blk)
            for j in range(nf):
                o[j][T['Lr'][x][j]] = blk[j]
            blk = o
        res.append(blk)
    return res

if __name__ == '__main__':
    rng = np.random.default_rng(5)
    b = Bicm('9_15', 4)
    print('16QAM 9/15 64K: K', b.K, 'Kbch', b.Kbch, 'Q', b.Q, 'block type', b.block_type, 'edges', b.code.E)
    print('constellation (first quadrant):', np.round(b.const[:4], 4))
    T = hti_tables(16200, 1, 4, 4)
    pay = rng.integers(0, 2, (4, b.Kbch)).astype(np.uint8)
    cells = []; cws = []
    for i in range(4):
        cl, cw = b.encode(pay[i]); cells.append(cl); cws.append(cw)
        assert b.code.encode_check(cw)
    tx = hti_interleave(np.concatenate(cells), T, 16200)
    for snr in (7.0, 7.6, 8.5):
        nv = 10 ** (-snr / 10)
        y = tx + np.sqrt(nv / 2) * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        blk = hti_deinterleave(y, T, 16200)[0]
        r = b.decode(blk, np.full(blk.shape, nv), iters=50)
        print(f'AWGN SNR {snr} dB:', [(x['ldpc_ok'], x['bch_ok'], x['iters'], bool(np.array_equal(x['info'][:b.Kbch], pay[i]))) for i, x in enumerate(r)])
