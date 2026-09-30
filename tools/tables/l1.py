"""L1-Basic / L1-Detail FEC chain of ATSC A/322 (sect. 6.5): transmitter model (for self-test) and receiver.
Tables (shortening order, parity group permutation, LDPC addresses, BCH generator factors) are read from
the gr-atsc3 sources; structure per A/322."""
import numpy as np
import cparse, ldpc

SHORT = cparse.nested('framemapper_cc_impl.cc', 'shortening_table', int)
GROUP = cparse.nested('framemapper_cc_impl.cc', 'group_table', int)
NLDPC = 16200
BIG = 40.0

def _poly_tables():
    import re
    s = cparse.src('framemapper_cc_impl.cc')
    polys = re.findall(r'const int polys(\d+)\[\] = \{([^}]*)\}', s)
    out = []
    for n, body in polys[:12]:
        out.append([int(v) for v in body.split(',')])
    return out
def bch_generator():
    g = np.array([1], np.int64)
    for p in _poly_tables():
        g = np.convolve(g, np.array(p)) & 1
    return g            # index i = coefficient of x^i, degree 168
BCH_G = bch_generator()
assert len(BCH_G) == 169 and BCH_G[-1] == 1

def bch_parity(bits):
    """systematic BCH parity (168 bits) of message bits (msb first = highest power)"""
    reg = np.zeros(168, np.int64)     # reg[167] is the highest power
    g = BCH_G[:168]
    for b in bits:
        fb = int(b) ^ int(reg[167])
        reg[1:] = reg[:-1]
        reg[0] = 0
        if fb:
            reg ^= g
    return reg[::-1].astype(np.uint8)     # transmitted highest power first

def crc32(bits):
    crc = 0xffffffff
    for b in bits:
        x = int(b) ^ ((crc >> 31) & 1)
        crc = (crc << 1) & 0xffffffff
        if x: crc ^= 0x00210801 | 0x04C11DB7 * 0   # placeholder, replaced below
    return crc

def crc32(bits):
    # as in gr-atsc3 add_crc32_bits (CRC_POLY 0x00210801 with 32 bit register) -- see note in crc_check
    POLY = 0x00210801
    crc = 0xffffffff
    for b in bits:
        x = int(b) ^ ((crc >> 31) & 1)
        crc = (crc << 1) & 0xffffffff
        if x: crc ^= POLY
    return np.array([(crc >> n) & 1 for n in range(31, -1, -1)], np.uint8)

def scramble_bytes(n):
    sr = 0x18f; out = []
    for i in range(n):
        v = ((sr & 0x4) << 5) | ((sr & 0x8) << 3) | ((sr & 0x10) << 1) | ((sr & 0x20) >> 1) | \
            ((sr & 0x200) >> 6) | ((sr & 0x1000) >> 10) | ((sr & 0x2000) >> 12) | ((sr & 0x8000) >> 15)
        out.append(v)
        b = sr & 1
        sr >>= 1
        if b: sr ^= 0xd31c
    return np.array(out, np.uint8)
def scramble_bits(nbits):
    return np.unpackbits(scramble_bytes((nbits + 7) // 8))[:nbits]

# mode parameters: (ldpc kind, table index, Anum, Aden, B, bits/cell)
L1B_MODES = {1: ('a', 0, 0, 1, 9360, 2), 2: ('a', 0, 0, 1, 11460, 2), 3: ('a', 0, 0, 1, 12360, 2),
             4: ('a', 0, 0, 1, 12292, 4), 5: ('a', 0, 0, 1, 12350, 6), 6: ('a', 0, 0, 1, 12432, 8), 7: ('a', 0, 0, 1, 12766, 8)}
L1D_MODES = {1: ('a', 1, 7, 2, 0, 2), 2: ('a', 2, 2, 1, 6036, 2), 3: ('b', 3, 11, 16, 4653, 2),
             4: ('b', 4, 29, 32, 3200, 4), 5: ('b', 5, 3, 4, 4284, 6), 6: ('b', 6, 11, 16, 4900, 8), 7: ('b', 7, 49, 256, 8246, 8)}

def layout(ksig, mode, basic):
    kind, table, Anum, Aden, B, mod = (L1B_MODES if basic else L1D_MODES)[mode]
    code, tab, p = ldpc.l1_code(kind)
    nbch = p['K']; npar = NLDPC - nbch
    nouter = ksig + 168
    npad_total = nbch - nouter
    npad = npad_total // 360
    padbits = npad_total - 360 * npad
    padded = np.zeros(nbch, bool)
    for i in range(npad):
        g = SHORT[table][i]; padded[g * 360:(g + 1) * 360] = True
    if padbits:
        g = SHORT[table][npad]; padded[g * 360:g * 360 + padbits] = True
    info_pos = np.where(~padded)[0]
    assert len(info_pos) == nouter
    if basic:
        nrepeat = 3672 if mode == 1 else 0
    else:
        nrepeat = (2 * ((61 * nouter) // 16) - 508) if mode == 1 else 0
    npunctemp = (Anum * (nbch - nouter)) // Aden + B
    nfectemp = nouter + npar - npunctemp
    nfec = ((nfectemp + mod - 1) // mod) * mod
    npunc = npunctemp - (nfec - nfectemp)
    numbits = nfec + nrepeat
    ngroups = npar // 360
    gt = GROUP[table][:ngroups]
    # interleaved parity index -> codeword index
    ip2cw = np.concatenate([np.arange(g * 360, g * 360 + 360) for g in gt])
    return dict(kind=kind, code=code, tab=tab, p=p, nbch=nbch, npar=npar, nouter=nouter, info_pos=info_pos,
                padded=padded, nrepeat=nrepeat, npunc=npunc, numbits=numbits, mod=mod, ip2cw=ip2cw,
                cells=numbits // mod, ksig=ksig)

def tx_bits(msg_bits, mode, basic):
    """msg_bits: signalling bits including CRC (ksig). Returns transmitted bit sequence before the block
    interleaver (A/322 6.5: scramble, BCH, zero pad, LDPC, parity permutation, repetition, puncturing, zero removal)."""
    L = layout(len(msg_bits), mode, basic)
    sc = np.asarray(msg_bits, np.uint8) ^ scramble_bits(len(msg_bits))
    outer = np.concatenate([sc, bch_parity(sc)])
    info = np.zeros(L['nbch'], np.uint8); info[L['info_pos']] = outer
    if L['kind'] == 'a':
        cw = ldpc.encode_type_a(info, L['tab'], **L['p'])
    else:
        cw = ldpc.encode_type_b(info, L['tab'], **L['p'])
    ipar = cw[L['ip2cw']]
    rep = ipar[np.arange(L['nrepeat']) % L['npar']]
    t = np.concatenate([outer, rep, ipar[:L['npar'] - L['npunc']]])
    assert len(t) == L['numbits']
    return t, L

def qpsk_map(t):
    rows = len(t) // 2
    b0 = t[:rows].astype(float); b1 = t[rows:].astype(float)
    return ((1 - 2 * b1) + 1j * (1 - 2 * b0)) / np.sqrt(2)

def qpsk_llr(cells, nvar):
    """cells equalised (unit power constellation), nvar = noise variance per cell (complex). returns LLR of the
    transmit bit sequence t (before block interleaver)"""
    s = 2 * np.sqrt(2) / nvar
    b0 = s * cells.imag; b1 = s * cells.real
    return np.concatenate([b0, b1])

def rx_decode(llr_t, ksig, mode, basic, iters=30):
    L = layout(ksig, mode, basic)
    assert len(llr_t) == L['numbits'], (len(llr_t), L['numbits'])
    cw = np.zeros(NLDPC)
    cw[:L['nbch']][L['padded']] = BIG
    cw[L['info_pos']] = llr_t[:L['nouter']]
    ipar = np.zeros(L['npar'])
    r = llr_t[L['nouter']:L['nouter'] + L['nrepeat']]
    np.add.at(ipar, np.arange(L['nrepeat']) % L['npar'], r)
    rest = llr_t[L['nouter'] + L['nrepeat']:]
    ipar[:len(rest)] += rest
    cw[L['ip2cw']] += ipar
    hard, ok, nit = L['code'].decode(cw[None, :], iters)
    outer = hard[0][L['info_pos']]
    sc = outer[:ksig]
    bch_ok = bool(np.array_equal(bch_parity(sc), outer[ksig:]))
    msg = sc ^ scramble_bits(ksig)
    crc_ok = bool(np.array_equal(crc32(msg[:-32]), msg[-32:]))
    return dict(msg=msg, ldpc_ok=bool(ok[0]), iters=int(nit[0]), bch_ok=bch_ok, crc_ok=crc_ok,
                pad_ok=not hard[0][:L['nbch']][L['padded']].any())

class Bits:
    def __init__(self, b): self.b = np.asarray(b); self.i = 0
    def u(self, n):
        v = 0
        for x in self.b[self.i:self.i + n]: v = (v << 1) | int(x)
        self.i += n
        return v

def parse_l1b(msg):
    r = Bits(msg); d = {}
    d['L1B_version'] = r.u(3); d['L1B_mimo_scattered_pilot_encoding'] = r.u(1); d['L1B_lls_flag'] = r.u(1)
    d['L1B_time_info_flag'] = r.u(2); d['L1B_return_channel_flag'] = r.u(1); d['L1B_papr_reduction'] = r.u(2)
    d['L1B_frame_length_mode'] = r.u(1)
    if d['L1B_frame_length_mode'] == 0:
        d['L1B_frame_length'] = r.u(10); d['L1B_excess_samples_per_symbol'] = r.u(13)
    else:
        d['L1B_time_offset'] = r.u(16); d['L1B_additional_samples'] = r.u(7)
    d['L1B_num_subframes'] = r.u(8); d['L1B_preamble_num_symbols'] = r.u(3); d['L1B_preamble_reduced_carriers'] = r.u(3)
    d['L1B_L1_Detail_content_tag'] = r.u(2); d['L1B_L1_Detail_size_bytes'] = r.u(13); d['L1B_L1_Detail_fec_type'] = r.u(3)
    d['L1B_L1_Detail_additional_parity_mode'] = r.u(2); d['L1B_L1_Detail_total_cells'] = r.u(19)
    d['L1B_first_sub_mimo'] = r.u(1); d['L1B_first_sub_miso'] = r.u(2); d['L1B_first_sub_fft_size'] = r.u(2)
    d['L1B_first_sub_reduced_carriers'] = r.u(3); d['L1B_first_sub_guard_interval'] = r.u(4)
    d['L1B_first_sub_num_ofdm_symbols'] = r.u(11); d['L1B_first_sub_scattered_pilot_pattern'] = r.u(5)
    d['L1B_first_sub_scattered_pilot_boost'] = r.u(3); d['L1B_first_sub_sbs_first'] = r.u(1); d['L1B_first_sub_sbs_last'] = r.u(1)
    d['L1B_reserved'] = r.u(48); d['L1B_crc'] = r.u(32)
    assert r.i == 200
    return d

def parse_l1d(msg, l1b):
    r = Bits(msg); d = {}
    d['L1D_version'] = r.u(4); d['L1D_num_rf'] = r.u(3)
    d['bonded'] = []
    for _ in range(d['L1D_num_rf']):
        d['bonded'].append(r.u(16)); r.u(3)
    tif = l1b['L1B_time_info_flag']
    if tif != 0:
        d['L1D_time_sec'] = r.u(32); d['L1D_time_msec'] = r.u(10)
        if tif != 1:
            d['L1D_time_usec'] = r.u(10)
            if tif != 2:
                d['L1D_time_nsec'] = r.u(10)
    d['subframes'] = []
    for i in range(l1b['L1B_num_subframes'] + 1):
        s = {}
        if i > 0:
            s['L1D_mimo'] = r.u(1); s['L1D_miso'] = r.u(2); s['L1D_fft_size'] = r.u(2); s['L1D_reduced_carriers'] = r.u(3)
            s['L1D_guard_interval'] = r.u(4); s['L1D_num_ofdm_symbols'] = r.u(11); s['L1D_scattered_pilot_pattern'] = r.u(5)
            s['L1D_scattered_pilot_boost'] = r.u(3); s['L1D_sbs_first'] = r.u(1); s['L1D_sbs_last'] = r.u(1)
            sbs = s['L1D_sbs_first'] | s['L1D_sbs_last']; mimo = s['L1D_mimo']
        else:
            sbs = l1b['L1B_first_sub_sbs_first'] | l1b['L1B_first_sub_sbs_last']; mimo = l1b['L1B_first_sub_mimo']
        if l1b['L1B_num_subframes'] > 0:
            s['L1D_subframe_multiplex'] = r.u(1)
        s['L1D_frequency_interleaver'] = r.u(1)
        if sbs:
            s['L1D_sbs_null_cells'] = r.u(13)
        s['L1D_num_plp'] = r.u(6)
        s['plps'] = []
        for j in range(s['L1D_num_plp'] + 1):
            p = {}
            p['L1D_plp_id'] = r.u(6); p['L1D_plp_lls_flag'] = r.u(1); p['L1D_plp_layer'] = r.u(2)
            p['L1D_plp_start'] = r.u(24); p['L1D_plp_size'] = r.u(24); p['L1D_plp_scrambler_type'] = r.u(2)
            p['L1D_plp_fec_type'] = r.u(4)
            if p['L1D_plp_fec_type'] <= 5:
                p['L1D_plp_mod'] = r.u(4); p['L1D_plp_cod'] = r.u(4)
            p['L1D_plp_TI_mode'] = r.u(2)
            if p['L1D_plp_TI_mode'] == 0:
                p['L1D_plp_fec_block_start'] = r.u(15)
            elif p['L1D_plp_TI_mode'] == 1:
                p['L1D_plp_CTI_fec_block_start'] = r.u(22)
            if d['L1D_num_rf'] > 0:
                p['L1D_plp_num_channel_bonded'] = r.u(3)
                if p['L1D_plp_num_channel_bonded'] > 0:
                    p['L1D_plp_channel_bonding_format'] = r.u(2)
                    p['bonded_rf'] = [r.u(3) for _ in range(p['L1D_plp_num_channel_bonded'] + 1)]
            if mimo:
                p['L1D_plp_mimo_stream_combining'] = r.u(1); p['L1D_plp_mimo_IQ_interleaving'] = r.u(1); p['L1D_plp_mimo_PH'] = r.u(1)
            if p['L1D_plp_layer'] == 0:
                p['L1D_plp_type'] = r.u(1)
                if p['L1D_plp_type'] == 1:
                    p['L1D_plp_num_subslices'] = r.u(14); p['L1D_plp_subslice_interval'] = r.u(24)
                if p['L1D_plp_TI_mode'] in (1, 2) and p.get('L1D_plp_mod') == 0:
                    p['L1D_plp_TI_extended_interleaving'] = r.u(1)
                if p['L1D_plp_TI_mode'] == 1:
                    p['L1D_plp_CTI_depth'] = r.u(3); p['L1D_plp_CTI_start_row'] = r.u(11)
                elif p['L1D_plp_TI_mode'] == 2:
                    p['L1D_plp_HTI_inter_subframe'] = r.u(1); p['L1D_plp_HTI_num_ti_blocks'] = r.u(4)
                    p['L1D_plp_HTI_num_fec_blocks_max'] = r.u(12)
                    if p['L1D_plp_HTI_inter_subframe'] == 0:
                        p['L1D_plp_HTI_num_fec_blocks'] = r.u(12)
                    else:
                        p['L1D_plp_HTI_num_fec_blocks'] = [r.u(12) for _ in range(p['L1D_plp_HTI_num_ti_blocks'] + 1)]
                    p['L1D_plp_HTI_cell_interleaver'] = r.u(1)
            else:
                p['L1D_plp_ldm_injection_level'] = r.u(5)
            s['plps'].append(p)
        d['subframes'].append(s)
    d['L1D_bsid'] = r.u(16)
    d['reserved_bits'] = len(msg) - 32 - r.i
    d['parsed_bits'] = r.i
    return d

if __name__ == '__main__':
    # self-test: encode -> QPSK -> noise -> decode, L1-Basic mode 1 and L1-Detail modes 1..3
    rng = np.random.default_rng(3)
    for basic, mode, ksig, snr in ((True, 1, 200, -8.0), (True, 3, 200, -1.0), (False, 1, 400, -8.0), (False, 1, 2352, -7.0), (False, 2, 800, -2.0), (False, 3, 1200, 1.0)):
        msg = rng.integers(0, 2, ksig - 32).astype(np.uint8)
        msg = np.concatenate([msg, crc32(msg)])
        t, L = tx_bits(msg, mode, basic)
        cells = qpsk_map(t)
        nv = 10 ** (-snr / 10)
        y = cells + np.sqrt(nv / 2) * (rng.standard_normal(len(cells)) + 1j * rng.standard_normal(len(cells)))
        r = rx_decode(qpsk_llr(y, nv), ksig, mode, basic)
        print(f"basic={basic} mode={mode} ksig={ksig} cells={L['cells']} nrepeat={L['nrepeat']} npunc={L['npunc']} SNR={snr} dB -> "
              f"ldpc_ok={r['ldpc_ok']} bch_ok={r['bch_ok']} crc_ok={r['crc_ok']} match={np.array_equal(r['msg'], msg)} iters={r['iters']}")
