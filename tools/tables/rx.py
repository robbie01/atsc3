"""Receiver front half: frame -> equalised preamble cells."""
import numpy as np
import a3common as c, frame as f, chan

def load_frame(start_native, nsym=101):
    x = f.get_frame(start_native, nsym=nsym)
    cfo, _ = f.cp_cfo(x, nsym=nsym)
    x = x * np.exp(-2j * np.pi * cfo * np.arange(len(x)) / c.FS_A3)
    return x, cfo

def preamble_cells(x, cred_first=4):
    Y = f.demod_symbol(x, f.BS_LEN)
    noc = f.CARRIERS[cred_first]; shift = (f.NOC_MAX - noc) // 2
    Yr = Y[shift:shift + noc]
    H, n0, t, fitinfo = chan.preamble_estimate(Yr, cred_first)
    d = np.where(t == 'd')[0]
    eq = Yr[d] / H[d]
    nv = n0 / np.abs(H[d]) ** 2
    snr = np.mean(np.abs(H[d]) ** 2) / n0
    return f.freq_deinterleave(eq, 0), f.freq_deinterleave(nv, 0), dict(snr_db=10 * np.log10(snr), n0=n0, H=H, Yr=Yr, types=t, fit=fitinfo)
