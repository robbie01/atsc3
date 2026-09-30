# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baseband packets -> ALP -> IPv4/UDP -> ROUTE (LCT): the parsers the gateway builds on.
Transport only: nothing is decrypted or decoded.
"""
import struct, gzip, zlib, re


def bbp_parse(b):
    b0 = b[0]; mode = b0 >> 7; ptr = b0 & 0x7f; hl = 1
    if mode:
        b1 = b[1]; ptr |= (b1 >> 2) << 7; ofi = b1 & 3; hl = 2
        if ofi == 1: hl = 3 + (b[2] & 31)
        elif ofi == 2: hl = 4 + ((b[2] & 31) | (b[3] << 5))
        elif ofi == 3: hl = 4 + ((b[2] & 31) | (b[3] << 5))
    return ptr, hl


def alp_total_len(buf):
    if len(buf) < 2: return None
    ptype = buf[0] >> 5; pc = (buf[0] >> 4) & 1; hm = (buf[0] >> 3) & 1
    length = ((buf[0] & 7) << 8) | buf[1]
    if ptype == 7: return 'ts'
    hl = 2; sig = 5 if ptype == 4 else 0
    if pc == 0 and hm == 0: return hl + sig, length
    if len(buf) < 3: return None
    a = buf[2]; hl = 3 + sig
    if pc == 0:
        length |= (a >> 3) << 11; sif = (a >> 1) & 1; hef = a & 1
    elif hm == 0:
        sif = (a >> 1) & 1; hef = a & 1
    else:
        length |= (a >> 4) << 11; cnt = (a >> 1) & 7; sif = a & 1; hef = 0
        hl += (3 * (cnt + 1) + 1) // 2
    if sif: hl += 1
    if hef:
        if len(buf) < hl + 2: return None
        hl += 2 + (buf[hl + 1] + 1)
    return hl, length


def lct(d):
    """parse an ALC/LCT packet as ROUTE uses it; returns dict or None"""
    if len(d) < 8 or (d[0] >> 4) != 1: return None
    c = (d[0] >> 2) & 3; psi = d[0] & 3
    s = d[1] >> 7; o = (d[1] >> 5) & 3; h = (d[1] >> 4) & 1; a = (d[1] >> 1) & 1; b = d[1] & 1
    hl = d[2] * 4; cp = d[3]
    p = 4 + 4 * (c + 1)
    tl = 4 * s + 2 * h; ol = 4 * o + 2 * h
    if hl > len(d) or p + tl + ol > hl: return None
    tsi = int.from_bytes(d[p:p + tl], 'big'); p += tl
    toi = int.from_bytes(d[p:p + ol], 'big'); p += ol
    ext = {}
    while p < hl:
        het = d[p]
        if het < 128:
            n = d[p + 1] * 4
            if n == 0: break
            ext[het] = d[p + 2:p + n]; p += n
        else:
            ext[het] = d[p + 1:p + 4]; p += 4
    if len(d) < hl + 4: return None
    start = int.from_bytes(d[hl:hl + 4], 'big')
    tol = None
    if 194 in ext: tol = int.from_bytes(ext[194], 'big')
    elif 67 in ext: tol = int.from_bytes(ext[67][:6], 'big')
    elif 64 in ext and len(ext[64]) >= 6: tol = int.from_bytes(ext[64][:6], 'big')     # EXT_FTI: transfer length first
    return dict(tsi=tsi, toi=toi, cp=cp, psi=psi, close_obj=b, close_sess=a, start=start, tol=tol, payload=d[hl + 4:], ext=sorted(ext))


def split_multipart(body, ctype_hint=None):
    """SLS objects are multipart/related packages; returns list of (headers dict, bytes)"""
    m = re.search(rb'boundary="?([^";\r\n]+)"?', body[:400])
    if not m:
        m = re.match(rb'\s*--([^\r\n]+)\r?\n', body)
    if not m: return []
    bnd = b'--' + m.group(1).strip()
    parts = []
    for chunk in body.split(bnd)[1:]:
        if chunk.startswith(b'--'): break
        chunk = chunk.lstrip(b'\r\n')
        sep = chunk.find(b'\r\n\r\n'); k = 4
        if sep < 0: sep = chunk.find(b'\n\n'); k = 2
        if sep < 0: continue
        hdr = {}
        for line in chunk[:sep].decode('latin1').splitlines():
            if ':' in line:
                a, b = line.split(':', 1); hdr[a.strip().lower()] = b.strip()
        parts.append((hdr, chunk[sep + k:].rstrip(b'\r\n')))
    return parts


def maybe_gunzip(b):
    if b[:2] == b'\x1f\x8b':
        try: return gzip.decompress(b), True
        except Exception:
            try: return zlib.decompressobj(47).decompress(b), True
            except Exception: pass
    return b, False
