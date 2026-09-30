# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baseband packets -> ALP -> IPv4/UDP -> ROUTE (LCT) objects.

    route.py STORE_DIR packets.npz [packets.npz ...]

Writes every completed ROUTE object to STORE_DIR/<dst>_<port>/<tsi>/<toi>, the parts of the service layer
signalling (USBD, S-TSID, MPD ...) to STORE_DIR/<dst>_<port>/sls/, and STORE_DIR/index.json.
Transport only: objects are stored exactly as broadcast; nothing is decrypted or decoded.
"""
import sys, os, json, struct, gzip, zlib, re, collections
import numpy as np

BLOCKS_PER_FRAME = 160


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


def alp_packets(keys, data):
    """yield (key, header, payload) of every ALP packet that can be reassembled from contiguous baseband packets"""
    cur = None; prev = None
    for key, b in zip(keys, data):
        b = bytes(b); ptr, hl = bbp_parse(b); pay = b[hl:]
        seq = key[0] * BLOCKS_PER_FRAME + key[1]
        if prev is None or seq != prev + 1: cur = None
        prev = seq
        if ptr == 8191:
            if cur is not None: cur += pay
            continue
        stream = (bytes(cur) + pay[:ptr] + pay[ptr:]) if cur is not None else pay[ptr:]
        cur = None; pos = 0
        while pos < len(stream):
            r = alp_total_len(stream[pos:pos + 16])
            if r is None: cur = bytearray(stream[pos:]); break
            if r == 'ts': break
            h, l = r
            if l == 0: break
            if pos + h + l > len(stream): cur = bytearray(stream[pos:]); break
            yield key, stream[pos:pos + h], stream[pos + h:pos + h + l]
            pos += h + l


def udp_datagrams(keys, data):
    seg = {}
    def ip(key, p):
        if len(p) < 28 or (p[0] >> 4) != 4 or p[9] != 17: return None
        ihl = (p[0] & 15) * 4
        sp, dp, ul = struct.unpack('>HHH', p[ihl:ihl + 6])
        return key, '.'.join(map(str, p[12:16])), '.'.join(map(str, p[16:20])), dp, p[ihl + 8:ihl + ul]
    for key, h, p in alp_packets(keys, data):
        ptype = h[0] >> 5; pc = (h[0] >> 4) & 1; sc = (h[0] >> 3) & 1
        out = []
        if ptype == 0 and pc == 0: out.append(ip(key, p))
        elif ptype == 0 and pc == 1 and sc == 0:
            sn = h[2] >> 3; last = (h[2] >> 2) & 1
            if sn == 0: seg['buf'] = bytearray(p); seg['sn'] = 0
            elif 'buf' in seg and sn == seg['sn'] + 1:
                seg['buf'] += p; seg['sn'] = sn
                if last: out.append(ip(key, bytes(seg['buf']))); seg.clear()
            else: seg.clear()
        elif ptype == 0 and pc == 1 and sc == 1:
            cnt = (h[2] >> 1) & 7
            bits = ''.join(f'{v:08b}' for v in h[3:])
            o = 0
            for i in range(cnt + 1):
                ln = int(bits[12 * i:12 * i + 12], 2); out.append(ip(key, p[o:o + ln])); o += ln
        for r in out:
            if r: yield r


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


class Obj:
    __slots__ = ('buf', 'have', 'tol', 'cp', 'first_key', 'last_key', 'packets', 'closed')
    def __init__(self):
        self.buf = bytearray(); self.have = []; self.tol = None; self.cp = None; self.first_key = None; self.last_key = None; self.packets = 0; self.closed = False
    def add(self, start, payload, key):
        end = start + len(payload)
        if end > len(self.buf): self.buf.extend(b'\0' * (end - len(self.buf)))
        self.buf[start:end] = payload
        self.have.append((start, end)); self.packets += 1
        if self.first_key is None: self.first_key = key
        self.last_key = key
    def covered(self):
        n = 0; pos = 0
        for a, b in sorted(self.have):
            if b > pos: n += b - max(a, pos); pos = b
        return n
    def complete(self):
        if self.tol is None and self.closed:          # length not announced: the packet with the close flag is the end
            self.tol = len(self.buf)
        return self.tol is not None and self.covered() >= self.tol


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


def run(store, files):
    ks = []; ds = []
    for fn in files:
        z = np.load(fn)
        if len(z['keys']): ks.append(z['keys']); ds.append(z['data'])
    keys = np.concatenate(ks); data = np.concatenate(ds)
    o = np.lexsort((keys[:, 1], keys[:, 0])); keys = [tuple(int(v) for v in keys[i]) for i in o]; data = data[o]
    frames = sorted(set(k[0] for k in keys))
    objs = collections.OrderedDict(); flows = collections.Counter(); flow_bytes = collections.Counter(); notlct = collections.Counter()
    for key, src, dst, port, d in udp_datagrams(keys, data):
        f = '%s_%d' % (dst, port); flows[f] += 1; flow_bytes[f] += len(d)
        if dst == '224.0.23.60': continue
        r = lct(d)
        if r is None: notlct[f] += 1; continue
        ok = (f, r['tsi'], r['toi'])
        ob = objs.get(ok)
        if ob is None: ob = objs[ok] = Obj()
        if r['tol'] is not None: ob.tol = r['tol']
        ob.cp = r['cp']
        if r['close_obj']: ob.closed = True
        if r['payload']: ob.add(r['start'], r['payload'], key)
    index = dict(frames=[frames[0], frames[-1]] if frames else None, n_frames=len(frames), baseband_packets=len(keys),
                 flows={f: dict(datagrams=flows[f], bytes=flow_bytes[f], not_lct=notlct.get(f, 0)) for f in flows}, objects=[], sls={})
    span = (len(frames) * 0.2488889) or 1
    for f in index['flows']: index['flows'][f]['kbps_in_decoded_frames'] = round(flow_bytes[f] * 8 / span / 1e3, 1)
    for (f, tsi, toi), ob in objs.items():
        done = ob.complete()
        body = bytes(ob.buf[:ob.tol]) if done else None
        rec = dict(flow=f, tsi=tsi, toi=toi, codepoint=ob.cp, length=ob.tol, received=ob.covered(), complete=done, packets=ob.packets, closed=ob.closed, extent=len(ob.buf),
                   first_frame=ob.first_key[0] if ob.first_key else None)
        if done:
            d = os.path.join(store, f, str(tsi)); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, str(toi)), 'wb') as fh: fh.write(body)
            rec['magic'] = body[:12].hex()
            if body[4:8] in (b'ftyp', b'styp', b'moof', b'moov', b'sidx', b'emsg', b'prft'): rec['kind'] = 'isobmff ' + body[4:8].decode()
            if tsi == 0:
                raw, gz = maybe_gunzip(body)
                parts = []
                for hdr, pb in split_multipart(raw):           # signed package: the content, then its signature
                    if hdr.get('content-type', '').startswith('multipart/'):
                        parts += split_multipart(pb)
                    else:
                        parts.append((hdr, pb))
                sd = os.path.join(store, f, 'sls'); os.makedirs(sd, exist_ok=True)
                names = []
                for hdr, pb in parts:
                    pb, _ = maybe_gunzip(pb)
                    loc = hdr.get('content-location', 'part%d' % len(names))
                    name = re.sub(r'[^A-Za-z0-9._-]', '_', loc)
                    with open(os.path.join(sd, name), 'wb') as fh: fh.write(pb)
                    names.append(dict(name=name, content_type=hdr.get('content-type'), location=loc, bytes=len(pb)))
                if not parts:
                    with open(os.path.join(sd, 'toi%d.bin' % toi), 'wb') as fh: fh.write(raw)
                    names.append(dict(name='toi%d.bin' % toi, bytes=len(raw)))
                rec['sls_parts'] = names
                index['sls'].setdefault(f, {}); index['sls'][f][str(toi)] = names
        index['objects'].append(rec)
    os.makedirs(store, exist_ok=True)
    json.dump(index, open(os.path.join(store, 'index.json'), 'w'), indent=1)
    return index


def publish(store):
    """Give the objects the file names the manifests use (from each flow's S-TSID), and join each track's
    initialisation segment and whole media segments into one playable fragmented MP4."""
    import xml.etree.ElementTree as ET
    idx = json.load(open(os.path.join(store, 'index.json')))
    done = {(o['flow'], o['tsi'], o['toi']): o for o in idx['objects'] if o['complete']}
    pub = {}
    for flow in idx['flows']:
        sp = os.path.join(store, flow, 'sls', 'stsid.sls')
        if not os.path.exists(sp): continue
        root = ET.fromstring(open(sp, 'rb').read())
        fd = os.path.join(store, flow, 'files'); os.makedirs(fd, exist_ok=True)
        tracks = []
        for ls in root.iter():
            if not ls.tag.endswith('}LS'): continue
            tsi = int(ls.get('tsi')); tmpl = None; init = {}; kind = rep = None
            for e in ls.iter():
                t = e.tag.split('}')[-1]
                if t == 'FDT-Instance':
                    tmpl = next((v for k, v in e.attrib.items() if k.endswith('fileTemplate')), None)
                elif t == 'File': init[int(e.get('TOI'))] = e.get('Content-Location')
                elif t == 'MediaInfo': kind = e.get('contentType'); rep = e.get('repId')
            names = []
            for (f, ts, toi), o in sorted(done.items()):
                if f != flow or ts != tsi: continue
                name = init.get(toi) or (tmpl.replace('$TOI$', str(toi)) if tmpl else None)
                if not name: continue
                name = os.path.basename(name)
                src = os.path.join(store, flow, str(tsi), str(toi)); dst = os.path.join(fd, name)
                if os.path.lexists(dst): os.remove(dst)
                os.link(src, dst)
                names.append((toi, name, o['length'], toi in init))
            inits = [n for n in names if n[3]]; segs = [n for n in names if not n[3]]
            joined = None
            if inits and segs:
                joined = '%s-%s.mp4' % (kind or 'track', rep or tsi)
                with open(os.path.join(fd, joined), 'wb') as out:
                    out.write(open(os.path.join(fd, inits[0][1]), 'rb').read())
                    for s in segs: out.write(open(os.path.join(fd, s[1]), 'rb').read())
            tracks.append(dict(tsi=tsi, kind=kind, rep=rep, template=tmpl, init=inits[0][1] if inits else None,
                               segments=[dict(toi=s[0], name=s[1], bytes=s[2]) for s in segs], joined=joined))
        for n in os.listdir(os.path.join(store, flow, 'sls')):
            dst = os.path.join(fd, n)
            if os.path.lexists(dst): os.remove(dst)
            os.link(os.path.join(store, flow, 'sls', n), dst)
        pub[flow] = tracks
    idx['published'] = pub
    json.dump(idx, open(os.path.join(store, 'index.json'), 'w'), indent=1)
    return pub


if __name__ == '__main__':
    idx = run(sys.argv[1], sys.argv[2:])
    publish_after = True
    print('frames', idx['frames'], 'count', idx['n_frames'], 'baseband packets', idx['baseband_packets'])
    for f, v in sorted(idx['flows'].items(), key=lambda kv: -kv[1]['bytes']): print('  flow %-22s %6d datagrams %9d bytes  ~%8.1f kbit/s  non-LCT %d' % (f, v['datagrams'], v['bytes'], v['kbps_in_decoded_frames'], v['not_lct']))
    c = collections.Counter((o['flow'], o['tsi'], o['complete']) for o in idx['objects'])
    for k in sorted(c): print('  objects flow %s tsi %d complete=%s: %d' % (*k, c[k]))

    pub = publish(sys.argv[1])
    for f, tr in pub.items():
        for t in tr:
            if t['joined']: print('  playable', f, t['joined'], len(t['segments']), 'segment(s)')
