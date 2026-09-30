#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live HTTP gateway to an ATSC 3.0 signal: HDHomeRun-style lineup, MPEG-TS / Matroska / HLS streams, trampolines
for the internet-delivered channels and the broadcaster apps, the programme guide, a signal meter and an outage log.

    uv run --project ~/atsc3 python ~/atsc3/gateway/live.py [--gain 34] [--port 8023] [--bind 0.0.0.0] [--file capture.ci16]

The physical layer runs in ../a3rx (Rust); this program reads its baseband packets and does the rest.
Transport only: objects are served exactly as broadcast.  Nothing is decrypted.
Paths can be overridden with the environment: A3RX, AC3CLI, A3_STATE.
"""
import argparse, collections, gzip, json, os, re, struct, subprocess, threading, time, zlib
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import route

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A3RX = os.environ.get('A3RX') or os.path.join(ROOT, 'a3rx', 'target', 'release', 'a3rx')
# ac3forge: a clean-room AC-4 decoder that handles every layout, the immersive (Atmos) presentations included
AC3CLI = os.environ.get('AC3CLI') or os.path.join(ROOT, 'third_party', 'ac3forge', 'build', 'llvm', 'bin', 'ac3cli')
STATE = os.environ.get('A3_STATE') or os.path.join(ROOT, 'state')
os.makedirs(STATE, exist_ok=True)
NTP = 2208988800
KEEP_SEGMENTS = 60
DEBUG = bool(os.environ.get('GW_DEBUG'))
NEED_MAIN, NEED_ROBUST = 17.5, 10.0
BLOCKS = {0: 4, 1: 160}
lock = threading.RLock()


def strip(tag):
    return tag.split("}")[-1]


class Alp:
    """Baseband packets of one pipe -> ALP packets -> UDP datagrams."""

    def __init__(self, blocks):
        self.blocks = blocks; self.cur = None; self.prev = None; self.seg = {}

    def feed(self, frame, block, b):
        ptr, hl = route.bbp_parse(b); pay = b[hl:]
        seq = frame * self.blocks + block
        if self.prev is None or seq != self.prev + 1:
            self.cur = None; self.seg.clear()
        self.prev = seq
        if ptr == 8191:
            if self.cur is not None: self.cur += pay
            return
        stream = (bytes(self.cur) + pay) if self.cur is not None else pay[ptr:]
        self.cur = None; pos = 0
        while pos < len(stream):
            r = route.alp_total_len(stream[pos:pos + 16])
            if r is None: self.cur = bytearray(stream[pos:]); break
            if r == 'ts': break
            h, l = r
            if l == 0: break
            if pos + h + l > len(stream): self.cur = bytearray(stream[pos:]); break
            yield from self.alp(stream[pos:pos + h], stream[pos + h:pos + h + l])
            pos += h + l

    def alp(self, h, p):
        ptype = h[0] >> 5; pc = (h[0] >> 4) & 1; sc = (h[0] >> 3) & 1
        if ptype == 4:
            yield ('lmt', bytes(h[-5:]) + bytes(p)); return
        if ptype != 0: return
        if pc == 0: yield from self.ip(p)
        elif sc == 0:
            sn = h[2] >> 3; last = (h[2] >> 2) & 1
            if sn == 0: self.seg = {'buf': bytearray(p), 'sn': 0}
            elif self.seg.get('sn') == sn - 1:
                self.seg['buf'] += p; self.seg['sn'] = sn
                if last:
                    yield from self.ip(bytes(self.seg['buf'])); self.seg = {}
            else: self.seg = {}
        else:
            cnt = (h[2] >> 1) & 7
            bits = ''.join(f'{v:08b}' for v in h[3:]); o = 0
            for i in range(cnt + 1):
                ln = int(bits[12 * i:12 * i + 12], 2)
                yield from self.ip(p[o:o + ln]); o += ln

    @staticmethod
    def ip(p):
        if len(p) < 28 or (p[0] >> 4) != 4 or p[9] != 17: return
        ihl = (p[0] & 15) * 4
        sp, dp, ul = struct.unpack('>HHH', p[ihl:ihl + 6])
        yield ('udp', '%d.%d.%d.%d' % tuple(p[16:20]), dp, p[ihl + 8:ihl + ul])


class Obj:
    __slots__ = ('buf', 'got', 'tol', 'cp', 'born', 'spans', 'closed')

    def __init__(self):
        self.buf = bytearray(); self.got = 0; self.tol = None; self.cp = None; self.born = time.time(); self.spans = None; self.closed = False

    def add(self, start, payload):
        end = start + len(payload)
        if self.spans is None and start == len(self.buf):          # the usual case: in order
            self.buf += payload; self.got = end; return
        if self.spans is None:
            self.spans = [(0, len(self.buf))]
        if end > len(self.buf): self.buf.extend(b'\0' * (end - len(self.buf)))
        self.buf[start:end] = payload
        self.spans.append((start, end)); n = 0; pos = 0
        for a, b in sorted(self.spans):
            if b > pos: n += b - max(a, pos); pos = b
        self.got = n

    def complete(self):
        if self.tol is None and self.closed: self.tol = len(self.buf)
        return self.tol is not None and self.got >= self.tol


class Flow:
    def __init__(self, name):
        self.name = name
        self.open = {}                       # (tsi, toi) -> Obj
        self.done_sls = set()
        self.sls = {}                        # name -> bytes
        self.ls = {}                         # tsi -> dict(template, files{toi: (name, enc)}, kind, rep)
        self.tracks = collections.defaultdict(lambda: dict(init=None, segs=collections.OrderedDict()))
        self.files = {}                      # name -> (bytes, time)
        self.bytes = 0; self.rate = collections.deque(maxlen=64); self.last = 0.0; self.quiet = False
        self.reps = {}                       # repId -> dict from the MPD
        self.internet = False                # the manifest points at the broadcaster's servers, not at the broadcast

    def parse_stsid(self):
        try: root = ET.fromstring(self.sls['stsid.sls'])
        except Exception: return
        ls = {}
        for e in root.iter():
            if strip(e.tag) != 'LS': continue
            d = dict(template=None, files={}, kind=None, rep=None)
            for c in e.iter():
                t = strip(c.tag)
                if t == 'FDT-Instance': d['template'] = next((v for k, v in c.attrib.items() if k.endswith('fileTemplate')), None)
                elif t == 'File': d['files'][int(c.get('TOI'))] = (c.get('Content-Location'), c.get('Content-Encoding'))
                elif t == 'MediaInfo': d['kind'] = c.get('contentType'); d['rep'] = c.get('repId')
            ls[int(e.get('tsi'))] = d
        self.ls = ls

    def parse_mpd(self):
        try: root = ET.fromstring(self.sls['mpd.mpd'])
        except Exception: return
        reps = {}
        anybase = next(((e.text or '').strip() for e in root.iter() if strip(e.tag) == 'BaseURL' and (e.text or '').strip().startswith('http')), None)
        for a in root.iter():
            if strip(a.tag) != 'AdaptationSet': continue
            at = next((c for c in a if strip(c.tag) == 'SegmentTemplate'), None)
            for r in a:
                if strip(r.tag) != 'Representation': continue
                st = next((c for c in r if strip(c.tag) == 'SegmentTemplate'), at)
                g = lambda k: r.get(k) or a.get(k)
                dur = None; line = []; ts = 1
                if st is not None:
                    ts = int(st.get('timescale') or 1)
                    if st.get('duration'): dur = int(st.get('duration')) / ts
                    for e in st.iter():
                        if strip(e.tag) != 'S': continue
                        t = int(e.get('t')) if e.get('t') else (line[-1][0] + line[-1][1] if line else 0)
                        for k in range(int(e.get('r') or 0) + 1): line.append((t + k * int(e.get('d')), int(e.get('d'))))
                    if line and dur is None: dur = line[0][1] / ts
                base = next(((c.text or '').strip() for c in list(r) + list(a) if strip(c.tag) == 'BaseURL'), None) or anybase
                acc = next((c.get('value') for c in list(r) + list(a) if strip(c.tag) == 'AudioChannelConfiguration'), None)
                rid = r.get('id')
                fill = lambda v: v.replace('$RepresentationID$', rid) if v else v
                reps[rid] = dict(codecs=g('codecs'), width=g('width'), height=g('height'), frame_rate=g('frameRate'), mime=g('mimeType'),
                                 bandwidth=int(r.get('bandwidth') or 0), lang=a.get('lang'), kind=a.get('contentType') or (g('mimeType') or '').split('/')[0], duration=dur,
                                 base=base, init=fill(st.get('initialization')) if st is not None else None, media=fill(st.get('media')) if st is not None else None,
                                 timescale=ts, timeline=line, cicp=int(acc) if acc and acc.isdigit() else None)
        self.reps = reps
        self.internet = bool(anybase)
        ev = [e for e in root.iter() if strip(e.tag) in ('EventStream', 'InbandEventStream')]
        sig = [(strip(e.tag), e.get('schemeIdUri'), e.get('value'), len(list(e))) for e in ev]
        if sig != getattr(self, 'events_seen', []):
            self.events_seen = sig
            if sig: OUT.note('manifest events on %s: %s' % (self.name.replace('_', ':'), sig))


class State:
    def __init__(self):
        self.flows = {}
        self.lls = {}                        # name -> (xml text, time)
        self.services = []
        self.lmt = []
        self.stat = {}; self.hist = collections.deque(maxlen=2400)
        self.guide = dict(services={}, content={}, schedule={}, updated=None)
        self.started = time.time(); self.frames = 0; self.decoder = 'starting'
        self.alp = {p: Alp(n) for p, n in BLOCKS.items()}

    def flow(self, name):
        f = self.flows.get(name)
        if f is None:
            f = self.flows[name] = Flow(name)
            if self.started and time.time() - self.started > 20: OUT.note('new flow: first packet on %s' % name.replace('_', ':'))
        return f

    # ---- low level signalling
    def on_lls(self, d):
        if len(d) < 5: return
        tid, gid, ver = d[0], d[1], d[3]; body = d[4:]
        def un(b):
            try: return gzip.decompress(b)
            except Exception:
                try: return zlib.decompress(b, 47)
                except Exception: return None
        names = {1: 'SLT', 2: 'RRT', 3: 'SystemTime', 4: 'AEAT', 5: 'OnscreenMessageNotification', 6: 'CertificationData', 0xff: 'UserDefined'}
        items = []
        if tid == 0xfe:
            try:
                n = body[0]; o = 1
                for i in range(n):
                    pid, pver, plen = body[o], body[o + 1], (body[o + 2] << 8) | body[o + 3]
                    items.append((pid, pver, un(body[o + 4:o + 4 + plen]))); o += 4 + plen
            except IndexError: pass
        else:
            items.append((tid, ver, un(body)))
        for pid, pver, x in items:
            if x is None: continue
            name = names.get(pid, 'table%d' % pid)
            old = self.lls.get(name)
            if old and old[2] == pver and old[0] == x: 
                self.lls[name] = (x, time.time(), pver); continue
            self.lls[name] = (x, time.time(), pver)
            if pid in (4, 5): OUT.note('ALERT TABLE %s version %d: %s' % (name, pver, x.decode('utf-8', 'replace')[:2000]))
            elif pid != 3 and (old is None or old[2] != pver) and time.time() - self.started > 20:
                OUT.note('LLS %s %s version %d' % (name, 'appeared,' if old is None else 'changed to', pver))
            if pid == 1: self.parse_slt(x)

    def parse_slt(self, x):
        try: root = ET.fromstring(x)
        except Exception: return
        out = []
        for s in root.iter():
            if strip(s.tag) != 'Service': continue
            b = next((c for c in s if strip(c.tag) == 'BroadcastSvcSignaling'), None)
            out.append(dict(id=s.get('serviceId'), name=s.get('shortServiceName'), major=s.get('majorChannelNo'), minor=s.get('minorChannelNo'),
                            category=int(s.get('serviceCategory') or 0), gsid=s.get('globalServiceID'), broadband=s.get('broadbandAccessRequired') == 'true',
                            configuration=s.get('configuration'),
                            flow='%s_%s' % (b.get('slsDestinationIpAddress'), b.get('slsDestinationUdpPort')) if b is not None else None))
        self.services = out

    def on_lmt(self, p):
        if p[0] != 1: return
        try:
            b = p[5:]; o = 0; out = []
            nplp = (b[o] >> 2) + 1; o += 1
            for i in range(nplp):
                pid = b[o] >> 2; nm = b[o + 1]; o += 2
                for j in range(nm):
                    dst = '.'.join(map(str, b[o + 4:o + 8])); sp, dp = struct.unpack('>HH', b[o + 8:o + 12]); fl = b[o + 12]; o += 13
                    if fl & 0x80: o += 1
                    if fl & 0x40: o += 1
                    out.append(dict(plp=pid, dst=dst, port=dp))
            self.lmt = out
        except (IndexError, struct.error): pass

    # ---- ROUTE
    def on_udp(self, dst, port, d):
        if dst == '224.0.23.60':
            return self.on_lls(d)
        f = self.flow('%s_%d' % (dst, port))
        f.bytes += len(d); f.last = time.time()
        r = route.lct(d)
        if r is None: return
        key = (r['tsi'], r['toi'])
        if r['tsi'] == 0 and key in f.done_sls: return
        if r['tsi'] != 0 and (key[1] in f.tracks[r['tsi']]['segs'] or (f.tracks[r['tsi']]['init'] or (None,))[0] == key[1]): return
        o = f.open.get(key)
        if o is None: o = f.open[key] = Obj()
        if r['tol'] is not None: o.tol = r['tol']
        if r['close_obj']: o.closed = True
        o.cp = r['cp']
        if r['payload']: o.add(r['start'], r['payload'])
        if o.complete():
            del f.open[key]
            self.on_object(f, r['tsi'], r['toi'], bytes(o.buf[:o.tol]))

    def on_emsg(self, f, tsi, body):
        """DASH in-band event messages (emsg boxes: SCTE-35 splices, app events...) in a media segment"""
        i = body.find(b'emsg')
        while 0 <= i < len(body):
            n = struct.unpack('>I', body[i - 4:i])[0] if i >= 4 else 0
            box = body[i + 4:i - 4 + n] if n > 8 else b''
            ver = box[0] if box else 0
            try:
                if ver == 0:
                    parts = box[4:].split(b'\0', 2); scheme, value = parts[0].decode(), parts[1].decode()
                    rest = parts[2]; tsc, pt, dur, eid = struct.unpack('>IIII', rest[:16]); data = rest[16:]
                else:
                    tsc, pt, dur, eid = struct.unpack('>IQII', box[4:24]); parts = box[24:].split(b'\0', 2); scheme, value = parts[0].decode(), parts[1].decode(); data = parts[2]
                OUT.note('in-band event on %s track %d: scheme %s value %s time %s dur %s id %d data %s' % (f.name.replace('_', ':'), tsi, scheme, value, pt, dur, eid, data[:200].hex()))
            except Exception as e: OUT.note('in-band event on %s track %d (unparsed: %s)' % (f.name.replace('_', ':'), tsi, e))
            i = body.find(b'emsg', i + 4)

    def on_object(self, f, tsi, toi, body):
        now = time.time()
        if tsi == 0:
            f.done_sls.add((tsi, toi))
            if len(f.done_sls) > 64: f.done_sls = set(list(f.done_sls)[-32:])
            if toi == 0: return
            raw, _ = route.maybe_gunzip(body); parts = []
            for hdr, pb in route.split_multipart(raw):
                if hdr.get('content-type', '').startswith('multipart/'): parts += route.split_multipart(pb)
                else: parts.append((hdr, pb))
            for hdr, pb in parts:
                pb, _ = route.maybe_gunzip(pb)
                loc = hdr.get('content-location')
                if loc: f.sls[os.path.basename(loc)] = pb
            if 'stsid.sls' in f.sls: f.parse_stsid()
            if 'mpd.mpd' in f.sls: f.parse_mpd()
            return
        ls = f.ls.get(tsi)
        named = ls['files'].get(toi) if ls else None
        if body[4:8] == b'ftyp':
            f.tracks[tsi]['init'] = (toi, body); return
        if body[4:8] in (b'styp', b'sidx', b'moof', b'emsg', b'prft'):
            segs = f.tracks[tsi]['segs']; segs[toi] = (body, now)
            if b'emsg' in body[:4096] or b'emsg' in body[-4096:]: self.on_emsg(f, tsi, body)
            while len(segs) > KEEP_SEGMENTS: segs.popitem(last=False)
            return
        if named:
            name, enc = named
            if enc == 'gzip': body, _ = route.maybe_gunzip(body)
            f.files[os.path.basename(name)] = (body, now)
            if name.startswith('sgdu') or name.startswith('sgdd'): self.on_esg(name, body)
            f.tracks[tsi]['segs'][toi] = (b'', now)          # remember it as seen, without keeping a second copy
            while len(f.tracks[tsi]['segs']) > 400: f.tracks[tsi]['segs'].popitem(last=False)

    def on_esg(self, name, raw):
        if not name.startswith('sgdu'): return
        txt = raw.decode('utf-8', 'replace')
        g = self.guide
        for part in txt.split('<?xml')[1:]:
            end = part.rfind('>')
            try: e = ET.fromstring(('<?xml' + part[:end + 1]).encode('utf-8'))
            except Exception: continue
            t = strip(e.tag); kids = {strip(c.tag): c for c in e}
            if t == 'Service':
                ext = {strip(c.tag): (c.text or '').strip() for c in e.iter()}
                # fragment ids end in a generation timestamp (urn:digicap:svc:5004:1790741945371): a new one supersedes the old
                stem = e.get('id', '').rsplit(':', 1)[0]
                for old in [k for k in g['services'] if k.rsplit(':', 1)[0] == stem and k != e.get('id')]:
                    del g['services'][old]; g['schedule'].pop(old, None)
                g['services'][e.get('id')] = dict(id=e.get('id'), name=kids['Name'].get('text') if 'Name' in kids else None, gsid=e.get('globalServiceID'),
                                                  major=ext.get('MajorChannelNum'), minor=ext.get('MinorChannelNum'), icon=ext.get('Icon'))
            elif t == 'Content':
                icon = next(((c.text or '').strip() for c in e.iter() if strip(c.tag) == 'ContentIcon'), None)
                rating = next(((c.text or '').strip() for c in e.iter() if strip(c.tag) == 'RatingValueString'), None)
                g['content'][e.get('id')] = dict(title=kids['Name'].get('text') if 'Name' in kids else None,
                                                 description=kids['Description'].get('text') if 'Description' in kids else None,
                                                 length=(kids['Length'].text if 'Length' in kids else None), icon=icon, rating=rating)
            elif t == 'Schedule':
                sref = next((c.get('idRef') for c in e if strip(c.tag) == 'ServiceReference'), None)
                rows = []
                for c in e:
                    if strip(c.tag) != 'ContentReference': continue
                    w = next((k for k in c if strip(k.tag) == 'PresentationWindow'), None)
                    if w is None: continue
                    rows.append((int(w.get('startTime')) - NTP, int(w.get('endTime')) - NTP, c.get('idRef')))
                g['schedule'][sref] = sorted(rows)
        g['updated'] = time.time()

    def feed(self, frame, plp, block, payload):
        a = self.alp.get(plp)
        if a is None: return
        for ev in a.feed(frame, block, payload):
            if ev[0] == 'udp': self.on_udp(ev[1], ev[2], ev[3])
            elif ev[0] == 'lmt': self.on_lmt(ev[1])

    def housekeeping(self):
        now = time.time()
        for f in self.flows.values():
            for k in [k for k, o in f.open.items() if now - o.born > 15]: del f.open[k]
            f.rate.append((now, f.bytes))
            quiet = now - f.last > 30
            if quiet != f.quiet and now - self.started > 60:
                f.quiet = quiet
                OUT.note('flow %s %s' % (f.name.replace('_', ':'), 'silent for 30 s' if quiet else 'carrying data again'))


S = State()


def decoder(args):
    while True:
        cmd = [A3RX] + (['--file', args.file] if args.file else ['--live', '--gain', str(args.gain)])
        with lock: S.decoder = 'starting'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
        rd = p.stdout.read
        try:
            while True:
                h = rd(8)
                if len(h) < 8: break
                n = struct.unpack('<I', h[4:])[0]; body = rd(n)
                if len(body) < n: break
                if h[:4] == b'BBP0':
                    frame, plp, block, ok = struct.unpack('<IBBB', body[:7])
                    if ok:
                        with lock: S.feed(frame, plp, block, body[8:])
                elif h[:4] == b'STAT':
                    st = json.loads(body); st['at'] = time.time()
                    with lock:
                        S.stat = st; S.hist.append((st['at'], st['snr_db'], st['plp1_ok'])); S.frames += 1; S.decoder = 'running'
                        OUT.frame(st)
                        if S.frames % 8 == 0: S.housekeeping()
                    if args.file and args.pace: time.sleep(0.2489)
        finally:
            p.kill()
        with lock:
            S.decoder = 'stopped; restarting'
            OUT.close(why='decoder stopped'); OUT.note('decoder restarted')
        if args.file: break
        time.sleep(2)


class Outages:
    """One record per loss of the main pipe, written when it ends: a burst of dropouts closer together than
    `settle` seconds is a single event with a dip count, so the log stays short however much the signal flaps.
    JSON lines in signal.log; the open event, if any, is in `.cur`."""
    def __init__(self, path, settle=2.0):
        self.path, self.settle = path, settle
        self.cur = None; self.good_run = 0; self.recent = collections.deque(maxlen=200); self.glitches = collections.deque(maxlen=5000)
        try:
            with open(path) as f: self.recent.extend(json.loads(l) for l in f if l.strip())
        except (OSError, ValueError): pass

    def frame(self, st):
        lost = st['plp1_blocks'] - st['plp1_ok']
        # a frame at full strength with a block or two undecoded is a glitch (an impulse on one block), not an outage
        if 0 < lost <= 4 and st['snr_db'] >= 20 and st.get('sync', 1) > 0.2 and self.cur is None:
            self.glitches.append((st['at'], lost)); return
        if lost:
            if self.cur is None:
                self.cur = dict(start=st['at'], dips=1, frames_bad=0, snr_min=st['snr_db'], snrs=[], sync_min=st.get('sync', 1), dropped0=st.get('frames_dropped', 0), blocks_min=st['plp1_ok'])
            elif self.good_run: self.cur['dips'] += 1
            c = self.cur; c['frames_bad'] += 1; c['last'] = st['at']; c['snr_min'] = min(c['snr_min'], st['snr_db']); c['sync_min'] = min(c['sync_min'], st.get('sync', 1))
            c['blocks_min'] = min(c['blocks_min'], st['plp1_ok'])
            if len(c['snrs']) < 4000: c['snrs'].append(st['snr_db'])
            self.good_run = 0
        elif self.cur is not None:
            self.good_run += 1
            if st['at'] - self.cur['last'] >= self.settle: self.close(st)

    def close(self, st=None, why='recovered'):
        c = self.cur; self.cur = None; self.good_run = 0
        if c is None: return
        sn = sorted(c['snrs']) or [c['snr_min']]
        ev = dict(start=round(c['start'], 2), at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['start'])), duration_s=round(c['last'] - c['start'] + 0.25, 1),
                  dips=c['dips'], frames_bad=c['frames_bad'], snr_min=round(c['snr_min'], 1), snr_median=round(sn[len(sn) // 2], 1), sync_min=round(c['sync_min'], 3),
                  blocks_min=c['blocks_min'], frames_dropped=(st or {}).get('frames_dropped', c['dropped0']) - c['dropped0'], end=why,
                  kind='sync lost' if c['sync_min'] < 0.05 else 'low snr' if c['snr_min'] < 20 else 'blocks')
        self.recent.append(ev)
        try:
            with open(self.path, 'a') as f: f.write(json.dumps(ev) + '\n')
        except OSError: pass

    def note(self, what):
        ev = dict(start=round(time.time(), 2), at=time.strftime('%Y-%m-%d %H:%M:%S'), event=what)
        self.recent.append(ev)
        try:
            with open(self.path, 'a') as f: f.write(json.dumps(ev) + '\n')
        except OSError: pass


OUT = Outages(os.path.join(STATE, 'signal.log'))


# ------------------------------------------------------------------ views
def service(sid):
    return next((s for s in S.services if s['id'] == sid), None)


def chan(s):
    return '%s.%s' % (s['major'], s['minor']) if s.get('major') else None


def tracks_of(f):
    """[(tsi, kind, rep id, rep info)] for the tracks of a flow that have an initialisation segment"""
    out = []
    for tsi in sorted(f.tracks):
        tr = f.tracks[tsi]
        if not tr['init']: continue
        ls = f.ls.get(tsi) or {}
        rep = ls.get('rep'); info = f.reps.get(rep) or {}
        kind = ls.get('kind') or info.get('kind') or ('video' if tsi < 200 else 'audio' if tsi < 300 else 'subtitles')
        out.append((tsi, kind, rep, info))
    return out


def segdur(f, tsi, info):
    if info.get('duration'): return info['duration']
    return 2.002


def internet_master(f, video_only):
    """Internet-delivered channel: a playlist that sends the player to the broadcaster's own servers."""
    vid = [(k, v) for k, v in f.reps.items() if v['kind'] == 'video' and v['timeline'] and v['base']]
    aud = [(k, v) for k, v in f.reps.items() if v['kind'] == 'audio' and v['timeline'] and v['base']]
    if not vid: return None
    L = ['#EXTM3U', '#EXT-X-VERSION:7', '#EXT-X-INDEPENDENT-SEGMENTS']
    if aud and not video_only:
        for i, (k, v) in enumerate(aud):
            L.append('#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="%s",LANGUAGE="%s",DEFAULT=%s,AUTOSELECT=YES,URI="r_%s.m3u8"' % (v.get('lang') or k, v.get('lang') or 'und', 'YES' if i == 0 else 'NO', k))
    for k, v in sorted(vid, key=lambda kv: -kv[1]['bandwidth']):
        codecs = [v['codecs']] + ([aud[0][1]['codecs']] if aud and not video_only else [])
        a = 'BANDWIDTH=%d,CODECS="%s"' % (v['bandwidth'] or 4000000, ','.join(codecs))
        if v.get('width'): a += ',RESOLUTION=%sx%s' % (v['width'], v['height'])
        if aud and not video_only: a += ',AUDIO="aud"'
        L += ['#EXT-X-STREAM-INF:' + a, 'r_%s.m3u8' % k]
    return '\n'.join(L) + '\n'


def internet_media(f, rid):
    v = f.reps.get(rid)
    if not v or not v['timeline'] or not v['base']: return None
    d = v['timeline'][0][1]
    L = ['#EXTM3U', '#EXT-X-VERSION:7', '#EXT-X-TARGETDURATION:%d' % (int(d / v['timescale']) + 1), '#EXT-X-MEDIA-SEQUENCE:%d' % (v['timeline'][0][0] // d),
         '#EXT-X-MAP:URI="go/%s/init.mp4"' % rid]
    # players are picky about segment file extensions, so list them here and redirect each request to the origin
    for i, (t, dd) in enumerate(v['timeline']):
        L += ['#EXTINF:%.3f,' % (dd / v['timescale']), 'go/%s/%d.m4s' % (rid, t)]
    return '\n'.join(L) + '\n'


def master(sid, video_only=False):
    s = service(sid); f = S.flows.get(s['flow']) if s else None
    if not f: return None
    if f.internet: return internet_master(f, video_only)
    tr = tracks_of(f)
    vid = [t for t in tr if t[1] == 'video']; aud = [t for t in tr if t[1] == 'audio']
    if not vid: return None
    L = ['#EXTM3U', '#EXT-X-VERSION:7', '#EXT-X-INDEPENDENT-SEGMENTS']
    if aud and not video_only:
        for i, (tsi, kind, rep, info) in enumerate(aud):
            L.append('#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="%s",LANGUAGE="%s",DEFAULT=%s,AUTOSELECT=YES,URI="t%d.m3u8"' % (
                info.get('lang') or rep or tsi, info.get('lang') or 'und', 'YES' if i == 0 else 'NO', tsi))
    for tsi, kind, rep, info in vid:
        codecs = [info.get('codecs') or 'hev1.2.4.L123.B0']
        if aud and not video_only: codecs.append(aud[0][3].get('codecs') or 'ac-4.02.01.01')
        a = 'BANDWIDTH=%d,CODECS="%s"' % (info.get('bandwidth') or 8000000, ','.join(codecs))
        if info.get('width'): a += ',RESOLUTION=%sx%s' % (info['width'], info['height'])
        if aud and not video_only: a += ',AUDIO="aud"'
        L += ['#EXT-X-STREAM-INF:' + a, 't%d.m3u8' % tsi]
    return '\n'.join(L) + '\n'


def media(sid, tsi):
    s = service(sid); f = S.flows.get(s['flow']) if s else None
    if not f or tsi not in f.tracks or not f.tracks[tsi]['init']: return None
    tr = f.tracks[tsi]; info = next((t[3] for t in tracks_of(f) if t[0] == tsi), {})
    tois = [t for t, v in tr['segs'].items() if v[0]]
    tois = tois[-max(12, int(24 / segdur(f, tsi, info))):]
    d = segdur(f, tsi, info)
    L = ['#EXTM3U', '#EXT-X-VERSION:7', '#EXT-X-TARGETDURATION:%d' % (int(d) + 1), '#EXT-X-MEDIA-SEQUENCE:%d' % (tois[0] if tois else 0),
         '#EXT-X-MAP:URI="%d/init.mp4"' % tsi]
    prev = None
    for t in tois:
        if prev is not None and t != prev + 1: L.append('#EXT-X-DISCONTINUITY')
        L += ['#EXTINF:%.3f,' % d, '%d/%d.m4s' % (tsi, t)]; prev = t
    return '\n'.join(L) + '\n'


# MPEG CICP ChannelConfiguration (ISO/IEC 23001-8), as the MPD's AudioChannelConfiguration gives it
CICP = {1: (1, 'mono'), 2: (2, 'stereo'), 3: (3, '3.0'), 6: (6, '5.1'), 7: (8, '7.1'), 12: (8, '7.1'), 14: (8, '7.1'), 16: (10, '5.1.4'), 17: (12, '5.1.6'), 19: (12, '7.1.4')}


def ac4_layout(info):
    """channels, name of an audio representation: the MPD's channel configuration, else a guess from the AC-4 codecs
    string (whose last field is the presentation's md_compat level, which on this station happens to track the layout)"""
    if info.get('cicp') in CICP: return CICP[info['cicp']]
    m = re.fullmatch(r'ac-4\.(\w+)\.(\w+)\.(\w+)', info.get('codecs') or '')
    return {0: (2, 'stereo'), 1: (6, '5.1'), 2: (10, '5.1.4')}.get(int(m.group(3), 16) if m else -1, (2, 'stereo'))


def audio_choice(sid, want='auto'):
    """(video playlist, audio playlist or None, description, channels) for the MPEG-TS stream of a service"""
    s = service(sid); f = S.flows.get(s['flow']) if s else None
    if not f: return None
    if f.internet:
        v = max(((k, r) for k, r in f.reps.items() if r['kind'] == 'video' and r['timeline']), key=lambda kr: kr[1]['bandwidth'], default=None)
        if not v: return None
        auds = [('r_%s.m3u8' % k, r) for k, r in f.reps.items() if r['kind'] == 'audio' and r['timeline']]
        vp = 'r_%s.m3u8' % v[0]
    else:
        tr = tracks_of(f)
        v = next((t for t in tr if t[1] == 'video'), None)
        if not v: return None
        auds = [('t%d.m3u8' % t[0], t[3]) for t in tr if t[1] == 'audio']
        vp = 't%d.m3u8' % v[0]
    if want == 'none' or not auds or not os.path.exists(AC3CLI): return vp, None, 'no audio (ac3forge not built)', 0
    pick = int(want) if want.isdigit() and int(want) < len(auds) else 0
    a = auds[pick]
    ch, name = ac4_layout(a[1])
    note = 'track %d (%s, %s %s) as Opus %s' % (pick, a[1].get('lang') or 'und', a[1].get('codecs'), name, 'stereo' if ch <= 2 else '5.1' if ch <= 10 else '5.1, or 7.1 with ?ch=8')
    return vp, a[0], note, ch


def moof_time(seg):
    """decode time of a media segment (tfdt), in the track's timescale"""
    i = seg.find(b'tfdt')
    if i < 0: return None
    return struct.unpack('>Q', seg[i + 8:i + 16])[0] if seg[i + 4] == 1 else struct.unpack('>I', seg[i + 8:i + 12])[0]


def mdhd_timescale(init):
    i = init.find(b'mdhd')
    return (struct.unpack('>I', init[i + (16 if init[i + 4] == 0 else 24):][:4])[0] if i >= 0 else 0) or 240000


class Track:
    """a media track as a sequence of segments keyed by TOI (over the air) or by start time (an internet-delivered
    representation, fetched from the broadcaster's origin): init(), get(key) waiting for the segment, after(key)"""
    def __init__(self, f, tsi=None, rid=None):
        self.f, self.tsi, self.rid = f, tsi, rid
        self.cache = collections.OrderedDict()

    def init(self):
        with lock:
            if self.tsi is not None: return self.f.tracks[self.tsi]['init'][1]
            r = self.f.reps[self.rid]; url = r['base'] + r['init']
        return self.fetch(url)

    def fetch(self, url):
        import urllib.request
        return urllib.request.urlopen(url, timeout=10).read()

    def keys(self):
        """the segments there are, in order, with the decode time of each in seconds"""
        with lock:
            if self.tsi is not None:
                tr = self.f.tracks[self.tsi]; ts = mdhd_timescale(tr['init'][1])
                return [(t, (moof_time(v[0]) or 0) / ts) for t, v in tr['segs'].items() if v[0]]
            r = self.f.reps[self.rid]
            return [(t, t / r['timescale']) for t, d in r['timeline']]

    def edge(self):
        """internet only: the newest segment the origin actually has, found by asking past the end of the (stale) manifest"""
        import urllib.request
        with lock:
            r = self.f.reps[self.rid]; line = r['timeline']
            if not line: return None
            key, d = line[-1]; tmpl = r['base'] + r['media']
        for i in range(40):
            nxt = key + d; url = tmpl.replace('$Time$', str(nxt)).replace('$Number$', str(nxt // d))
            try: urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=5)
            except OSError: break
            key = nxt
        return key

    def after(self, key):
        with lock:
            if self.tsi is not None: return key + 1
            r = self.f.reps[self.rid]
            return key + next((d for t, d in r['timeline'] if t == key), r['timeline'][-1][1] if r['timeline'] else 0)

    def samples(self, sr=48000):
        """samples per segment at `sr`"""
        with lock:
            if self.tsi is not None:
                tr = self.f.tracks[self.tsi]; ts = mdhd_timescale(tr['init'][1])
                tt = [(t, moof_time(v[0])) for t, v in tr['segs'].items() if v[0]]
                return next((round((b - a) * sr / ts) for (t1, a), (t2, b) in zip(tt, tt[1:]) if t2 == t1 + 1 and a and b), 48048)
            r = self.f.reps[self.rid]
            return round(r['timeline'][0][1] * sr / r['timescale']) if r['timeline'] else 48048

    def get(self, key, patience=3.0):
        """the bytes of a segment once it has arrived, or None when it has not within `patience` seconds of its successor"""
        if key in self.cache: return self.cache[key]
        t0 = time.time()
        while True:
            with lock:
                if self.tsi is not None:
                    tr = self.f.tracks.get(self.tsi); seg = tr['segs'].get(key) if tr else None
                    data = seg[0] if seg and seg[0] else None
                    later = any(t > key and v[0] for t, v in tr['segs'].items()) if tr else False
                    gone = bool(tr and tr['segs'] and next(iter(tr['segs'])) > key); url = None
                else:
                    r = self.f.reps.get(self.rid) or {}; line = r.get('timeline') or []
                    # the manifest over the air lags the origin by a while, so the origin itself is asked for the next segment
                    data = None; later = any(t > key for t, d in line); gone = bool(line) and line[0][0] > key
                    url = r['base'] + r['media'].replace('$Time$', str(key)).replace('$Number$', str(key // line[0][1])) if line and not gone else None
            if url:
                try:
                    tf = time.time(); data = self.fetch(url)
                    if DEBUG: print('fetch %s %s %d bytes %.2fs (waited %.2fs)' % (self.rid, key, len(data), time.time() - tf, tf - t0), flush=True)
                except OSError as e:
                    if DEBUG: print('fetch %s %s failed: %s later=%s' % (self.rid, key, e, later), flush=True)
                    # the origin keeps a shorter window than the manifest lists: an older segment that is missing is gone for good
                    if later or time.time() - t0 > 20: return None
                    time.sleep(0.5); continue
            if data:
                self.cache[key] = data
                while len(self.cache) > 4: self.cache.popitem(last=False)
                return data
            if gone or (later and time.time() - t0 > patience) or time.time() - t0 > 30: return None
            time.sleep(0.05)


def feed_video(src, key, w):
    """push a video track into a pipe as one fragmented MP4 stream, segment by segment as they arrive"""
    try:
        w.write(src.init()); w.flush()
        while True:
            seg = src.get(key)
            if seg: w.write(seg); w.flush()
            key = src.after(key)
    except BrokenPipeError: pass
    except (OSError, KeyError, TypeError, IndexError) as e: print('video feed ended:', repr(e), flush=True)
    finally:
        try: w.close()
        except OSError: pass


def ac4_decode(init, segs, fold):
    """ac3forge decodes a whole file at a time, so the audio is decoded a window of segments at a time:
    float32 PCM, frames x `fold` channels (2, 6 or 8), whatever layout the presentation renders as"""
    import numpy as np
    cmd = [AC3CLI, 'decode', '-', '-', 'conceal=repeat'] + (['channels=2', 'downmix=auto'] if fold == 2 else [])
    p = subprocess.run(cmd, input=init + b''.join(segs), capture_output=True, timeout=10)
    out = p.stdout
    if not out.startswith(b'RIFF'): return None
    i = 12; ch = 2; bits = 32; raw = b''
    while i + 8 <= len(out):
        tag = out[i:i + 4]; n = struct.unpack('<I', out[i + 4:i + 8])[0]
        if tag == b'fmt ': ch = struct.unpack('<H', out[i + 10:i + 12])[0]; bits = struct.unpack('<H', out[i + 22:i + 24])[0]
        if tag == b'data': raw = out[i + 8:i + 8 + n]; break
        i += 8 + n + (n & 1)
    x = np.frombuffer(raw, dtype=np.float32 if bits == 32 else np.float64).reshape(-1, ch).astype(np.float32)
    if ch > fold:
        # ac3forge writes L R C LFE Ls Rs [Lb Rb] Tfl Tfr Tbl Tbr; fold to FFmpeg's 5.1 (FL FR FC LFE BL BR) or 7.1 (.. BL BR SL SR),
        # heights -3 dB into the corners, and the backs into the surrounds when only 5.1 is wanted
        y = np.zeros((len(x), fold), np.float32)
        L, R, C, LFE, Ls, Rs = (x[:, i] for i in range(6))
        Lb, Rb = (x[:, 6], x[:, 7]) if ch == 12 else (None, None)
        Tfl, Tfr, Tbl, Tbr = (x[:, i] for i in range(ch - 4, ch)) if ch >= 10 else (0, 0, 0, 0)
        y[:, 0] = L + 0.707 * Tfl; y[:, 1] = R + 0.707 * Tfr; y[:, 2] = C; y[:, 3] = LFE
        if fold == 8:
            y[:, 4] = (Lb if Lb is not None else Ls) + 0.707 * Tbl; y[:, 5] = (Rb if Rb is not None else Rs) + 0.707 * Tbr; y[:, 6] = Ls; y[:, 7] = Rs
        else:
            y[:, 4] = Ls + 0.707 * Tbl + (0.707 * Lb if Lb is not None else 0); y[:, 5] = Rs + 0.707 * Tbr + (0.707 * Rb if Rb is not None else 0)
        x = y
    elif ch < fold:
        x = np.concatenate([x, np.zeros((len(x), fold - ch), np.float32)], axis=1)
    return x


def imsc_cues(body):
    """(begin s, end s, ASS text, region) for the <p> elements of an IMSC1 caption segment; times on the media clock.
    Styling kept: colour, italic/bold, line breaks; padding-only spans (transparent background) dropped."""
    i = body.find(b'<tt'); j = body.rfind(b'</tt>')
    if i < 0 or j < 0: return []
    try: root = ET.fromstring(body[i:j + 5])
    except ET.ParseError: return []
    T = '{http://www.w3.org/ns/ttml#styling}'
    styles = {e.get('{http://www.w3.org/XML/1998/namespace}id'): e.attrib for e in root.iter() if strip(e.tag) == 'style'}
    regions = {e.get('{http://www.w3.org/XML/1998/namespace}id'): e.attrib for e in root.iter() if strip(e.tag) == 'region'}
    def secs(v):
        h, m, sec = v.split(':'); return int(h) * 3600 + int(m) * 60 + float(sec)
    def pct(v):
        try: return [float(x.rstrip('%')) for x in v.split()]
        except (ValueError, AttributeError): return None
    def esc(t): return t.replace('\\', '\\\\').replace('{', '(').replace('}', ')')
    def render(e, inherited):
        st = dict(inherited); st.update(styles.get(e.get('style'), {}))
        for k, v in e.attrib.items():
            if k.startswith(T): st[k] = v
        bg = st.get(T + 'backgroundColor', '')
        if len(bg) == 9 and bg.endswith('00') and not list(e): return ''          # transparent padding
        tags = ''
        c = st.get(T + 'color')
        if c and re.fullmatch(r'#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?', c): tags += '{\\c&H%s%s%s&}' % (c[5:7], c[3:5], c[1:3])
        if st.get(T + 'fontStyle') == 'italic': tags += '{\\i1}'
        if st.get(T + 'fontWeight') == 'bold': tags += '{\\b1}'
        out = tags + esc(e.text or '')
        for k in e:
            out += '\\N' if strip(k.tag) == 'br' else render(k, st)
            out += esc(k.tail or '')
        if st.get(T + 'fontStyle') == 'italic': out += '{\\i0}'
        if st.get(T + 'fontWeight') == 'bold': out += '{\\b0}'
        return out
    out = []
    for p in root.iter():
        if strip(p.tag) != 'p' or not p.get('begin'): continue
        try: a, b = secs(p.get('begin')), secs(p.get('end') or p.get('begin'))
        except ValueError: continue
        t = render(p, {})
        t = re.sub(r'[ \t]+', ' ', t)
        t = '\\N'.join(l.strip() for l in t.split('\\N'))
        t = re.sub(r'^(\\N)+|(\\N)+$', '', t)
        if not re.sub(r'\{[^}]*\}|\\N', '', t).strip(): continue
        r = regions.get(p.get('region'), {})
        out.append((a, b, t, dict(origin=pct(r.get(T + 'origin')), extent=pct(r.get(T + 'extent')))))
    return out


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Courier New,44,&H00AAAAAA,&H00AAAAAA,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,3,0,2,60,60,50,1

[Events]
Format: ReadOrder, Layer, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_event(n, text, region):
    """a Matroska ASS block: the caption placed at its region's top-left corner (the 608 row/column grid, as IMSC1 carries it)"""
    o = region.get('origin')
    if o and len(o) == 2: text = '{\\an7\\pos(%d,%d)}' % (o[0] * 19.2, o[1] * 10.8) + text
    return '%d,0,Caption,,0,0,0,,%s' % (n, text)


def ebml(id_, payload):
    """an EBML element: id bytes, size as a vint, payload"""
    n = len(payload)
    for k in range(1, 9):
        if n < (1 << (7 * k)) - 1:
            size = (n | (1 << (7 * k))).to_bytes(k, 'big'); break
    return id_ + size + payload


def ebml_uint(v):
    return v.to_bytes(max(1, (v.bit_length() + 7) // 8), 'big')


def mkv_subtitle_header():
    head = ebml(b'\x1a\x45\xdf\xa3', ebml(b'\x42\x86', b'\x01') + ebml(b'\x42\xf7', b'\x01') + ebml(b'\x42\xf2', b'\x04') + ebml(b'\x42\xf3', b'\x08')
                + ebml(b'\x42\x82', b'matroska') + ebml(b'\x42\x87', b'\x04') + ebml(b'\x42\x85', b'\x02'))
    info = ebml(b'\x15\x49\xa9\x66', ebml(b'\x2a\xd7\xb1', ebml_uint(1000000)) + ebml(b'\x4d\x80', b'a3 gateway') + ebml(b'\x57\x41', b'a3 gateway'))
    track = ebml(b'\xae', ebml(b'\xd7', b'\x01') + ebml(b'\x73\xc5', b'\x01') + ebml(b'\x83', b'\x11') + ebml(b'\x86', b'S_TEXT/ASS') + ebml(b'\x22\xb5\x9c', b'eng')
                 + ebml(b'\x9c', b'\x00') + ebml(b'\x63\xa2', ASS_HEADER.encode('utf-8')))
    # the Segment has no known length: it is being written live
    return head + b'\x18\x53\x80\x67' + b'\x01\xff\xff\xff\xff\xff\xff\xff' + info + ebml(b'\x16\x54\xae\x6b', track)


def mkv_cue(ms, dur_ms, text):
    """one Cluster holding one subtitle block"""
    block = ebml(b'\xa1', b'\x81' + b'\x00\x00' + b'\x00' + text.encode('utf-8'))
    return ebml(b'\x1f\x43\xb6\x75', ebml(b'\xe7', ebml_uint(ms)) + ebml(b'\xa0', block + ebml(b'\x9b', ebml_uint(max(1, dur_ms)))))


def feed_captions(src, key, w, t0):
    """IMSC1 caption segments -> a live Matroska stream of one ASS track, times relative to t0 (the video's start)"""
    try:
        w.write(mkv_subtitle_header()); w.write(mkv_cue(0, 1, ass_event(0, ' ', {}))); w.flush()      # a cue at 0 pins the track's start to the video's
        last = -1; n = 1; pend = None                      # the captions come as snapshots every ~0.27 s: identical neighbours merge into one event
        ts = mdhd_timescale(src.init())
        while True:
            seg = src.get(key)
            if seg:
                # a blank cue at each segment's time keeps the muxer from holding the video back while nobody is speaking
                st = moof_time(seg)
                if st is not None:
                    ms = int(round((st / ts - t0) * 1000))
                    if ms > last and not (pend and pend[1] > ms):
                        w.write(mkv_cue(ms, 1, ass_event(n, ' ', {}))); n += 1; last = ms
                for a, b, t, region in imsc_cues(seg):
                    ms = int(round((a - t0) * 1000)); end = int(round((b - t0) * 1000))
                    if ms <= last: continue                    # segments overlap by a cue now and then
                    last = ms
                    if pend and pend[2] == t and pend[3].get('origin') == region.get('origin') and ms - pend[1] < 60: pend[1] = max(pend[1], end); continue
                    if pend: w.write(mkv_cue(pend[0], pend[1] - pend[0], ass_event(n, pend[2], pend[3]))); n += 1
                    pend = [ms, end, t, region]
                w.flush()
            key = src.after(key)
    except BrokenPipeError: pass
    except (OSError, KeyError, TypeError, IndexError) as e: print('caption feed ended:', repr(e), flush=True)
    finally:
        try: w.close()
        except OSError: pass


def feed_audio(src, key, w, fold, nch):
    """decode an AC-4 track segment by segment into a pipe of raw float32 PCM at 48 kHz.
    Each segment is decoded together with the two before it (there is an I-frame at least every two segments) and the
    last second kept, with a 10 ms crossfade over the join to hide the decoder's noise-fill randomness."""
    import numpy as np
    X = 480
    try:
        init = src.init(); n_seg = src.samples()
        window = collections.OrderedDict(); held = None
        # start with the two segments before, so the first one decodes from an I-frame
        prior = [k for k, t in src.keys() if k < key][-2:]
        for k in prior:
            seg = src.get(k, 0)
            if seg: window[k] = seg
            else: window.clear()
        while True:
            seg = src.get(key)
            if seg is None:
                w.write(np.zeros((n_seg, nch), np.float32).tobytes()); w.flush()
                window.clear(); key = src.after(key); continue
            window[key] = seg
            while len(window) > 3: window.popitem(last=False)
            pcm = ac4_decode(init, list(window.values()), nch)
            if pcm is None: key = src.after(key); continue
            take = pcm[-(n_seg + X):] if len(window) > 1 else pcm[-n_seg:]
            if len(take) < n_seg: take = np.concatenate([np.zeros((n_seg - len(take), take.shape[1]), np.float32), take])
            if held is not None and len(take) >= n_seg + X:
                r = np.linspace(0, 1, X, dtype=np.float32)[:, None]
                body = np.concatenate([held * (1 - r) + take[:X] * r, take[X:-X]])
            else:
                body = take[-n_seg:][:-X] if held is None else np.concatenate([held, take[-n_seg:][:-X]])
            held = take[-X:]
            w.write(body.astype(np.float32).tobytes()); w.flush()
            key = src.after(key)
    except BrokenPipeError: pass
    except (OSError, KeyError, TypeError, IndexError, subprocess.TimeoutExpired) as e: print('audio feed ended:', repr(e), flush=True)
    finally:
        try: w.close()
        except OSError: pass


def lineup(host):
    out = []
    for s in S.services:
        f = S.flows.get(s['flow']) if s['flow'] else None
        e = dict(GuideNumber=chan(s) or s['id'], GuideName=s['name'], ServiceId=s['id'], Category=s['category'])
        held = None
        if f and 'held.held' in f.sls:
            m = re.search(rb'bbandEntryPageUrl="([^"]+)"', f.sls['held.held'])
            held = m.group(1).decode().replace('&amp;', '&') if m else None
        has_video = bool(f and any(t[1] == 'video' for t in tracks_of(f)))
        if s['category'] == 1 and has_video:
            tr = tracks_of(f); v = next(t for t in tr if t[1] == 'video'); a = next((t for t in tr if t[1] == 'audio'), None)
            e.update(Delivery='broadcast', URL='http://%s/auto/v%s' % (host, chan(s)), MKV='http://%s/auto/v%s.mkv' % (host, chan(s)), HLS='http://%s/live/%s/master.m3u8' % (host, s['id']),
                     Audio=(audio_choice(s['id']) or (0, 0, None, 0))[2],
                     VideoCodec=(v[3].get('codecs') or 'hevc'), AudioCodec=(a[3].get('codecs') if a else None),
                     Resolution=('%sx%s' % (v[3]['width'], v[3]['height']) if v[3].get('width') else None), HD=1)
        elif s['category'] == 1 and f and f.internet:
            v = max((r for r in f.reps.values() if r['kind'] == 'video'), key=lambda r: r['bandwidth'], default={})
            a = next((r for r in f.reps.values() if r['kind'] == 'audio'), {})
            e.update(Delivery='internet', URL='http://%s/live/%s/master.m3u8' % (host, s['id']), HLS='http://%s/live/%s/master.m3u8' % (host, s['id']),
                     Manifest='http://%s/live/%s/manifest.mpd' % (host, s['id']), Origin=v.get('base'), Audio=(audio_choice(s['id']) or (0, 0, None, 0))[2],
                     TS='http://%s/auto/v%s' % (host, chan(s)), VideoCodec=v.get('codecs'), AudioCodec=a.get('codecs'),
                     Resolution=('%sx%s' % (v['width'], v['height']) if v.get('width') else None))
        elif held:
            e.update(Delivery='app', URL='http://%s/app/%s' % (host, s['id']), EntryPage=held)
        elif s['category'] == 4:
            e.update(Delivery='guide', URL='http://%s/guide' % host)
        else:
            e.update(Delivery='pending', URL=None)
        if held and 'EntryPage' not in e: e['App'] = 'http://%s/app/%s' % (host, s['id'])
        out.append(e)
    return out


def guide_json():
    g = S.guide; now = time.time(); out = []
    svc_by_gsid = {s['gsid']: s for s in S.services}
    for sid, sv in g['services'].items():
        rows = []
        for a, b, cid in g['schedule'].get(sid, []):
            if b < now - 3600 or a > now + 12 * 3600: continue
            c = g['content'].get(cid, {})
            rows.append(dict(start=a, end=b, title=c.get('title'), description=c.get('description'), icon=c.get('icon'), rating=c.get('rating')))
        s = svc_by_gsid.get(sv.get('gsid'))
        out.append(dict(name=sv['name'], channel='%s.%s' % (sv['major'], sv['minor']), icon=sv.get('icon'), service_id=s['id'] if s else None, programmes=rows, esg_ids=[sid]))
    # the ESG carries each channel as more than one Service fragment (different fragment ids, same channel): one row each, schedules merged
    by_chan = {}
    for r in out:
        k = r['channel'], r['name']
        if k in by_chan:
            m = by_chan[k]; m['esg_ids'] += r['esg_ids']
            seen = {(p['start'], p['end']) for p in m['programmes']}
            m['programmes'] = sorted(m['programmes'] + [p for p in r['programmes'] if (p['start'], p['end']) not in seen], key=lambda p: p['start'])
            m['service_id'] = m['service_id'] or r['service_id']; m['icon'] = m['icon'] or r['icon']
        else: by_chan[k] = r
    out = list(by_chan.values())
    out.sort(key=lambda r: [int(x) for x in r['channel'].split('.') if x.isdigit()])
    return dict(now=now, updated=g['updated'], services=out)


STYLE = "<style>body{font:15px/1.45 system-ui;margin:1.5em;background:#15171a;color:#e8e8e8}a{color:#7cc4ff}td,th{padding:4px 12px;text-align:left;border-bottom:1px solid #333}code{background:#262a2f;padding:1px 5px;border-radius:3px}h1{margin:.2em 0}.ok{color:#5d5}.warn{color:#fc3}.bad{color:#f66}small{opacity:.65}</style>"

INDEX = """<!doctype html><meta charset=utf-8><title>RF 23 ATSC 3.0 gateway</title>""" + STYLE + """
<h1>RF 23 ATSC 3.0 gateway</h1><div id=s>loading</div>
<p><a href=/guide>Programme guide</a> &middot; <a href=/meter>Signal meter</a> &middot; <a href=/events.json>signal outages</a> &middot; <a href=/lineup.json>lineup.json</a> &middot; <a href=/lineup.m3u>lineup.m3u</a> &middot; <a href=/status.json>status.json</a> &middot; <a href=/tables/>signalling tables</a></p>
<table id=t></table>
<p><small>Play a channel with <code>mpv URL</code>. The TS streams carry Opus decoded from the broadcast's AC-4 by ac3forge: stereo, 5.1, or 7.1 with the heights of a 7.1.4 Atmos presentation folded in; add <code>?ch=2</code> for the stream's own stereo downmix, <code>?audio=1</code> for the second track, <code>?lead=N</code> seconds of cushion. MPEG-TS cannot carry the broadcast's IMSC1 text captions, so the <b>MKV</b> flavour of the same stream (<code>/auto/v6.1.mkv</code>) adds them as a positioned ASS track (row placement, colour and italics kept); select it in mpv with <code>j</code>. The HLS streams carry the AC-4 untouched, which most players cannot decode.</small></p>
<script>
async function tick(){try{const s=await (await fetch('/status.json')).json(), l=await (await fetch('/lineup.json')).json();
 const g=s.signal||{}, c=g.snr_db>=__MAIN__?'ok':g.snr_db>=__ROB__?'warn':'bad';
 document.getElementById('s').innerHTML='<p>decoder: <b>'+s.decoder+'</b> &middot; SNR <b class='+c+'>'+(g.snr_db??'-')+' dB</b> &middot; main pipe '+(g.plp1_ok??'-')+'/'+(g.plp1_blocks??'-')+' blocks &middot; '+(g.ms??'-')+' ms per frame &middot; behind by '+(g.backlog_ms??'-')+' ms</p>';
 document.getElementById('t').innerHTML='<tr><th>channel</th><th>name</th><th>delivery</th><th>video</th><th>audio in the TS stream</th><th>stream</th></tr>'+l.map(e=>'<tr><td>'+e.GuideNumber+'</td><td>'+e.GuideName+'</td><td>'+e.Delivery+'</td><td>'+(e.VideoCodec?(e.Resolution||'')+' '+e.VideoCodec:'')+'</td><td>'+(e.Audio||'')+'</td><td>'+(e.URL?'<a href="'+e.URL+'">'+(e.Delivery=='broadcast'?'TS':e.Delivery=='internet'?'HLS':'open')+'</a>':'')+(e.TS?' &middot; <a href="'+e.TS+'">TS</a>':'')+(e.MKV?' &middot; <a href="'+e.MKV+'">MKV + captions</a>':'')+(e.HLS&&e.Delivery=='broadcast'?' &middot; <a href="'+e.HLS+'">HLS</a>':'')+(e.App?' &middot; <a href="'+e.App+'">app</a>':'')+'</td></tr>').join('')}catch(e){}}
tick();setInterval(tick,2000)</script>"""

METER = """<!doctype html><meta charset=utf-8><title>RF 23 live SNR</title>
<style>body{font:16px system-ui;margin:0;background:#111;color:#eee;text-align:center}
#n{font:700 22vw/1 system-ui;margin:.1em 0 0}#u{font-size:4vw;opacity:.7}#v{font-size:3.2vw;margin:.4em}
#bar{height:5vh;margin:1vh 4vw;background:#333;position:relative;border-radius:6px;overflow:hidden}#fill{height:100%;width:0;background:#4c4}
.m{position:absolute;top:0;bottom:0;width:2px;background:#fff8}canvas{width:92vw;height:26vh;margin-top:1vh}small{opacity:.6}</style>
<div id=n>--</div><div id=u>dB</div><div id=v></div>
<div id=bar><div id=fill></div><div class=m id=m1></div><div class=m id=m2></div></div>
<canvas id=c width=1200 height=300></canvas><div><small id=s></small></div>
<script>
const MAX=35,MAIN=__MAIN__,ROB=__ROB__;m1.style.left=(ROB/MAX*100)+'%';m2.style.left=(MAIN/MAX*100)+'%';
async function tick(){try{const d=await (await fetch('/snr.json?n=300',{cache:'no-store'})).json();
 if(d.snr_db==null){n.textContent='--';v.textContent=d.decoder||'waiting for the radio';return}
 const age=Date.now()/1000-d.at, s=d.snr_db, col=s>=MAIN?'#4c4':s>=ROB?'#fc3':'#f55';
 n.textContent=s.toFixed(1);n.style.color=age>2?'#777':col;fill.style.width=Math.max(0,Math.min(100,s/MAX*100))+'%';fill.style.background=col;
 v.textContent=age>2?'stale ('+age.toFixed(0)+' s old)':'main pipe '+d.plp1_ok+' of '+d.plp1_blocks+' blocks';
 s_.textContent=d.ms+' ms per frame, behind by '+d.backlog_ms+' ms, '+d.frames_dropped+' frames skipped';
 const g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);g.strokeStyle='#555';g.setLineDash([6,6]);
 for(const y of [ROB,MAIN]){g.beginPath();g.moveTo(0,H-y/MAX*H);g.lineTo(W,H-y/MAX*H);g.stroke()}
 g.setLineDash([]);g.strokeStyle='#6cf';g.lineWidth=2;g.beginPath();const h=d.history,t1=d.at,span=60;
 h.forEach((p,i)=>{const x=W-(t1-p[0])/span*W,y=H-Math.max(0,Math.min(MAX,p[1]))/MAX*H;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}catch(e){v.textContent='gateway not responding'}}
const s_=document.getElementById('s');setInterval(tick,250);tick()</script>"""

GUIDE = """<!doctype html><meta charset=utf-8><title>RF 23 programme guide</title>""" + STYLE + """
<style>#g{position:relative;overflow-x:auto;border:1px solid #333;margin-top:1em}.row{position:relative;height:74px;border-bottom:1px solid #2a2d31}
.ch{position:sticky;left:0;z-index:3;width:150px;height:74px;background:#1d2024;border-right:1px solid #333;display:flex;align-items:center;gap:8px;padding:0 8px;box-sizing:border-box}
.ch img{height:40px;max-width:60px}.p{position:absolute;top:4px;height:64px;background:#26303b;border:1px solid #3b4856;border-radius:5px;box-sizing:border-box;padding:4px 7px;overflow:hidden;font-size:13px;cursor:default}
.p.now{background:#1f4a35;border-color:#3c8f66}.p b{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.p span{opacity:.7;font-size:12px}
#axis{position:relative;height:22px;font-size:12px;opacity:.8}#axis div{position:absolute;top:2px;border-left:1px solid #444;padding-left:4px;height:20px}
#nowl{position:absolute;top:0;bottom:0;width:2px;background:#f55;z-index:2}#d{min-height:4.5em;margin-top:1em;max-width:70em}#d img{float:left;height:110px;margin-right:12px;border-radius:4px}</style>
<h1>Programme guide <small id=u></small></h1><p><a href=/>channels</a> &middot; <a href=/guide.json>guide.json</a></p>
<div id=g></div><div id=d><small>Hover over a programme for its description.</small></div>
<script>
const PX=5, LEFT=150;   // pixels per minute
function hm(t){return new Date(t*1000).toLocaleTimeString([], {hour:'numeric',minute:'2-digit'})}
async function draw(){const d=await (await fetch('/guide.json',{cache:'no-store'})).json();const g=document.getElementById('g');
 if(!d.services.length){g.innerHTML='<p style="padding:1em">The guide has not arrived yet. It is sent as a carousel and can take a minute after tuning.</p>';return}
 const t0=Math.floor((d.now-1800)/1800)*1800, t1=t0+8*3600, x=t=>LEFT+(t-t0)/60*PX;
 let h='<div id=axis style="width:'+x(t1)+'px">';for(let t=t0;t<t1;t+=1800)h+='<div style="left:'+x(t)+'px">'+hm(t)+'</div>';h+='</div>';
 window.P=[];
 for(const s of d.services){h+='<div class=row style="width:'+x(t1)+'px"><div class=ch>'+(s.icon?'<img src="/esg/icon/'+s.icon+'">':'')+'<div><b>'+s.channel+'</b><br>'+(s.service_id?'<a href="/live/'+s.service_id+'/master.m3u8">'+s.name+'</a>':s.name)+'</div></div>';
  for(const p of s.programmes){const a=Math.max(p.start,t0),b=Math.min(p.end,t1);if(b<=a)continue;const i=P.push(p)-1;
   h+='<div class="p'+(p.start<=d.now&&d.now<p.end?' now':'')+'" data-i='+i+' style="left:'+x(a)+'px;width:'+Math.max(6,(b-a)/60*PX-2)+'px"><b>'+(p.title||'')+'</b><span>'+hm(p.start)+' to '+hm(p.end)+(p.rating?' &middot; '+p.rating:'')+'</span></div>'}
  h+='</div>'}
 h+='<div id=nowl style="left:'+x(d.now)+'px"></div>';g.innerHTML=h;g.scrollLeft=Math.max(0,x(d.now)-LEFT-200);
 document.getElementById('u').textContent=d.updated?'received '+hm(d.updated):'';
 g.querySelectorAll('.p').forEach(e=>e.onmouseenter=()=>{const p=P[e.dataset.i];document.getElementById('d').innerHTML=(p.icon?'<img src="'+p.icon+'">':'')+'<b>'+p.title+'</b> <small>'+hm(p.start)+' to '+hm(p.end)+(p.rating?' &middot; '+p.rating:'')+'</small><br>'+(p.description||'')})}
draw();setInterval(draw,60000)</script>"""


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a): pass

    def out(self, body, ctype='application/json', code=200, extra=None):
        # only remembered here: the state lock may be held.  do_GET sends it once the lock is released.
        self.pending = (body, ctype, code, extra)

    def send(self, body, ctype, code, extra):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store'); self.send_header('Access-Control-Allow-Origin', '*')
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(b)

    def nf(self, why='not found'):
        self.out(json.dumps({'error': why}), code=404)

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        self.pending = None
        try:
            self.route()
            if self.pending: self.send(*self.pending)
        except (BrokenPipeError, ConnectionResetError): pass

    def route(self):
        p = self.path.split('?')[0]; host = self.headers.get('Host') or '127.0.0.1'
        parts = [x for x in p.split('/') if x]
        if p == '/': return self.out(INDEX.replace('__MAIN__', str(NEED_MAIN)).replace('__ROB__', str(NEED_ROBUST)), 'text/html; charset=utf-8')
        if p == '/meter': return self.out(METER.replace('__MAIN__', str(NEED_MAIN)).replace('__ROB__', str(NEED_ROBUST)), 'text/html; charset=utf-8')
        if p == '/guide': return self.out(GUIDE, 'text/html; charset=utf-8')
        with lock:
            if p == '/snr.json':
                d = dict(S.stat); d['decoder'] = S.decoder; d.setdefault('snr_db', None)
                n = 300
                m = re.search(r'n=(\d+)', self.path)
                if m: n = int(m.group(1))
                d['history'] = [list(x[:2]) for x in list(S.hist)[-n:]]
                return self.out(json.dumps(d))
            if p == '/events.json':
                cur = None
                if OUT.cur: cur = dict(since=time.strftime('%H:%M:%S', time.localtime(OUT.cur['start'])), duration_s=round(time.time() - OUT.cur['start'], 1), dips=OUT.cur['dips'], snr_min=round(OUT.cur['snr_min'], 1))
                now = time.time(); gl = list(OUT.glitches)
                glitches = dict(total=len(gl), blocks=sum(n for t, n in gl), last_hour=sum(1 for t, n in gl if t > now - 3600), last=time.strftime('%H:%M:%S', time.localtime(gl[-1][0])) if gl else None,
                                note='single frames at full strength with up to 4 of 160 blocks undecoded; not outages')
                return self.out(json.dumps(dict(ongoing=cur, glitches=glitches, events=list(OUT.recent)[-200:]), indent=1))
            if p == '/signal.txt':
                st = S.stat
                if not st: return self.out('no frame decoded yet (decoder %s)\n' % S.decoder, 'text/plain')
                v = [x[1] for x in S.hist if x[0] > time.time() - 5] or [st['snr_db']]
                return self.out('%s  SNR %5.1f dB  (last 5 s: mean %.1f, min %.1f, max %.1f)  main pipe %d/%d blocks  %d ms per frame\n' % (
                    time.strftime('%H:%M:%S', time.localtime(st['at'])), st['snr_db'], sum(v) / len(v), min(v), max(v), st['plp1_ok'], st['plp1_blocks'], st['ms']), 'text/plain')
            if p == '/status.json':
                now = time.time(); flows = {}
                for k, f in S.flows.items():
                    r = list(f.rate); kbps = None
                    if len(r) > 4 and r[-1][0] > r[0][0]: kbps = round((r[-1][1] - r[0][1]) * 8 / (r[-1][0] - r[0][0]) / 1e3, 1)
                    flows[k.replace('_', ':')] = dict(kbps=kbps, signalling=sorted(f.sls), tracks={str(t): dict(kind=kd, rep=rp, codecs=i.get('codecs'), segments=len([1 for v in f.tracks[t]['segs'].values() if v[0]])) for t, kd, rp, i in tracks_of(f)},
                                                      open_objects=len(f.open))
                return self.out(json.dumps(dict(decoder=S.decoder, uptime_s=round(now - S.started), frames=S.frames, signal=S.stat, services=len(S.services),
                                                guide=dict(services=len(S.guide['services']), programmes=len(S.guide['content']), updated=S.guide['updated']),
                                                link_mapping=S.lmt, flows=flows), indent=1))
            if p == '/discover.json':
                return self.out(json.dumps(dict(FriendlyName='Pluto ATSC 3.0 gateway', ModelNumber='a3rx', FirmwareName='a3rx', FirmwareVersion='0.1', DeviceID='A3RF0023', TunerCount=1,
                                                BaseURL='http://%s' % host, LineupURL='http://%s/lineup.json' % host)))
            if p == '/lineup_status.json':
                return self.out(json.dumps(dict(ScanInProgress=0, ScanPossible=0, Source='Antenna', SourceList=['Antenna'])))
            if p == '/lineup.json': return self.out(json.dumps(lineup(host), indent=1))
            if p == '/lineup.m3u':
                L = ['#EXTM3U']
                for e in lineup(host):
                    if not e['URL'] or e['Delivery'] in ('app', 'guide', 'pending'): continue
                    L += ['#EXTINF:-1 tvg-chno="%s" tvg-name="%s" group-title="%s",%s %s' % (e['GuideNumber'], e['GuideName'], e['Delivery'], e['GuideNumber'], e['GuideName']),
                          e.get('HLS') or e['URL']]
                return self.out('\n'.join(L) + '\n', 'audio/x-mpegurl')
            if p == '/guide.json': return self.out(json.dumps(guide_json()))
            if p == '/services.json': return self.out(json.dumps(S.services, indent=1))
            if len(parts) == 3 and parts[:2] == ['esg', 'icon']:
                for f in S.flows.values():
                    if parts[2] in f.files: return self.out(f.files[parts[2]][0], 'image/png', extra={'Cache-Control': 'max-age=600'})
                return self.nf()
            if parts and parts[0] == 'tables':
                if len(parts) == 1:
                    return self.out('<!doctype html>' + STYLE + '<h2>Low level signalling</h2><ul>' + ''.join('<li><a href="/tables/%s.xml">%s</a> version %s, last seen %s (<a href="/tables/%s.xml?raw">raw</a>)</li>' % (k, k, v[2], time.strftime('%H:%M:%S', time.localtime(v[1])), k) for k, v in sorted(S.lls.items())) + '</ul><p><small>Only the tables the station actually broadcasts are listed; it sends no AEAT, RRT or OnscreenMessageNotification.</small></p>', 'text/html; charset=utf-8')
                t = S.lls.get(parts[1].replace('.xml', ''))
                if not t: return self.nf()
                x = t[0].decode('utf-8', 'replace')
                if 'raw' in self.path: return self.out(x, 'application/xml')
                # browsers show a bare XML document as its text nodes only, which for these is nothing: pretty-print as text
                try:
                    import xml.dom.minidom
                    x = xml.dom.minidom.parseString(x.encode('utf-8')).toprettyxml(indent='  ')
                    x = '\n'.join(l for l in x.split('\n') if l.strip())
                except Exception: pass
                return self.out(x, 'text/plain; charset=utf-8')
            if len(parts) == 2 and parts[0] == 'app':
                s = service(parts[1]); f = S.flows.get(s['flow']) if s else None
                m = re.search(rb'bbandEntryPageUrl="([^"]+)"', f.sls.get('held.held', b'')) if f else None
                if not m: return self.nf('no application entry page has been received for that service')
                return self.out('', 'text/plain', 302, {'Location': m.group(1).decode().replace('&amp;', '&')})
            if len(parts) >= 2 and parts[0] == 'sls':
                s = service(parts[1]); f = S.flows.get(s['flow']) if s else None
                if not f: return self.nf()
                if len(parts) == 2: return self.out(json.dumps(sorted(f.sls)))
                b = f.sls.get(parts[2])
                return self.out(b, 'application/xml') if b is not None else self.nf()
            if len(parts) >= 3 and parts[0] == 'live':
                sid = parts[1]; s = service(sid); f = S.flows.get(s['flow']) if s else None
                if not f: return self.nf('that service has not been received yet')
                name = parts[2]
                if name in ('master.m3u8', 'video.m3u8'):
                    m = master(sid, video_only=name == 'video.m3u8')
                    return self.out(m, 'application/vnd.apple.mpegurl') if m else self.nf('no video track yet')
                if name == 'manifest.mpd':
                    b = f.sls.get('mpd.mpd')
                    return self.out(b, 'application/dash+xml') if b else self.nf('no manifest yet')
                if name == 'go' and len(parts) == 5:
                    v = f.reps.get(parts[3])
                    if not v or not v.get('base'): return self.nf()
                    t = parts[4].split('.')[0]
                    if parts[4] == 'init.mp4': loc = v['base'] + v['init']
                    elif t.isdigit(): loc = v['base'] + v['media'].replace('$Time$', t).replace('$Number$', str(int(t) // (v['timeline'][0][1] if v['timeline'] else 1)))
                    else: return self.nf()
                    return self.out('', 'text/plain', 302, {'Location': loc})
                m = re.fullmatch(r'r_(.+)\.m3u8', name)
                if m:
                    pl = internet_media(f, m.group(1))
                    return self.out(pl, 'application/vnd.apple.mpegurl') if pl else self.nf()
                m = re.fullmatch(r't(\d+)\.m3u8', name)
                if m:
                    pl = media(sid, int(m.group(1)))
                    return self.out(pl, 'application/vnd.apple.mpegurl') if pl else self.nf()
                if name.isdigit() and len(parts) == 4:
                    tr = f.tracks.get(int(name))
                    if not tr: return self.nf()
                    if parts[3] == 'init.mp4':
                        return self.out(tr['init'][1], 'video/mp4') if tr['init'] else self.nf()
                    toi = parts[3].split('.')[0]
                    seg = tr['segs'].get(int(toi)) if toi.isdigit() else None
                    return self.out(seg[0], 'video/mp4') if seg and seg[0] else self.nf('segment is gone or not here yet')
                # the names the broadcast manifest uses
                for tsi, ls in f.ls.items():
                    tr = f.tracks.get(tsi)
                    if not tr: continue
                    for toi, (n, enc) in ls['files'].items():
                        if os.path.basename(n) == name and tr['init'] and tr['init'][0] == toi: return self.out(tr['init'][1], 'video/mp4')
                    if ls['template']:
                        rx = re.escape(ls['template']).replace(re.escape('$TOI$'), r'(\d+)')
                        m = re.fullmatch(rx, name)
                        if m:
                            seg = tr['segs'].get(int(m.group(1)))
                            return self.out(seg[0], 'video/mp4') if seg and seg[0] else self.nf('segment is gone or not here yet')
                return self.nf()
        if len(parts) == 2 and parts[0] == 'auto':
            return self.ts(parts[1].lstrip('v').removesuffix('.mkv'), mkv=parts[1].endswith('.mkv'))
        self.nf()

    def ts(self, number, mkv=False):
        m = re.search(r'audio=(\w+)', self.path)
        ml = re.search(r'lead=([\d.]+)', self.path)
        lead = min(20.0, float(ml.group(1))) if ml else TS_LEAD
        with lock:
            s = next((x for x in S.services if chan(x) == number or x['id'] == number), None)
            pick = audio_choice(s['id'], m.group(1) if m else 'auto') if s else None
            f = S.flows.get(s['flow']) if s else None
            d = 2.0
            if f:
                d = next((r['duration'] for r in f.reps.values() if r['kind'] == 'video' and r.get('duration')), 2.0)
        # start this many segments back, so the player gets a cushion before the stream settles to real time
        back = str(-max(3, int(round(lead / d)) + 1)) if not (f and f.internet) else str(-max(2, int(round(lead / d))))
        if not s: return self.nf('no such channel')
        if not pick: return self.nf('that channel has no video yet')
        if not os.path.exists(AC3CLI): return self.ts_hls(s, f, pick, back)
        fold = 2 if re.search(r'ch=2', self.path) else 0
        nch = 2 if fold else 8 if re.search(r'ch=8', self.path) and pick[3] > 6 else min(pick[3], 6)
        layout = {2: 'stereo', 6: '5.1', 8: '7.1'}[nch]
        # the same instant in both tracks: the video from `back` segments ago, the audio segment whose decode time matches
        vsrc = Track(f, rid=pick[0][2:-5]) if f.internet else Track(f, tsi=int(pick[0][1:-5]))
        vk = vsrc.keys()
        if not vk: return self.nf('no video segment yet')
        v0, t_v = vk[max(0, len(vk) + int(back))]
        if f.internet:
            r = f.reps[vsrc.rid]; e = vsrc.edge()
            if e is not None:
                v0 = e + int(back) * r['timeline'][0][1] + r['timeline'][0][1]; t_v = v0 / r['timescale']
        a0 = None; offset = 0.0; asrc = None
        if pick[1]:
            asrc = Track(f, rid=pick[1][2:-5]) if f.internet else Track(f, tsi=int(pick[1][1:-5]))
            cands = [(abs(t - t_v), k, t) for k, t in asrc.keys()]
            if cands:
                d, a0, t_a = min(cands); offset = t_a - t_v
        csrc = None
        if mkv and not f.internet:
            ct = next((t for t in tracks_of(f) if t[1] == 'subtitles'), None)
            if ct:
                csrc = Track(f, tsi=ct[0]); ck = [k for k, t in csrc.keys() if t <= t_v + 0.5]
                c0 = ck[-1] if ck else (csrc.keys() or [(None,)])[0][0]
        vr, vw = os.pipe(); fds = [vr]
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:%d' % vr]
        outs = ['-map', '0:v:0', '-c:v', 'copy']
        if a0 is not None:
            ar, aw = os.pipe(); fds.append(ar)
            cmd += ['-f', 'f32le', '-ar', '48000', '-ac', str(nch), '-channel_layout', layout, '-itsoffset', '%.6f' % offset, '-i', 'pipe:%d' % ar]
            outs += ['-map', '1:a:0', '-c:a', 'libopus', '-mapping_family', '1', '-b:a', {2: '128k', 6: '256k', 8: '320k'}[nch]]
        if csrc and c0 is not None:
            cr, cw = os.pipe(); fds.append(cr)
            cmd += ['-i', 'pipe:%d' % cr]
            outs += ['-map', '%d:s:0' % (len(fds) - 1), '-c:s', 'copy', '-metadata:s:s:0', 'language=eng', '-metadata:s:s:0', 'title=captions']
        cmd += outs
        if mkv: cmd += ['-max_interleave_delta', '2000000', '-flush_packets', '1', '-f', 'matroska', '-cluster_time_limit', '500', 'pipe:1']
        else: cmd += ['-muxdelay', '0', '-muxpreload', '0', '-flush_packets', '1', '-f', 'mpegts', '-mpegts_flags', 'resend_headers', 'pipe:1']
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(os.path.join(STATE, 'ffmpeg.log'), 'ab'), pass_fds=fds)
        for r in fds: os.close(r)
        threading.Thread(target=feed_video, args=(vsrc, v0, os.fdopen(vw, 'wb')), daemon=True).start()
        if a0 is not None: threading.Thread(target=feed_audio, args=(asrc, a0, os.fdopen(aw, 'wb'), fold, nch), daemon=True).start()
        if csrc and c0 is not None: threading.Thread(target=feed_captions, args=(csrc, c0, os.fdopen(cw, 'wb'), t_v), daemon=True).start()
        self.pump(pr, 'video/x-matroska' if mkv else 'video/mp2t')

    def ts_hls(self, s, f, pick, back):
        """a gateway without ac3forge: FFmpeg follows the HLS playlist itself, video only"""
        base = 'http://127.0.0.1:%d/live/%s/' % (self.server.server_address[1], s['id'])
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-live_start_index', back, '-i', base + pick[0], '-map', '0:v:0', '-c', 'copy',
               '-muxdelay', '0', '-muxpreload', '0', '-flush_packets', '1', '-f', 'mpegts', '-mpegts_flags', 'resend_headers', 'pipe:1']
        self.pump(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))

    def pump(self, pr, ctype='video/mp2t'):
        self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Connection', 'close'); self.end_headers()
        self.close_connection = True
        try:
            while True:
                b = pr.stdout.read(188 * 64)
                if not b: break
                self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError): pass
        finally: pr.kill()


TS_AUDIO = False
TS_LEAD = 8.0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gain', type=float, default=34); ap.add_argument('--port', type=int, default=8023); ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--file', help='decode a 6.912 MS/s recording in place of the radio'); ap.add_argument('--pace', action='store_true', help='with --file: play it at real speed')
    ap.add_argument('--ts-lead', type=float, default=8.0, help='seconds of programme a new MPEG-TS client is given at once')
    a = ap.parse_args()
    TS_LEAD = a.ts_lead
    threading.Thread(target=decoder, args=(a,), daemon=True).start()
    print('gateway on http://%s:%d/' % (a.bind, a.port), flush=True)
    srv = ThreadingHTTPServer((a.bind, a.port), H); srv.daemon_threads = True
    srv.serve_forever()
