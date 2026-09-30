# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure a live MPEG-TS stream: how far the delivered video timestamps run ahead of the wall clock.
A player can only play smoothly while that lead stays above zero.   tsprobe.py URL seconds"""
import sys, time, urllib.request
url, secs = sys.argv[1], float(sys.argv[2])
r = urllib.request.urlopen(url, timeout=20)
t0 = time.time(); buf = b''; first = None; last = None; rows = []; nbytes = 0; tprev = t0; gaps = []
while time.time() - t0 < secs:
    b = r.read(188 * 40)
    if not b: break
    now = time.time()
    if now - tprev > 0.7: gaps.append((round(tprev - t0, 2), round(now - tprev, 2)))
    tprev = now
    nbytes += len(b); buf += b
    while len(buf) >= 188:
        p = buf[:188]; buf = buf[188:]
        if p[0] != 0x47: buf = buf[1:]; continue
        pid = ((p[1] & 0x1f) << 8) | p[2]
        if pid != 0x100 or not (p[1] & 0x40): continue
        o = 4 + (1 + p[4] if p[3] & 0x20 else 0)
        if o + 14 > 188 or p[o:o + 3] != b'\x00\x00\x01' or not (p[o + 7] & 0x80): continue
        q = p[o + 9:o + 14]
        pts = (((q[0] >> 1) & 7) << 30 | q[1] << 22 | (q[2] >> 1) << 15 | q[3] << 7 | (q[4] >> 1)) / 90000
        if first is None: first = (now, pts)
        last = pts
    if first and last is not None:
        rows.append((now - first[0], last - first[1]))
print('%.0f s, %.2f Mbit/s' % (time.time() - t0, nbytes * 8 / (time.time() - t0) / 1e6))
step = 2.0; k = 0
print('wall(s)  media delivered(s)  lead(s)')
for w, m in rows:
    if w >= k * step:
        print('%6.1f   %8.2f          %+6.2f' % (w, m, m - w)); k += 1
lead = [m - w for w, m in rows if w > 3]
print('lead after the first 3 s: min %.2f  median %.2f  max %.2f' % (min(lead), sorted(lead)[len(lead) // 2], max(lead)))
print('pauses in delivery longer than 0.7 s (at, length):', gaps[:20])
