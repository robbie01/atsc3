"""Extract constant tables from the gr-atsc3 C++ sources (reference only, GPLv3, not copied)."""
import re, numpy as np, os
# the gr-atsc3 sources (GPLv3, github.com/drmpeg/gr-atsc3) are read for their A/322 constant tables, not copied
ROOT = os.path.join(os.environ.get('GR_ATSC3') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'third_party', 'gr-atsc3'), 'lib')
_cache = {}
def src(fn):
    if fn not in _cache:
        _cache[fn] = open(os.path.join(ROOT, fn)).read()
    return _cache[fn]

def table_text(fn, name):
    s = src(fn)
    m = re.search(r'[\w:]*\b' + re.escape(name) + r'\s*((\[[^\]]*\])+)\s*=\s*\{', s)
    if not m:
        raise KeyError(name)
    i = m.end() - 1
    depth = 0; j = i
    while True:
        ch = s[j]
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0: break
        j += 1
    return m.group(1), s[i:j + 1]

def nested(fn, name, conv=float):
    """return nested python lists"""
    dims, t = table_text(fn, name)
    t = re.sub(r'/\*.*?\*/', '', t, flags=re.S)
    t = re.sub(r'//[^\n]*', '', t)
    t = re.sub(r'gr_complex\(([^,]+),([^)]+)\)', r'complex(\1,\2)', t)
    t = t.replace('{', '[').replace('}', ']')
    t = re.sub(r',\s*\]', ']', t)
    return eval(t, {'complex': complex})

if __name__ == '__main__':
    import sys
    print(nested(sys.argv[1], sys.argv[2]))
