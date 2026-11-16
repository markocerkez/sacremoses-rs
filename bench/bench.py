"""Benchmark: original sacremoses vs sacremoses_rs.
Usage: python3 -I bench.py <path-to-original-sacremoses-checkout> <corpus.txt> [n_lines]
"""
import sys, time
orig, corpus = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
sys.path.insert(0, orig)
import sacremoses_rs as rs
from sacremoses.normalize import MosesPunctNormalizer as PyN
from sacremoses.tokenize import MosesDetokenizer as PyD, MosesTokenizer as PyT

lines = [l.rstrip("\n") for l in open(corpus, encoding="utf-8", errors="replace") if l.strip()][:n]
mb = sum(len(l.encode()) for l in lines) / 1e6

def best(fn, reps=3):
    r = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); r.append(time.perf_counter() - t)
    return min(r), out

pn, rn = PyN(), rs.MosesPunctNormalizer()
pt, rt = PyT("en"), rs.MosesTokenizer("en")
pd, rd = PyD("en"), rs.MosesDetokenizer("en")

rows = []
a, normed = best(lambda: [pn.normalize(l) for l in lines]); b, got = best(lambda: [rn.normalize(l) for l in lines]); assert normed == got
rows.append(("punct normalize", a, b))
b2, got = best(lambda: rn.normalize_many(lines)); assert normed == got
rows.append(("punct normalize (batch)", a, b2))
a, toks = best(lambda: [pt.tokenize(l, return_str=True) for l in normed]); b, got = best(lambda: [rt.tokenize(l, return_str=True) for l in normed]); assert toks == got
rows.append(("tokenize (return_str)", a, b))
a, tl = best(lambda: [pt.tokenize(l) for l in normed]); b, got = best(lambda: [rt.tokenize(l) for l in normed]); assert tl == got
rows.append(("tokenize (token list)", a, b))
b2, got = best(lambda: rt.tokenize_many(normed)); assert toks == got
rows.append(("tokenize (batch)", a, b2))
a, det = best(lambda: [pd.detokenize(x) for x in tl]); b, got = best(lambda: [rd.detokenize(x) for x in tl]); assert det == got
rows.append(("detokenize", a, b))
b2, got = best(lambda: rd.detokenize_many(tl)); assert det == got
rows.append(("detokenize (batch)", a, b2))
a, _ = best(lambda: [pt.tokenize(l, aggressive_dash_splits=True, return_str=True) for l in normed]); b, _ = best(lambda: [rt.tokenize(l, aggressive_dash_splits=True, return_str=True) for l in normed])
rows.append(("tokenize (aggressive dashes)", a, b))

import os
print(f"{len(lines):,} lines, {mb:.2f} MB, Python {sys.version.split()[0]}, {os.cpu_count()} CPU threads (batch rows use all of them)\n")
print(f"{'step':32s} {'python':>9s} {'rust':>9s} {'speedup':>9s}   {'MB/s rust':>9s}")
for name, a, b in rows:
    print(f"{name:32s} {a:8.3f}s {b:8.3f}s {a/b:8.1f}x   {mb/b:9.1f}")
