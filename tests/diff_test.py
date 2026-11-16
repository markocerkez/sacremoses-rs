"""Differential test: Rust fast path vs. original sacremoses, must be byte-identical.

Usage: python3 -I diff_test.py <path-to-original-sacremoses-checkout> <corpus.txt> [n_lines] [n_fuzz]
"""
import itertools
import random
import sys

orig_path, corpus_path = sys.argv[1], sys.argv[2]
N_LINES = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
N_FUZZ = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
sys.path.insert(0, orig_path)

import sacremoses_rs as rs
from sacremoses.normalize import MosesPunctNormalizer as PyN
from sacremoses.tokenize import MosesDetokenizer as PyD
from sacremoses.tokenize import MosesTokenizer as PyT


def run(fn, *a, **k):
    try:
        return ("ok", fn(*a, **k))
    except Exception as e:  # compare exception types too
        return ("exc", type(e).__name__)


fails = {}
counts = {}


def check(kind, ref, got, ctx):
    counts[kind] = counts.get(kind, 0) + 1
    if ref != got:
        fails.setdefault(kind, []).append((ctx, ref, got))


lines = [l.rstrip("\n") for l in open(corpus_path, encoding="utf-8", errors="replace") if l.strip()][:N_LINES]

rnd = random.Random(1234)
ALPHA = list("abcxyzABCXYZ0123456789") + list(".,.,.,''\"\"--!?;:()[]{}<>&|@#$%/\\`~_*+=^") + [
    " ", " ", " ", "  ", "\t", "\n", "\x1c", "\x1f", "\x85", "\xa0", "　", " ", "\x00", "\x01", "\x0b",
    "é", "ß", "ñ", "ı", "İ", "中", "文", "ア", "한", "₹", "€", "£", "©", "°", "…", "“", "”", "‘", "’", "«", "»", "„", "–", "—",
    "，", "。", "！", "０", "९", "्", "़", "²", "٣", "ǅ", "ᾼ", "́",
    "DOT", "MULTI", "DOTMULTI", "THISISPROTECTED", "THISISPROTECTED000", "Mr.", "No.", "etc.", "U.S.", "...", "....",
    "'s", "n't", "'ll", "'re", "cannot", "gonna", ".'", ".\"", "http://x.y/z", "a@b.cd", "&amp;", "&#124;", "&lt;", "&quot;",
]


def fuzz(n):
    for _ in range(n):
        k = rnd.randint(0, 25)
        yield "".join(rnd.choice(ALPHA) for _ in range(k))


fuzz_texts = list(fuzz(N_FUZZ))

# ---------------- tokenizer ----------------
LANGS = ["en", "fr", "it", "de", "es", "cs", "fi", "ru", "zh", "ja", "ko", "ca", "pt", "el", "ga", "hi", "ta", "pl", "sv"]
for lang in LANGS:
    t = rs.MosesTokenizer(lang=lang)
    texts = (lines if lang in ("en", "fr", "de", "zh") else lines[:3000]) + (fuzz_texts if lang in ("en", "fr", "zh", "cs") else fuzz_texts[:8000])
    for agg, esc, rstr in itertools.product([False, True], [True, False], [True, False]):
        use = texts if (agg, esc, rstr) == (False, True, True) or lang == "en" else texts[:4000]
        for x in use:
            ref = run(PyT.tokenize, t, x, aggressive_dash_splits=agg, return_str=rstr, escape=esc)
            got = run(t.tokenize, x, aggressive_dash_splits=agg, return_str=rstr, escape=esc)
            check(f"tok[{lang}]", ref, got, (x, agg, esc, rstr))

# custom nonbreaking prefix file behaviour and mutation of the prefix list
t = rs.MosesTokenizer(lang="en")
t.NONBREAKING_PREFIXES.append("zzq")
for x in ["we met zzq. Smith there", "ok zzq. And"]:
    check("tok[mutated-prefixes]", run(PyT.tokenize, t, x, return_str=True), run(t.tokenize, x, return_str=True), x)

# protected patterns fall back
pp = rs.MosesTokenizer(lang="en")
for x in ["see http://a.b/c now", "mail a@b.com ok"]:
    check("tok[protected]", run(PyT.tokenize, pp, x, return_str=True, protected_patterns=pp.WEB_PROTECTED_PATTERNS),
          run(pp.tokenize, x, return_str=True, protected_patterns=pp.WEB_PROTECTED_PATTERNS), x)

# non-str input / surrogates
for x in [123, None, ["a"], "a\ud800b", b"abc"]:
    check("tok[odd-input]", run(PyT.tokenize, pp, x, return_str=True), run(pp.tokenize, x, return_str=True), repr(x))

# batch API equals per-item
t = rs.MosesTokenizer(lang="en")
check("tok[many]", [PyT.tokenize(t, x, return_str=True) for x in lines[:2000]], t.tokenize_many(lines[:2000]), "many")

# ---------------- normalizer ----------------
combos = itertools.product(["en", "de", "fr", "es", "cs", "ru", "zh"], [True, False], [True, False], [True, False], [True, False], [True, False], [True, False])
ntexts = lines[:1500] + fuzz_texts[:6000]
for lang, penn, nqc, nn, pre, post, parity in combos:
    n = rs.MosesPunctNormalizer(lang=lang, penn=penn, norm_quote_commas=nqc, norm_numbers=nn,
                                pre_replace_unicode_punct=pre, post_remove_control_chars=post, perl_parity=parity)
    sample = ntexts if (lang, penn, nqc, nn, pre, post, parity) in {("en", True, True, True, False, False, False), ("fr", True, True, True, True, True, True)} else ntexts[::10]
    for x in sample:
        check(f"norm[{lang}]", run(PyN.normalize, n, x), run(n.normalize, x), (x, penn, nqc, nn, pre, post, parity))

# ---------------- detokenizer ----------------
for lang in ["en", "fr", "it", "ga", "cs", "fi", "ko", "zh", "ja", "de", "es", "ru"]:
    t = rs.MosesTokenizer(lang=lang if lang not in ("ga",) else "en")
    d = rs.MosesDetokenizer(lang=lang)
    token_lists = [t.tokenize(x) for x in lines[:4000]] + [x.split(" ") for x in fuzz_texts[:20000]]
    token_lists += [[w for w in x.split(" ")] for x in lines[:1000]]
    for tl in token_lists:
        for unesc, rstr in [(True, True), (False, True), (True, False)]:
            check(f"detok[{lang}]", run(PyD.tokenize, d, tl, return_str=rstr, unescape=unesc),
                  run(d.tokenize, tl, return_str=rstr, unescape=unesc), (tl, unesc, rstr))
# weird inputs
d = rs.MosesDetokenizer("en")
for tl in [[], [""], [" "], ("a", "b"), (w for w in ["x", "y"]), ["a", 1], "abc", None]:
    ref = run(PyD.tokenize, d, list(tl) if hasattr(tl, "__iter__") and not isinstance(tl, str) and tl is not None and not isinstance(tl, list) else tl)
    got = run(d.tokenize, tl if not hasattr(tl, "__next__") else ["x", "y"])
    if not hasattr(tl, "__next__") and not isinstance(tl, tuple):
        check("detok[odd]", ref, got, repr(tl))

# ---------------- report ----------------
total = sum(counts.values())
bad = sum(len(v) for v in fails.values())
print(f"cases: {total:,}   mismatches: {bad:,}")
for k in sorted(counts):
    print(f"  {k:28s} cases={counts[k]:>9,} bad={len(fails.get(k, [])):>6,}")
shown = 0
for k, v in fails.items():
    for ctx, ref, got in v[:3]:
        print("\nMISMATCH", k, "ctx=", repr(ctx)[:200], "\n  ref:", repr(ref)[:300], "\n  got:", repr(got)[:300])
        shown += 1
sys.exit(1 if bad else 0)
