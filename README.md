# sacremoses-rs

Rust-accelerated drop-in classes for [sacremoses](https://github.com/hplt-project/sacremoses)
(the Python port of the Moses tokenizer, detokenizer and punctuation normalizer).
Same API, same output, byte for byte, much faster.

```python
from sacremoses_rs import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer

tok = MosesTokenizer("en")
tok.tokenize("Hello, world... it's 5,300 dollars.", return_str=True)
tok.tokenize_many(list_of_sentences)          # extra: whole batch in one call

# or speed up everything that already uses sacremoses (CLI, truecaser, ...)
import sacremoses_rs; sacremoses_rs.patch()
```

## Speed (20,000 lines / 1.7 MB of English prose, Python 3.13, one core)

| step                         | python  | rust    | speedup |
|------------------------------|---------|---------|---------|
| tokenize (return_str)        | 1.85 s  | 0.10 s  | 18x     |
| tokenize (token list)        | 2.40 s  | 0.12 s  | 20x     |
| tokenize (batch API)         | 2.40 s  | 0.07 s  | 35x     |
| tokenize (aggressive dashes) | 2.27 s  | 0.10 s  | 22x     |
| punctuation normalize        | 0.48 s  | 0.08 s  | 6x      |
| detokenize                   | 0.30 s  | 0.08 s  | 4x      |

Reproduce: `python3 -I bench/bench.py <sacremoses checkout> <corpus.txt> 20000`

## How it was verified

* `tests/diff_test.py`: differential test, original vs. Rust, outputs must be identical.
  1.2 million cases: a real corpus plus adversarial fuzz (unicode whitespace, control
  characters, surrogate-free exotic scripts, the library's own marker strings, ...)
  across 19 languages, all option combinations of the normalizer, and the detokenizer.
  Result: 0 mismatches.
* sacremoses' own test suite with `patch()` applied: same results as unpatched
  (32 pass; the 4 truecaser tests need network downloads and fail identically
  without the patch).
* The differential run found nothing, but the official suite found one real bug
  in `patch()` (infinite recursion); it is fixed and covered now.

## What is NOT accelerated (falls back to the original Python, same results)

* `tokenize(..., protected_patterns=...)` and `penn_tokenize`
* strings containing lone surrogates, non-str input
* Czech detokenization of `x - li` / `e-mail` style tokens (the original raises
  a `TypeError` there; the fallback keeps that behaviour)

## Known limits

* Changing `NONBREAKING_PREFIXES` / `NUMERIC_ONLY_PREFIXES` of an existing tokenizer is
  noticed when the list object or its length changes, not when an item is edited in place.
* The normalizer uses the Rust `regex` crate for `\d` and `\p{C}`; for code points that
  were newly assigned in a recent Unicode version, Python and Rust may disagree.
* Only built and tested on Linux x86-64 / Python 3.13 so far.

## Build

```
pip install maturin sacremoses
maturin build --release -o dist && pip install dist/*.whl
```

## License / attribution

MIT. The algorithms are ported from sacremoses (MIT, (c) 2020 alvations), which in turn
ports the Moses scripts. Keep both copyright lines.
