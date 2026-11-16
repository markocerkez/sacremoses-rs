# sacremoses-rs

Rust-accelerated drop-in classes for [sacremoses](https://github.com/hplt-project/sacremoses)
(the Python port of the Moses tokenizer, detokenizer and punctuation normalizer).
Same API, same output, byte for byte, much faster.

```python
from sacremoses_rs import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer

tok = MosesTokenizer("en")
tok.tokenize("Hello, world... it's 5,300 dollars.", return_str=True)

# batch API (extra): one call, all CPU cores, GIL released, order preserved
tok.tokenize_many(list_of_sentences)
MosesPunctNormalizer("en").normalize_many(list_of_sentences)
MosesDetokenizer("en").detokenize_many(list_of_token_lists)

# or speed up everything that already uses sacremoses (CLI, truecaser, ...)
import sacremoses_rs; sacremoses_rs.patch()
```

The classes subclass the originals, so every option and attribute keeps working.
Requires `sacremoses>=0.2.0`.

## Speed

20,000 lines / 1.7 MB of English prose, Python 3.13, three runs, one core unless noted
(the batch rows use both cores of the 2-thread test machine; more cores scale further):

| step                         | speedup vs. python |
|------------------------------|--------------------|
| tokenize (`return_str=True`) | 28-30x             |
| tokenize (token list)        | 18-19x             |
| tokenize (aggressive dashes) | 27-30x             |
| tokenize (batch API)         | 50-55x             |
| punctuation normalize        | 4-6x               |
| punctuation normalize (batch)| 6-7x               |
| detokenize                   | 4-5x               |
| detokenize (batch)           | 5-6x               |

Reproduce: `python -I bench/bench.py <sacremoses checkout> <corpus.txt> 20000`

## How it was verified

* `tests/diff_test.py`: differential test, original vs. Rust, outputs must be identical.
  About 1.2 million cases per run: a real corpus plus adversarial fuzz (unicode whitespace,
  control characters, exotic scripts, the library's own marker strings, ...) across 19
  languages, all option combinations of the normalizer, the detokenizer, and the batch APIs.
  Result: 0 mismatches against both sacremoses 0.2.0 (PyPI) and the current git HEAD.
* `tests/test_basic.py`: 54 self-contained pytest tests (golden outputs, fallbacks,
  batch order, thread safety, `patch()`).
* sacremoses' own test suite with `patch()` applied gives the same results as without
  it (the truecaser tests that need network downloads fail identically either way).
* `cargo fmt` and `cargo clippy -D warnings` are clean.

sacremoses 0.1.x is not supported: its handling of inputs that literally contain the
text `DOTMULTI` differs from 0.2.0 (a quirk upstream fixed).

## What is NOT accelerated (falls back to the original Python, same results)

* `tokenize(..., protected_patterns=...)` and `penn_tokenize`
* strings containing lone surrogates, non-str input
* Czech detokenization of `x - li` / `e-mail` style tokens (the original raises
  a `TypeError` there; the fallback keeps that behaviour)

## Known limits

* Changing `NONBREAKING_PREFIXES` / `NUMERIC_ONLY_PREFIXES` of an existing tokenizer is
  noticed when the list object or its length changes, not when an item is edited in place.
* The normalizer uses the Rust `regex` crate for `\d` and `\p{C}`; for code points newly
  assigned in a recent Unicode version, Python and Rust may disagree.
* Everything above was measured on Linux x86-64 with Python 3.13. The GitHub Actions
  workflows (`.github/workflows`) that test and build Linux, macOS and Windows wheels
  are written but have not been run yet.

## Build

```
pip install maturin "sacremoses>=0.2.0"
maturin build --release -o dist && pip install dist/*.whl
pytest tests/test_basic.py
```

## License / attribution

MIT. The algorithms are ported from sacremoses (MIT, (c) 2020 alvations), which in turn
ports the Moses scripts. Keep both copyright lines.
