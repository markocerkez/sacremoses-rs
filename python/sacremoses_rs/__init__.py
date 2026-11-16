"""Rust-accelerated drop-in replacements for sacremoses' Moses classes.

    from sacremoses_rs import MosesTokenizer, MosesDetokenizer, MosesPunctNormalizer

or accelerate everything that already uses sacremoses (CLI, truecaser, ...):

    import sacremoses_rs; sacremoses_rs.patch()

The classes subclass the originals, so every attribute and option keeps working.
Anything the Rust core does not handle (custom ``protected_patterns``, strings
containing lone surrogates, a few rare language branches) transparently falls
back to the original Python implementation, so results are always identical.
"""
from sacremoses.normalize import MosesPunctNormalizer as _PyNormalizer
from sacremoses.tokenize import MosesDetokenizer as _PyDetokenizer
from sacremoses.tokenize import MosesTokenizer as _PyTokenizer
from sacremoses.util import _CJKChars_ranges

from . import _core

__all__ = ["MosesTokenizer", "MosesDetokenizer", "MosesPunctNormalizer", "patch"]
__version__ = "0.1.0"

# Errors that mean "the Rust path cannot take this input" (lone surrogates,
# non-str items, ...). The original implementation then produces the result
# (or the original exception).
_FALLBACK = (UnicodeError, TypeError, ValueError)

# The original methods, captured before anything is patched. They are the
# fallback, and they are called explicitly (never through ``super()``) so that
# ``patch()`` cannot make a method call itself.
_orig_tokenize = _PyTokenizer.tokenize
_orig_detok = _PyDetokenizer.tokenize
_orig_normalize = _PyNormalizer.normalize


# --------------------------------------------------------------------- tokenizer
def _core_tok(self):
    key = (
        self.lang,
        self.IsAlnum,
        self.IsAlpha,
        self.IsN,
        self.IsLower,
        id(self.NONBREAKING_PREFIXES),
        len(self.NONBREAKING_PREFIXES),
        id(self.NUMERIC_ONLY_PREFIXES),
        len(self.NUMERIC_ONLY_PREFIXES),
    )
    rs = self.__dict__.get("_rs")
    if rs is None or self.__dict__.get("_rs_key") != key:
        rs = _core.Tokenizer(
            self.lang,
            self.IsAlnum,
            self.IsAlpha,
            self.IsN,
            self.IsLower,
            list(self.NONBREAKING_PREFIXES),
            list(self.NUMERIC_ONLY_PREFIXES),
        )
        self.__dict__["_rs"] = rs
        self.__dict__["_rs_key"] = key
    return rs


def _fast_tokenize(
    self,
    text,
    aggressive_dash_splits=False,
    return_str=False,
    escape=True,
    protected_patterns=None,
):
    if not protected_patterns and isinstance(text, str):
        try:
            rs = _core_tok(self)
            if return_str:
                return rs.tokenize(text, bool(aggressive_dash_splits), bool(escape))
            return rs.tokenize_split(text, bool(aggressive_dash_splits), bool(escape))
        except _FALLBACK:
            pass
    return _orig_tokenize(
        self,
        text,
        aggressive_dash_splits=aggressive_dash_splits,
        return_str=return_str,
        escape=escape,
        protected_patterns=protected_patterns,
    )


def _tokenize_many(self, texts, aggressive_dash_splits=False, escape=True):
    """Tokenize many strings at once and return a list of tokenized strings.

    Extra API (not in sacremoses): one call into Rust for the whole batch.
    """
    texts = list(texts)
    try:
        return _core_tok(self).tokenize_many(
            texts, bool(aggressive_dash_splits), bool(escape)
        )
    except _FALLBACK:
        return [
            _fast_tokenize(self, t, aggressive_dash_splits, True, escape) for t in texts
        ]


# ------------------------------------------------------------------- detokenizer
def _core_detok(self):
    rs = self.__dict__.get("_rs")
    if rs is None or self.__dict__.get("_rs_lang") != self.lang:
        rs = _core.Detokenizer(
            self.lang,
            self.IsAlpha,
            self.IsSc,
            list(_CJKChars_ranges),
            self.FINNISH_REGEX.pattern,
        )
        self.__dict__["_rs"] = rs
        self.__dict__["_rs_lang"] = self.lang
    return rs


def _fast_detok(self, tokens, return_str=True, unescape=True):
    tokens = tokens if isinstance(tokens, list) else list(tokens)
    try:
        res = _core_detok(self).detokenize(tokens, bool(unescape))
    except _FALLBACK:
        res = None
    if res is None:
        return _orig_detok(self, tokens, return_str, unescape)
    return res if return_str else res.split()


# ---------------------------------------------------------------------- normalizer
def _core_norm(self):
    rs = self.__dict__.get("_rs")
    if rs is None:
        rs = _core.Normalizer(
            [(p, r) for p, r in self.substitutions],
            [(p, r) for p, r in self.REPLACE_UNICODE_PUNCTUATION],
            bool(self.pre_replace_unicode_punct),
            bool(self.post_remove_control_chars),
        )
        self.__dict__["_rs"] = rs
    return rs


def _fast_normalize(self, text):
    if isinstance(text, str):
        try:
            return _core_norm(self).normalize(text)
        except _FALLBACK:
            pass
    return _orig_normalize(self, text)


def _normalize_many(self, texts):
    """Extra API: normalize a batch of strings with one call into Rust."""
    texts = list(texts)
    try:
        return _core_norm(self).normalize_many(texts)
    except _FALLBACK:
        return [_fast_normalize(self, t) for t in texts]


# ------------------------------------------------------------------- public classes
class MosesTokenizer(_PyTokenizer):
    tokenize = _fast_tokenize
    tokenize_many = _tokenize_many


class MosesDetokenizer(_PyDetokenizer):
    tokenize = _fast_detok


class MosesPunctNormalizer(_PyNormalizer):
    normalize = _fast_normalize
    normalize_many = _normalize_many


def patch():
    """Accelerate the original classes in place.

    After this, every user of ``sacremoses`` (including its CLI and truecaser)
    runs on the Rust core; class names and identities are unchanged.
    """
    _PyTokenizer.tokenize = _fast_tokenize
    _PyTokenizer.tokenize_many = _tokenize_many
    _PyDetokenizer.tokenize = _fast_detok
    _PyNormalizer.normalize = _fast_normalize
    _PyNormalizer.normalize_many = _normalize_many
