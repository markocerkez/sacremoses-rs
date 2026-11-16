"""Self-contained tests (no corpus needed). Run: pytest tests/test_basic.py"""
import threading

import pytest
import sacremoses
from sacremoses.normalize import MosesPunctNormalizer as PyN
from sacremoses.tokenize import MosesDetokenizer as PyD
from sacremoses.tokenize import MosesTokenizer as PyT

import sacremoses_rs as rs

SENTENCES = [
    "Hello, world... it's 5,300 dollars (approx.) -- \"quoted\" e.g. Mr. Smith's dog.",
    "They can't go; she'll stay & he won't.  Cost: $5.50/kg - 20% off!",
    "Visit http://example.com/a?b=c or mail me@example.org today...",
    "Dr. Müller aus Zürich sagte: „Das ist ein Test.“ Nr. 5 gilt.",
    "L'homme qu'il a vu s'est enfui. C'est l'été !",
    "中文 日本語 한국어 mixed with English, 12,345.67 and 3.14.",
    "",
    "   ",
    "a\tb\nc\x00d\x1fe f　g",
]


def test_tokenize_golden():
    t = rs.MosesTokenizer("en")
    assert t.tokenize("Hello, world... it's 5,300 dollars.", return_str=True) == (
        "Hello , world ... it &apos;s 5,300 dollars ."
    )
    assert t.tokenize("a b") == ["a", "b"]


@pytest.mark.parametrize("lang", ["en", "fr", "de", "it", "cs", "zh", "ja", "ko", "fi"])
@pytest.mark.parametrize("aggressive", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_tokenizer_matches_original(lang, aggressive, escape):
    t = rs.MosesTokenizer(lang)
    for s in SENTENCES:
        for ret in (True, False):
            assert t.tokenize(s, aggressive_dash_splits=aggressive, return_str=ret, escape=escape) == PyT.tokenize(
                t, s, aggressive_dash_splits=aggressive, return_str=ret, escape=escape
            )


@pytest.mark.parametrize("lang", ["en", "fr", "de", "cs", "fi", "ko", "zh"])
def test_detokenizer_roundtrip_matches_original(lang):
    t, d = rs.MosesTokenizer(lang), rs.MosesDetokenizer(lang)
    for s in SENTENCES:
        toks = t.tokenize(s)
        assert d.detokenize(toks) == PyD.tokenize(d, toks)


@pytest.mark.parametrize("kwargs", [{}, {"penn": False}, {"perl_parity": True}, {"pre_replace_unicode_punct": True, "post_remove_control_chars": True}])
def test_normalizer_matches_original(kwargs):
    n = rs.MosesPunctNormalizer("en", **kwargs)
    for s in SENTENCES + ["“Hi” – «x»…", "(a) ,b ;c  :d 5 %", "’‘´´"]:
        assert n.normalize(s) == PyN.normalize(n, s)


def test_protected_patterns_fall_back():
    t = rs.MosesTokenizer("en")
    s = "see http://a.b/c now"
    assert t.tokenize(s, return_str=True, protected_patterns=t.WEB_PROTECTED_PATTERNS) == PyT.tokenize(
        t, s, return_str=True, protected_patterns=t.WEB_PROTECTED_PATTERNS
    )


def test_odd_input_behaves_like_original():
    t = rs.MosesTokenizer("en")
    assert t.tokenize("a\ud800b", return_str=True) == PyT.tokenize(t, "a\ud800b", return_str=True)
    assert t.tokenize(123, return_str=True) == PyT.tokenize(t, 123, return_str=True)
    with pytest.raises(TypeError):
        rs.MosesDetokenizer("en").detokenize(["a", 1])


def test_batch_preserves_order_and_matches():
    t, n, d = rs.MosesTokenizer("en"), rs.MosesPunctNormalizer("en"), rs.MosesDetokenizer("en")
    texts = SENTENCES * 500
    assert t.tokenize_many(texts) == [PyT.tokenize(t, s, return_str=True) for s in texts]
    assert n.normalize_many(texts) == [PyN.normalize(n, s) for s in texts]
    tl = [t.tokenize(s) for s in texts]
    assert d.detokenize_many(tl) == [PyD.tokenize(d, x) for x in tl]
    assert t.tokenize_many([]) == []


def test_threads_share_one_tokenizer():
    t = rs.MosesTokenizer("en")
    expected = [PyT.tokenize(t, s, return_str=True) for s in SENTENCES]
    errors = []

    def work():
        for _ in range(200):
            if [t.tokenize(s, return_str=True) for s in SENTENCES] != expected:
                errors.append(1)

    ts = [threading.Thread(target=work) for _ in range(4)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    assert not errors


def test_patch_accelerates_original_classes_in_place():
    rs.patch()
    assert sacremoses.MosesTokenizer.tokenize is rs._fast_tokenize
    t = sacremoses.MosesTokenizer("en")
    assert t.tokenize("it's, ok.", return_str=True) == "it &apos;s , ok ."
    rs.patch()  # idempotent
    assert t.tokenize("x", return_str=True) == "x"


def test_prefix_list_change_is_noticed():
    t = rs.MosesTokenizer("en")
    before = t.tokenize("we met zzq. Smith", return_str=True)
    t.NONBREAKING_PREFIXES.append("zzq")
    assert t.tokenize("we met zzq. Smith", return_str=True) != before
