//! Rust fast paths for sacremoses (Moses tokenizer / detokenizer / punctuation normalizer).
//!
//! This is a port of the algorithms in https://github.com/hplt-project/sacremoses
//! (MIT, Copyright (c) 2020 alvations). The Python wrapper layer in
//! `python/sacremoses_rs` falls back to the original implementation for anything
//! this module does not handle (custom protected patterns, lone surrogates, ...).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Python's `str.isspace()` / `\s` for str patterns / `str.split()` whitespace.
#[inline]
fn is_py_space(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0d
            | 0x1c..=0x1f
            | 0x20
            | 0x85
            | 0xa0
            | 0x1680
            | 0x2000..=0x200a
            | 0x2028
            | 0x2029
            | 0x202f
            | 0x205f
            | 0x3000
    )
}

fn py_split(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|t| !t.is_empty()).collect()
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// A set of characters with an ASCII fast path and binary search for the rest.
#[derive(Clone)]
struct CharSet {
    ascii: [bool; 128],
    ranges: Vec<(u32, u32)>,
}

impl CharSet {
    fn new(s: &str) -> Self {
        let mut v: Vec<u32> = s.chars().map(|c| c as u32).collect();
        v.sort_unstable();
        v.dedup();
        let mut ascii = [false; 128];
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for &u in &v {
            if u < 128 {
                ascii[u as usize] = true;
            }
            match ranges.last_mut() {
                Some(last) if last.1 + 1 == u => last.1 = u,
                _ => ranges.push((u, u)),
            }
        }
        CharSet { ascii, ranges }
    }

    #[inline]
    fn has(&self, c: char) -> bool {
        let u = c as u32;
        if u < 128 {
            return self.ascii[u as usize];
        }
        self.ranges
            .binary_search_by(|&(lo, hi)| {
                if u < lo {
                    std::cmp::Ordering::Greater
                } else if u > hi {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
    }
}

/// `\s+` -> " " followed by removal of `[\000-\037]` (the two first Moses steps).
fn collapse_ws_and_junk(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_run = false;
    for c in s.chars() {
        if is_py_space(c) {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            in_run = false;
            if (c as u32) < 0x20 {
                continue;
            }
            out.push(c);
        }
    }
    out
}

/// `\s+` -> " " only.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_run = false;
    for c in s.chars() {
        if is_py_space(c) {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            in_run = false;
            out.push(c);
        }
    }
    out
}

/// Scan for `X ' Y` triples (X, Y single chars around `mid`), the shape shared by
/// all of the language specific apostrophe rules. `pred` decides whether a triple
/// matches, `rep` writes the replacement. Matches are leftmost and non-overlapping,
/// exactly like `re.sub`.
fn sub3<'a, P, R>(s: &'a str, mid: char, pred: P, rep: R) -> Cow<'a, str>
where
    P: Fn(char, char) -> bool,
    R: Fn(char, char, &mut String),
{
    if !s.contains(mid) {
        return Cow::Borrowed(s);
    }
    let cs: Vec<char> = s.chars().collect();
    let n = cs.len();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    while i < n {
        if i + 2 < n && cs[i + 1] == mid && pred(cs[i], cs[i + 2]) {
            rep(cs[i], cs[i + 2], &mut out);
            i += 3;
        } else {
            out.push(cs[i]);
            i += 1;
        }
    }
    Cow::Owned(out)
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[pyclass]
struct Tokenizer {
    lang: String,
    alnum: CharSet,
    alpha: CharSet,
    isn: CharSet,
    lower: CharSet,
    nbp: HashSet<String>,
    numeric_only: HashSet<String>,
    multidot_restore: Regex,
}

impl Tokenizer {
    fn pad_not_alnum(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len() + s.len() / 4);
        for c in s.chars() {
            let keep = self.alnum.has(c)
                || is_py_space(c)
                || matches!(c, '.' | '\'' | '`' | ',' | '-');
            if keep {
                out.push(c);
            } else {
                out.push(' ');
                out.push(c);
                out.push(' ');
            }
        }
        out
    }

    fn aggressive_hyphen_split(&self, s: &str) -> String {
        let cs: Vec<char> = s.chars().collect();
        let n = cs.len();
        let mut out = String::with_capacity(s.len() + 16);
        let mut i = 0;
        while i < n {
            if i + 2 < n && cs[i + 1] == '-' && self.alnum.has(cs[i]) && self.alnum.has(cs[i + 2]) {
                out.push(cs[i]);
                out.push_str(" @-@ ");
                i += 2;
            } else {
                out.push(cs[i]);
                i += 1;
            }
        }
        out
    }

    fn replace_multidots(&self, s: &str) -> String {
        if !s.contains("..") {
            return s.to_string();
        }
        let cs: Vec<char> = s.chars().collect();
        let n = cs.len();
        let mut out = String::with_capacity(s.len() + 16);
        let mut i = 0;
        while i < n {
            if cs[i] == '.' {
                let mut j = i;
                while j < n && cs[j] == '.' {
                    j += 1;
                }
                let k = j - i;
                if k >= 2 {
                    out.push(' ');
                    for _ in 0..k {
                        out.push_str("DOT");
                    }
                    out.push_str("MULTI");
                    if j < n {
                        out.push(' ');
                    }
                } else {
                    out.push('.');
                }
                i = j;
            } else {
                out.push(cs[i]);
                i += 1;
            }
        }
        out
    }

    fn restore_multidots(&self, s: &str) -> String {
        if !s.contains("MULTI") {
            return s.to_string();
        }
        self.multidot_restore
            .replace_all(s, |caps: &regex::Captures| {
                let m = caps.get(0).unwrap().as_str();
                ".".repeat((m.len() - 5) / 3)
            })
            .into_owned()
    }

    fn comma_separate(&self, s: String) -> String {
        if !s.contains(',') {
            return s;
        }
        // COMMA_SEPARATE_1: ([^N])[,] -> "\1 , "
        let s1 = {
            let cs: Vec<char> = s.chars().collect();
            let n = cs.len();
            let mut out = String::with_capacity(s.len() + 16);
            let mut i = 0;
            while i < n {
                if i + 1 < n && cs[i + 1] == ',' && !self.isn.has(cs[i]) {
                    out.push(cs[i]);
                    out.push_str(" , ");
                    i += 2;
                } else {
                    out.push(cs[i]);
                    i += 1;
                }
            }
            out
        };
        // COMMA_SEPARATE_2: [,]([^N]) -> " , \1"
        let s2 = {
            let cs: Vec<char> = s1.chars().collect();
            let n = cs.len();
            let mut out = String::with_capacity(s1.len() + 16);
            let mut i = 0;
            while i < n {
                if cs[i] == ',' && i + 1 < n && !self.isn.has(cs[i + 1]) {
                    out.push_str(" , ");
                    out.push(cs[i + 1]);
                    i += 2;
                } else {
                    out.push(cs[i]);
                    i += 1;
                }
            }
            out
        };
        // COMMA_SEPARATE_3: ([N])[,]$ -> "\1 , "
        let mut chars_rev = s2.chars().rev();
        if chars_rev.next() == Some(',') {
            if let Some(prev) = chars_rev.next() {
                if self.isn.has(prev) {
                    let mut out = s2[..s2.len() - 1].to_string();
                    out.push_str(" , ");
                    return out;
                }
            }
        }
        s2
    }

    fn apostrophes(&self, s: String) -> String {
        let a = &self.alpha;
        let n = &self.isn;
        match self.lang.as_str() {
            "en" => {
                let s = sub3(&s, '\'', |x, y| !a.has(x) && !a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                let s = sub3(&s, '\'', |x, y| !a.has(x) && !n.has(x) && a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                let s = sub3(&s, '\'', |x, y| a.has(x) && !a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                let s = sub3(&s, '\'', |x, y| a.has(x) && a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" '");
                    o.push(y);
                })
                .into_owned();
                sub3(&s, '\'', |x, y| n.has(x) && y == 's', |x, y, o| {
                    o.push(x);
                    o.push_str(" '");
                    o.push(y);
                })
                .into_owned()
            }
            "fr" | "it" => {
                let s = sub3(&s, '\'', |x, y| !a.has(x) && !a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                let s = sub3(&s, '\'', |x, y| !a.has(x) && a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                let s = sub3(&s, '\'', |x, y| a.has(x) && !a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str(" ' ");
                    o.push(y);
                })
                .into_owned();
                sub3(&s, '\'', |x, y| a.has(x) && a.has(y), |x, y, o| {
                    o.push(x);
                    o.push_str("' ");
                    o.push(y);
                })
                .into_owned()
            }
            _ => s.replace('\'', " ' "),
        }
    }

    fn handle_nonbreaking_prefixes(&self, text: &str) -> String {
        let tokens = py_split(text);
        let n = tokens.len();
        let mut out = String::with_capacity(text.len() + 8);
        for (i, tok) in tokens.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            if tok.len() > 1 && tok.ends_with('.') {
                let prefix = &tok[..tok.len() - 1];
                let keep_a = prefix.contains('.') && prefix.chars().any(|c| self.alpha.has(c));
                let keep_b = self.nbp.contains(prefix) && !self.numeric_only.contains(prefix);
                let keep_c = i != n - 1
                    && tokens[i + 1]
                        .chars()
                        .next()
                        .map(|c| self.lower.has(c))
                        .unwrap_or(false);
                if keep_a || keep_b || keep_c {
                    out.push_str(tok);
                } else if self.numeric_only.contains(prefix)
                    && i + 1 < n
                    && tokens[i + 1].starts_with(|c: char| c.is_ascii_digit())
                {
                    out.push_str(tok);
                } else {
                    out.push_str(prefix);
                    out.push_str(" .");
                }
            } else {
                out.push_str(tok);
            }
        }
        out
    }

    fn trailing_dot_apostrophe(&self, s: String) -> String {
        if let Some(stem) = s.strip_suffix(".'") {
            return format!("{} . ' ", stem);
        }
        if let Some(stem) = s.strip_suffix(".' ") {
            return format!("{} . ' ", stem);
        }
        s
    }

    fn escape_xml(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 16);
        for c in s.chars() {
            match c {
                '&' => out.push_str("&amp;"),
                '|' => out.push_str("&#124;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '\'' => out.push_str("&apos;"),
                '"' => out.push_str("&quot;"),
                '[' => out.push_str("&#91;"),
                ']' => out.push_str("&#93;"),
                _ => out.push(c),
            }
        }
        out
    }

    fn run(&self, text: &str, aggressive: bool, escape: bool) -> String {
        let text = collapse_ws_and_junk(text);
        let text = py_strip(&text).to_string();
        let mut text = self.pad_not_alnum(&text);
        if aggressive {
            text = self.aggressive_hyphen_split(&text);
        }
        let text = self.replace_multidots(&text);
        let text = self.comma_separate(text);
        let text = self.apostrophes(text);
        let text = self.handle_nonbreaking_prefixes(&text);
        let text = collapse_ws(&text);
        let text = py_strip(&text).to_string();
        let text = self.trailing_dot_apostrophe(text);
        let text = self.restore_multidots(&text);
        if escape {
            self.escape_xml(&text)
        } else {
            text
        }
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(
        lang: String,
        is_alnum: &str,
        is_alpha: &str,
        is_n: &str,
        is_lower: &str,
        nonbreaking_prefixes: Vec<String>,
        numeric_only_prefixes: Vec<String>,
    ) -> PyResult<Self> {
        let multidot_restore = Regex::new(r"(?:DOT)+MULTI")
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Tokenizer {
            lang,
            alnum: CharSet::new(is_alnum),
            alpha: CharSet::new(is_alpha),
            isn: CharSet::new(is_n),
            lower: CharSet::new(is_lower),
            nbp: nonbreaking_prefixes.into_iter().collect(),
            numeric_only: numeric_only_prefixes.into_iter().collect(),
            multidot_restore,
        })
    }

    fn tokenize(&self, text: &str, aggressive_dash_splits: bool, escape: bool) -> String {
        self.run(text, aggressive_dash_splits, escape)
    }

    fn tokenize_split(&self, text: &str, aggressive_dash_splits: bool, escape: bool) -> Vec<String> {
        let s = self.run(text, aggressive_dash_splits, escape);
        py_split(&s).into_iter().map(|t| t.to_string()).collect()
    }

    fn tokenize_many(
        &self,
        texts: Vec<String>,
        aggressive_dash_splits: bool,
        escape: bool,
    ) -> Vec<String> {
        texts
            .iter()
            .map(|t| self.run(t, aggressive_dash_splits, escape))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Punctuation normalizer
// ---------------------------------------------------------------------------

/// Convert a Python `re.sub` replacement template to the `regex` crate syntax.
fn convert_replacement(py: &str) -> Result<String, String> {
    let cs: Vec<char> = py.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c == '$' {
            out.push_str("$$");
            i += 1;
        } else if c == '\\' {
            i += 1;
            if i >= cs.len() {
                return Err("bad escape at end of replacement".into());
            }
            if cs[i] == 'g' {
                // \g<N>
                if i + 1 < cs.len() && cs[i + 1] == '<' {
                    let mut j = i + 2;
                    let mut num = String::new();
                    while j < cs.len() && cs[j] != '>' {
                        num.push(cs[j]);
                        j += 1;
                    }
                    if j >= cs.len() || num.is_empty() || !num.chars().all(|d| d.is_ascii_digit()) {
                        return Err("unsupported group reference".into());
                    }
                    out.push_str(&format!("${{{}}}", num));
                    i = j + 1;
                } else {
                    return Err("bad \\g".into());
                }
            } else if cs[i].is_ascii_digit() {
                out.push_str(&format!("${{{}}}", cs[i]));
                i += 1;
            } else if cs[i] == '\\' {
                out.push('\\');
                i += 1;
            } else {
                return Err(format!("unsupported escape \\{}", cs[i]));
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

/// Python's `\s` also matches U+001C..U+001F, the `regex` crate's does not.
fn convert_pattern(py: &str) -> String {
    py.replace("\\s", "[\\s\\x1c-\\x1f]")
}

#[pyclass]
struct Normalizer {
    pre: Vec<(Regex, String)>,
    subs: Vec<(Regex, String)>,
    use_pre: bool,
    post_ctrl: Option<Regex>,
}

/// One `re.sub` step that only allocates when something matched.
fn apply(re: &Regex, rep: &str, cur: String) -> String {
    let changed = match re.replace_all(&cur, rep) {
        Cow::Owned(s) => Some(s),
        Cow::Borrowed(_) => None,
    };
    match changed {
        Some(s) => s,
        None => cur,
    }
}

fn compile_pairs(pairs: Vec<(String, String)>) -> PyResult<Vec<(Regex, String)>> {
    let mut out = Vec::with_capacity(pairs.len());
    for (pat, rep) in pairs {
        let re = Regex::new(&convert_pattern(&pat))
            .map_err(|e| PyValueError::new_err(format!("pattern {:?}: {}", pat, e)))?;
        let rep = convert_replacement(&rep).map_err(PyValueError::new_err)?;
        out.push((re, rep));
    }
    Ok(out)
}

#[pymethods]
impl Normalizer {
    #[new]
    fn new(
        substitutions: Vec<(String, String)>,
        replace_unicode_punct: Vec<(String, String)>,
        pre_replace_unicode_punct: bool,
        post_remove_control_chars: bool,
    ) -> PyResult<Self> {
        let post_ctrl = if post_remove_control_chars {
            Some(Regex::new(r"\p{C}").map_err(|e| PyValueError::new_err(e.to_string()))?)
        } else {
            None
        };
        Ok(Normalizer {
            pre: compile_pairs(replace_unicode_punct)?,
            subs: compile_pairs(substitutions)?,
            use_pre: pre_replace_unicode_punct,
            post_ctrl,
        })
    }

    fn normalize(&self, text: &str) -> String {
        let mut cur = text.to_string();
        if self.use_pre {
            for (re, rep) in &self.pre {
                cur = apply(re, rep, cur);
            }
        }
        for (re, rep) in &self.subs {
            cur = apply(re, rep, cur);
        }
        if let Some(re) = &self.post_ctrl {
            cur = apply(re, "", cur);
        }
        py_strip(&cur).to_string()
    }

    fn normalize_many(&self, texts: Vec<String>) -> Vec<String> {
        texts.iter().map(|t| self.normalize(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// Detokenizer
// ---------------------------------------------------------------------------

#[pyclass]
struct Detokenizer {
    lang: String,
    alpha: CharSet,
    sc: CharSet,
    cjk: Vec<(u32, u32)>,
    fi_re: Regex,
}

impl Detokenizer {
    fn is_cjk(&self, c: char) -> bool {
        let u = c as u32;
        for &(start, end) in &self.cjk {
            if u <= end {
                return u >= start;
            }
        }
        false
    }

    fn is_currency(&self, tok: &str) -> bool {
        !tok.is_empty()
            && tok
                .chars()
                .all(|c| self.sc.has(c) || matches!(c, '(' | '[' | '{' | '¿' | '¡'))
    }

    fn is_punct(tok: &str) -> bool {
        !tok.is_empty()
            && tok.chars().all(|c| {
                matches!(c, ',' | '.' | '?' | '!' | ':' | ';' | '\\' | '%' | '}' | ']' | ')')
            })
    }

    fn is_open_quote(tok: &str) -> bool {
        !tok.is_empty() && tok.chars().all(|c| matches!(c, '\'' | '"' | '„' | '“' | '`'))
    }

    fn is_english_contraction(&self, tok: &str) -> bool {
        let mut it = tok.chars();
        it.next() == Some('\'') && it.next().map(|c| self.alpha.has(c)).unwrap_or(false)
    }

    fn is_french_contraction(&self, tok: &str) -> bool {
        let mut it = tok.chars().rev();
        it.next() == Some('\'') && it.next().map(|c| self.alpha.has(c)).unwrap_or(false)
    }

    fn starts_with_alpha(&self, tok: &str) -> bool {
        tok.chars().next().map(|c| self.alpha.has(c)).unwrap_or(false)
    }

    fn all_ascii_digits(tok: &str) -> bool {
        !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit())
    }

    /// Returns `None` when the input hits a path the original implementation
    /// handles in a way we deliberately do not reproduce (caller falls back).
    fn run(&self, tokens: &[String], unescape: bool) -> Option<String> {
        let mut text = String::with_capacity(tokens.iter().map(|t| t.len() + 1).sum::<usize>() + 2);
        text.push(' ');
        for (i, t) in tokens.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            text.push_str(t);
        }
        text.push(' ');
        let mut text = if text.contains(" @-@ ") {
            text.replace(" @-@ ", "-")
        } else {
            text
        };
        if unescape && text.contains('&') {
            for (from, to) in [
                ("&bar;", "|"),
                ("&#124;", "|"),
                ("&lt;", "<"),
                ("&gt;", ">"),
                ("&bra;", "["),
                ("&ket;", "]"),
                ("&quot;", "\""),
                ("&apos;", "'"),
                ("&#91;", "["),
                ("&#93;", "]"),
                ("&amp;", "&"),
            ] {
                if text.contains(from) {
                    text = text.replace(from, to);
                }
            }
        }

        let mut quote_counts: HashMap<String, i64> = HashMap::new();
        for k in ["'", "\"", "``", "`", "''"] {
            quote_counts.insert(k.to_string(), 0);
        }

        let toks = py_split(&text);
        let n = toks.len();
        let mut prepend_space = " ";
        let mut det = String::with_capacity(text.len() + 8);
        let lang = self.lang.as_str();

        for (i, token) in toks.iter().enumerate() {
            let token: &str = token;
            let first = token.chars().next().unwrap();
            if self.is_cjk(first) && lang != "ko" {
                let prev_last_cjk = i > 0 && self.is_cjk(toks[i - 1].chars().last().unwrap());
                if prev_last_cjk {
                    det.push_str(token);
                } else {
                    det.push_str(prepend_space);
                    det.push_str(token);
                }
                prepend_space = " ";
            } else if self.is_currency(token) {
                det.push_str(prepend_space);
                det.push_str(token);
                prepend_space = "";
            } else if Self::is_punct(token) {
                if lang == "fr"
                    && token.chars().count() == 1
                    && matches!(first, '?' | '!' | ':' | ';' | '\\' | '%')
                {
                    det.push(' ');
                }
                det.push_str(token);
                prepend_space = " ";
            } else if lang == "en" && i > 0 && self.is_english_contraction(token) {
                det.push_str(token);
                prepend_space = " ";
            } else if lang == "cs"
                && i > 1
                && Self::all_ascii_digits(toks[n - 2])
                && (toks[n - 1] == "." || toks[n - 1] == ",")
                && Self::all_ascii_digits(token)
            {
                det.push_str(token);
                prepend_space = " ";
            } else if matches!(lang, "fr" | "it" | "ga")
                && i + 2 <= n
                && self.is_french_contraction(token)
                && self.starts_with_alpha(toks[i + 1])
            {
                det.push_str(prepend_space);
                det.push_str(token);
                prepend_space = "";
            } else if lang == "cs"
                && i + 3 <= n
                && self.is_french_contraction(token)
                && (toks[i + 1] == "-" || toks[i + 1] == "–")
            {
                // The original may raise TypeError (`next(list)`) on this path
                // depending on the third token; let it decide.
                return None;
            } else if Self::is_open_quote(token) {
                let mut normalized = token;
                if token.chars().all(|c| matches!(c, '„' | '“' | '”')) {
                    normalized = "\"";
                }
                quote_counts.entry(normalized.to_string()).or_insert(0);
                if lang == "cs" && token == "„" {
                    quote_counts.insert(normalized.to_string(), 0);
                }
                if lang == "cs" && token == "“" {
                    quote_counts.insert(normalized.to_string(), 1);
                }
                let count = *quote_counts.get(normalized).unwrap();
                if count % 2 == 0 {
                    if lang == "en" && token == "'" && i > 0 && toks[i - 1].ends_with('s') {
                        det.push_str(token);
                        prepend_space = " ";
                    } else {
                        det.push_str(prepend_space);
                        det.push_str(token);
                        prepend_space = "";
                        *quote_counts.get_mut(normalized).unwrap() += 1;
                    }
                } else {
                    det.push_str(token);
                    prepend_space = " ";
                    *quote_counts.get_mut(normalized).unwrap() += 1;
                }
            } else if lang == "fi"
                && toks[(i + n - 1) % n].ends_with(':')
                && self.fi_re.is_match(token)
            {
                det.push_str(prepend_space);
                det.push_str(token);
                prepend_space = " ";
            } else {
                det.push_str(prepend_space);
                det.push_str(token);
                prepend_space = " ";
            }
        }

        // ONE_SPACE: " {2,}" -> " ", then strip.
        let mut out = String::with_capacity(det.len());
        let mut run = 0usize;
        for c in det.chars() {
            if c == ' ' {
                run += 1;
                if run == 1 {
                    out.push(' ');
                }
            } else {
                run = 0;
                out.push(c);
            }
        }
        Some(py_strip(&out).to_string())
    }
}

#[pymethods]
impl Detokenizer {
    #[new]
    fn new(
        lang: String,
        is_alpha: &str,
        is_sc: &str,
        cjk_ranges: Vec<(u32, u32)>,
        finnish_pattern: &str,
    ) -> PyResult<Self> {
        let fi_re = Regex::new(finnish_pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Detokenizer {
            lang,
            alpha: CharSet::new(is_alpha),
            sc: CharSet::new(is_sc),
            cjk: cjk_ranges,
            fi_re,
        })
    }

    /// Returns None when the caller must fall back to the Python implementation.
    fn detokenize(&self, tokens: Vec<String>, unescape: bool) -> Option<String> {
        self.run(&tokens, unescape)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<Normalizer>()?;
    m.add_class::<Detokenizer>()?;
    Ok(())
}
