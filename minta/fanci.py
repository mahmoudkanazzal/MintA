"""FANCI domain-name features (45-dimensional vector), after Schueppen et al.,
"FANCI: Feature-based Automated NXDomain Classification and Intelligence", USENIX
Security 2018.

* `has_valid_tld` is 1 for every domain with at least two labels.
* A feature that raises ValueError or ArithmeticError (for example, a ratio over an
  empty suffix-free string) is set to 0.

Layout (index: feature): 0 length; 1-4 parts one-hot; 5 vowel ratio; 6 digit ratio;
7 contains IPv4; 8 contains digits; 9 has valid TLD; 10 contains one-char subdomain;
11 contains 'www.'; 12 subdomain length mean; 13 exclusive prefix repetition;
14 char diversity; 15 contains only-digit subdomain; 16 TLD as infix;
17-37 n-gram statistics (1-, 2-, 3-grams x 7 stats); 38 hex part ratio;
39 underscore ratio; 40 alphabet size; 41 Shannon entropy; 42 repeated-char ratio;
43 consecutive consonant ratio; 44 consecutive digit ratio.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

N_FEATURES = 45
LENGTH = 0
CONTAINS_DIGITS = 8
ONE_CHAR_SUBDOMAIN = 10
WWW_PREFIX = 11
PREFIX_REPETITION = 13

HEX_DIGITS = set("0123456789abcdef")
VOWELS = set("aeiou")
PARTS_MAX_CONSIDERED = 4
PUBLIC_SUFFIX_FILE = Path(__file__).resolve().parent / "resources" / "public_suffix.txt"
IPV4_PATTERN = re.compile(
    r"(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])"
)


def _clean(lines: list[str]) -> list[str]:
    out = [d.strip().lower() for d in lines]
    return [d for d in out if d]


@lru_cache(maxsize=1)
def _suffix_tables() -> tuple[frozenset[str], frozenset[str]]:
    with open(PUBLIC_SUFFIX_FILE, encoding="utf-8") as f:
        data = f.readlines()
    data = _clean(data)
    data = ["." + s for s in data if not (s.startswith("/") or s.startswith("*"))]
    data = _clean(data)
    suffixes = frozenset(data)
    tld_tails = frozenset(s[1:] for s in data if len(s.split(".")) == 2)
    return suffixes, tld_tails


@lru_cache(maxsize=3)
def _analyzer(n: int) -> Callable[[str], list[str]]:
    return CountVectorizer(analyzer="char", ngram_range=(n, n)).build_analyzer()


class _Parsed:
    __slots__ = ("domain", "dot_split", "suffix_free", "joined_suffix_free", "public_suffix", "unigram")

    def __init__(self, domain: str):
        suffixes, _ = _suffix_tables()
        self.domain = domain
        self.dot_split = tuple(domain.split("."))
        match = ""
        cleared = self.dot_split
        if len(self.dot_split) >= 2:
            for i in range(len(self.dot_split)):
                match = "." + ".".join(self.dot_split[i:])
                if match in suffixes:
                    cleared = self.dot_split[0:i]
                    break
        self.suffix_free = tuple(cleared)
        self.public_suffix = match
        self.joined_suffix_free = "".join(self.suffix_free)
        self.unigram: np.ndarray | None = None


def _length(p: _Parsed) -> list[float]:
    return [len(p.domain)]


def _parts(p: _Parsed) -> list[float]:
    feature = [0] * PARTS_MAX_CONSIDERED
    n = len(p.suffix_free)
    if n >= PARTS_MAX_CONSIDERED:
        feature[PARTS_MAX_CONSIDERED - 1] = 1
    else:
        feature[n - 1] = 1
    return list(map(float, feature))


def _vowel_ratio(p: _Parsed) -> list[float]:
    vowels = sum(1 for c in p.joined_suffix_free if c in VOWELS)
    alpha = sum(1 for c in p.joined_suffix_free if c.isalpha())
    return [vowels / alpha] if alpha > 0 else [0]


def _digit_ratio(p: _Parsed) -> list[float]:
    d = p.joined_suffix_free
    return [sum(1 for c in d if c.isdigit()) / len(d)]


def _contains_ipv4(p: _Parsed) -> list[float]:
    return [1] if re.search(IPV4_PATTERN, p.domain) else [0]


def _contains_digits(p: _Parsed) -> list[float]:
    return [1] if any(c.isdigit() for c in p.domain) else [0]


def _has_valid_tld(p: _Parsed) -> list[float]:
    return [1] if p.public_suffix else [0]


def _contains_one_char_subdomains(p: _Parsed) -> list[float]:
    parts = p.dot_split
    if len(parts) > 2:
        parts = parts[:-1]
    return [1] if any(len(x) == 1 for x in parts) else [0]


def _contains_wwwdot(p: _Parsed) -> list[float]:
    return [1] if "www." in p.domain else [0]


def _subdomain_lengths_mean(p: _Parsed) -> list[float]:
    return [sum(len(x) for x in p.suffix_free) / len(p.suffix_free)]


def _prefix_repetition(p: _Parsed) -> list[float]:
    d = p.domain
    return [0] if (d + d).find(d, 1, -1) == -1 else [1]


def _char_diversity(p: _Parsed) -> list[float]:
    d = p.joined_suffix_free
    return [len(set(d)) / len(d)]


def _contains_subdomain_of_only_digits(p: _Parsed) -> list[float]:
    for part in p.dot_split:
        if not any(c.isalpha() for c in part):
            return [1]
    return [0]


def _contains_tld_as_infix(p: _Parsed) -> list[float]:
    _, tld_tails = _suffix_tables()
    return [1] if any(part in tld_tails for part in p.suffix_free) else [0]


def _ngram_stats(npa: np.ndarray) -> list[float]:
    if npa.size > 0:
        return [npa.std(), np.median(npa), npa.mean(), np.min(npa), np.max(npa),
                np.percentile(npa, 25), np.percentile(npa, 75)]
    return [-1, -1, -1, -1, -1, -1, -1]


def _n_grams(p: _Parsed) -> list[float]:
    feature: list[float] = []
    for n in range(1, 4):
        counts = _analyzer(n)(p.joined_suffix_free)
        npa = np.array(list(Counter(counts).values()), dtype=int)
        if n == 1:
            p.unigram = npa
        feature += _ngram_stats(npa)
    return feature


def _hex_part_ratio(p: _Parsed) -> list[float]:
    hex_parts = sum(1 for x in p.suffix_free if all(c in HEX_DIGITS for c in x))
    return [hex_parts / len(p.suffix_free)]


def _underscore_ratio(p: _Parsed) -> list[float]:
    d = p.joined_suffix_free
    return [d.count("_") / len(d)]


def _unigram(p: _Parsed) -> np.ndarray:
    if p.unigram is None:
        raise RuntimeError("_n_grams must run before unigram-based features")
    return p.unigram


def _alphabet_size(p: _Parsed) -> list[float]:
    return [len(_unigram(p))]


def _shannon_entropy(p: _Parsed) -> list[float]:
    return [stats.entropy(_unigram(p), base=2)]


def _ratio_of_repeated_chars(p: _Parsed) -> list[float]:
    u = _unigram(p)
    return [sum(1 for i in u if i > 1) / len(u)]


def _consecutive_ratio(p: _Parsed, predicate: Callable[[str], bool]) -> list[float]:
    total = 0
    for part in p.suffix_free:
        counter = 0
        for i, c in enumerate(part, start=1):
            if predicate(c):
                counter += 1
            else:
                if counter > 1:
                    total += counter
                counter = 0
            if i == len(part) and counter > 1:
                total += counter
    return [total / len(p.joined_suffix_free)]


def _consecutive_consonant_ratio(p: _Parsed) -> list[float]:
    return _consecutive_ratio(p, lambda c: c.isalpha() and c not in VOWELS)


def _consecutive_digits_ratio(p: _Parsed) -> list[float]:
    return _consecutive_ratio(p, lambda c: c.isdigit())


ALL_FEATURES: tuple[Callable[[_Parsed], list[float]], ...] = (
    _length, _parts, _vowel_ratio, _digit_ratio, _contains_ipv4, _contains_digits,
    _has_valid_tld, _contains_one_char_subdomains, _contains_wwwdot, _subdomain_lengths_mean,
    _prefix_repetition, _char_diversity, _contains_subdomain_of_only_digits, _contains_tld_as_infix,
    _n_grams, _hex_part_ratio, _underscore_ratio, _alphabet_size, _shannon_entropy,
    _ratio_of_repeated_chars, _consecutive_consonant_ratio, _consecutive_digits_ratio,
)


def fanci45(domain: str) -> np.ndarray:
    """Return the 45-d FANCI vector for one domain name (float64, may contain NaN)."""
    parsed = _Parsed(domain)
    vec: list[float] = []
    for fn in ALL_FEATURES:
        try:
            vec += fn(parsed)
        except (ValueError, ArithmeticError):
            vec += [0]
    if len(vec) != N_FEATURES:
        raise RuntimeError(f"FANCI produced {len(vec)} values for {domain!r}, expected {N_FEATURES}")
    return np.asarray(vec, dtype=np.float64)


def fanci45_batch(domains: list[str]) -> tuple[np.ndarray, int]:
    """Features for many names. Non-finite values are set to 0; returns (matrix, count)."""
    out = np.zeros((len(domains), N_FEATURES), dtype=np.float32)
    bad = 0
    for k, d in enumerate(domains):
        v = fanci45(d)
        finite = np.isfinite(v)
        if not finite.all():
            bad += int((~finite).sum())
            v = np.where(finite, v, 0.0)
        out[k] = v
    return out, bad
