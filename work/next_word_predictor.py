"""A small, dependency-free n-gram language model.

Follows askThili's "Language Modeling from First Principles": tokenize, count
n-grams, estimate probabilities, predict, generate, smooth, and evaluate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import re
from typing import Iterable, Sequence
from urllib.request import Request, urlopen

START = "<s>"
END = "</s>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize like the course's ``[a-z']+`` rule."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.findall(r"[a-z']+", text.lower())


def to_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation and drop empty fragments."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]


def split_sentences(
    sentences: Sequence[str], test_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Shuffle and split before fitting counts or vocabulary."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    items = list(sentences)
    if len(items) < 2:
        raise ValueError("at least two sentences are needed for a split")
    shuffled = items.copy()
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, min(len(items) - 1, round(len(items) * test_fraction)))
    return shuffled[test_size:], shuffled[:test_size]


def count_ngrams(sentences: Iterable[str], n: int) -> dict[tuple[str, ...], Counter[str]]:
    """Map each n-1 token context to counts of its next token."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")
    followers: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for sentence in sentences:
        padded = [START] * (n - 1) + tokenize(sentence) + [END]
        for index in range(len(padded) - n + 1):
            context = tuple(padded[index : index + n - 1])
            followers[context][padded[index + n - 1]] += 1
    return dict(followers)


class NgramLM:
    """N-gram language model with MLE or add-k estimates.

    Vocabulary comes only from fit text. Out-of-vocabulary words map to
    ``<unk>`` during scoring and prediction. Sentence boundaries are modeled.
    """

    def __init__(self, n: int = 2, k: float = 0.0) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError("n must be an integer >= 2")
        self.n = n
        self.k = self._validate_k(k)
        self.vocabulary: set[str] = {END, UNK}
        self.followers: dict[tuple[str, ...], Counter[str]] = {}
        self.fitted = False

    @staticmethod
    def _validate_k(k: float) -> float:
        value = float(k)
        if not math.isfinite(value) or value < 0:
            raise ValueError("k must be a finite, non-negative number")
        return value

    def fit(self, text: str | Iterable[str]) -> NgramLM:
        """Fit on raw text or on an iterable of sentence strings."""
        sentences = to_sentences(text) if isinstance(text, str) else list(text)
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise TypeError("all sentences must be strings")
        self.vocabulary = {END, UNK}
        for sentence in sentences:
            self.vocabulary.update(tokenize(sentence))
        self.followers = count_ngrams(sentences, self.n)
        self.fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit the model before prediction or scoring")

    def _map_word(self, word: str) -> str:
        normalized = word.lower()
        return normalized if normalized in self.vocabulary else UNK

    def _normalize_context(self, context: str | Sequence[str]) -> tuple[str, ...]:
        raw = tokenize(context) if isinstance(context, str) else [str(w).lower() for w in context]
        mapped = [self._map_word(word) for word in raw]
        padded = [START] * max(0, self.n - 1 - len(mapped)) + mapped
        return tuple(padded[-(self.n - 1) :])

    def probabilities(
        self, context: str | Sequence[str], k: float | None = None
    ) -> dict[str, float]:
        """Return probabilities over the fixed training vocabulary."""
        self._check_fitted()
        smoothing = self.k if k is None else self._validate_k(k)
        ctx = self._normalize_context(context)
        counts = self.followers.get(ctx, Counter())
        total = sum(counts.values())
        if total == 0 and smoothing == 0:
            return {}
        if smoothing == 0:
            return {
                word: counts.get(word, 0) / total
                for word in sorted(self.vocabulary)
            }
        scale = max(float(total), smoothing)
        scaled_k = smoothing / scale
        scaled_denominator = total / scale + scaled_k * len(self.vocabulary)
        return {
            word: (counts.get(word, 0) / scale + scaled_k) / scaled_denominator
            for word in sorted(self.vocabulary)
        }

    def suggest_next(
        self, context: str | Sequence[str], topk: int = 3
    ) -> list[tuple[str, float]]:
        """Rank likely next words; hide end-of-sentence and unknown markers."""
        self._check_fitted()
        if not isinstance(topk, int) or isinstance(topk, bool) or topk < 0:
            raise ValueError("topk must be a non-negative integer")
        ranked = [
            (word, probability)
            for word, probability in self.probabilities(context).items()
            if word not in {END, UNK}
        ]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:topk]

    def generate(self, seed: int = 0, max_tokens: int = 25, prompt: str = "") -> str:
        """Sample words one at a time until EOS or the token limit."""
        self._check_fitted()
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        rng = random.Random(seed)
        output = tokenize(prompt)
        context = self._normalize_context(output)
        for _ in range(max_tokens):
            distribution = self.probabilities(context)
            if not distribution:
                break
            words = list(distribution)
            weights = [distribution[word] for word in words]
            if not any(weight > 0 for weight in weights):
                break
            next_word = rng.choices(words, weights=weights, k=1)[0]
            if next_word == END:
                break
            if next_word != UNK:
                output.append(next_word)
            context = self._normalize_context(output)
        return " ".join(output)

    def cross_entropy(self, text: str, k: float | None = None) -> float:
        """Mean natural-log loss per predicted token, including EOS."""
        self._check_fitted()
        smoothing = self.k if k is None else self._validate_k(k)
        log_sum = 0.0
        event_count = 0
        for sentence in to_sentences(text):
            words = [self._map_word(word) for word in tokenize(sentence)]
            padded = [START] * (self.n - 1) + words + [END]
            for index in range(len(padded) - self.n + 1):
                context = tuple(padded[index : index + self.n - 1])
                target = padded[index + self.n - 1]
                counts = self.followers.get(context, Counter())
                total = sum(counts.values())
                if smoothing == 0:
                    probability = counts.get(target, 0) / total if total else 0.0
                elif total or smoothing:
                    scale = max(float(total), smoothing)
                    scaled_k = smoothing / scale
                    scaled_denominator = total / scale + scaled_k * len(self.vocabulary)
                    probability = (counts.get(target, 0) / scale + scaled_k) / scaled_denominator
                else:
                    probability = 0.0
                if probability <= 0:
                    return math.inf
                log_sum += math.log(probability)
                event_count += 1
        return -log_sum / event_count if event_count else math.nan

    def perplexity(self, text: str, k: float | None = None) -> float:
        """Exponentiated cross-entropy; infinity signals an MLE zero event."""
        entropy = self.cross_entropy(text, k=k)
        if math.isnan(entropy):
            return math.nan
        try:
            return math.exp(entropy)
        except OverflowError:
            return math.inf


def load_gutenberg_text(url: str, timeout: float = 15.0) -> str:
    """Fetch a caller-supplied Gutenberg plain-text URL into memory.

    A URL is required because the Gutenberg ebook page was unavailable when
    this project was prepared. No downloaded file is cached or redistributed.
    """
    if not isinstance(url, str) or not url.startswith("https://www.gutenberg.org/"):
        raise ValueError("provide an HTTPS URL hosted on www.gutenberg.org")
    request = Request(url, headers={"User-Agent": "TinyVerseNgramLM/1.0 (educational use)"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/plain", "application/octet-stream"}:
            raise ValueError(f"expected a plain-text response, got {content_type!r}")
        return response.read().decode("utf-8-sig")
