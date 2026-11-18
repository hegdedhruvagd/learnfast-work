import math
from pathlib import Path
import unittest

from next_word_predictor import (
    END,
    NgramLM,
    UNK,
    count_ngrams,
    load_gutenberg_text,
    split_sentences,
    to_sentences,
    tokenize,
)


HERE = Path(__file__).resolve().parents[1]


class TextProcessingTests(unittest.TestCase):
    def test_course_corpus_has_exactly_60_sentences(self):
        text = (HERE / "course_corpus.txt").read_text(encoding="utf-8")
        self.assertEqual(len(to_sentences(text)), 60)

    def test_tokenization_matches_course_rule(self):
        self.assertEqual(tokenize("I'm READY! It's 2026."), ["i'm", "ready", "it's"])
        self.assertEqual(tokenize("..."), [])
        self.assertEqual(to_sentences("  One! ? Two. "), ["One", "Two"])

    def test_split_is_repeatable_and_disjoint(self):
        items = [f"sentence {i}" for i in range(10)]
        train, test = split_sentences(items, seed=7)
        self.assertEqual((train, test), split_sentences(items, seed=7))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train) + len(test), len(items))

    def test_count_ngrams_adds_sentence_boundaries(self):
        counts = count_ngrams(["a b"], 2)
        self.assertEqual(counts[("<s>",)]["a"], 1)
        self.assertEqual(counts[("a",)]["b"], 1)
        self.assertEqual(counts[("b",)][END], 1)
        self.assertEqual(count_ngrams([""], 2)[("<s>",)][END], 1)


class NgramModelTests(unittest.TestCase):
    def test_mle_suggestions_and_distribution(self):
        model = NgramLM(n=2).fit("a b. a c.")
        self.assertEqual(model.suggest_next("a", topk=2), [("b", 0.5), ("c", 0.5)])
        self.assertAlmostEqual(sum(model.probabilities("a").values()), 1.0)
        self.assertEqual(model.suggest_next("unseen"), [])

    def test_add_k_assigns_probability_to_unseen_events(self):
        model = NgramLM(n=2, k=0.5).fit("a b.")
        distribution = model.probabilities("a")
        self.assertGreater(distribution[UNK], 0.0)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(model.probabilities("unseen")["b"], 0.0)

    def test_very_large_smoothing_stays_normalized(self):
        model = NgramLM(n=2).fit("a b.")
        distribution = model.probabilities("a", k=1e308)
        self.assertTrue(all(math.isfinite(value) for value in distribution.values()))
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_oov_maps_to_unknown_and_perplexity_is_finite_with_smoothing(self):
        model = NgramLM(n=2, k=0.2).fit("a b. a c.")
        self.assertTrue(math.isfinite(model.perplexity("new word.")))
        self.assertIn(UNK, model.vocabulary)

    def test_cross_entropy_and_perplexity_edge_cases(self):
        model = NgramLM(n=2, k=0.5).fit("a b.")
        self.assertAlmostEqual(model.perplexity("a b."), math.exp(model.cross_entropy("a b.")))
        self.assertTrue(math.isnan(model.perplexity("")))
        mle = NgramLM(n=2).fit("a b.")
        self.assertTrue(math.isinf(mle.perplexity("a c.")))

    def test_empty_fit_and_generation(self):
        model = NgramLM(n=3, k=0.5).fit("")
        self.assertEqual(model.suggest_next(""), [])
        self.assertEqual(model.generate(seed=3), "")
        self.assertTrue(math.isnan(model.perplexity("")))

    def test_generation_is_seeded_and_respects_limit(self):
        model = NgramLM(n=2, k=0.2).fit("a b c. a b d. a b e.")
        self.assertEqual(model.generate(seed=12), model.generate(seed=12))
        self.assertLessEqual(len(tokenize(model.generate(seed=12, max_tokens=2))), 2)
        self.assertEqual(model.generate(max_tokens=0), "")

    def test_models_must_be_fitted(self):
        with self.assertRaises(RuntimeError):
            NgramLM().suggest_next("hello")

    def test_argument_validation(self):
        for n in (1, 2.5, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                NgramLM(n=n)
        for k in (-0.1, float("nan"), float("inf")):
            with self.subTest(k=k), self.assertRaises(ValueError):
                NgramLM(k=k)
        model = NgramLM().fit("a b.")
        with self.assertRaises(ValueError):
            model.suggest_next("a", topk=-1)
        with self.assertRaises(ValueError):
            model.generate(max_tokens=-1)
        with self.assertRaises(ValueError):
            model.perplexity("a.", k=-1)

    def test_gutenberg_loader_requires_trusted_https_host(self):
        with self.assertRaises(ValueError):
            load_gutenberg_text("http://www.gutenberg.org/book.txt")
        with self.assertRaises(ValueError):
            load_gutenberg_text("https://example.com/book.txt")


if __name__ == "__main__":
    unittest.main()
