# TinyVerse Next-Word Predictor

This project rebuilds the askThili *Language Modeling from First Principles* course as a small, reproducible language-modeling portfolio project. It preserves the course's original 60-sentence corpus and implements next-word prediction with word counts, bigrams, and trigrams. It needs only the Python standard library: no neural network, GPU, API key, package install, or dataset download.

## Project files

- `next_word_predictor.ipynb` walks through tokenization, n-gram counts, MLE, suggestions, sampling, add-k smoothing, cross-entropy, perplexity, results, and limitations. Its cells have been executed and saved with outputs.
- `next_word_predictor.py` contains the reusable `NgramLM` class and helper functions.
- `course_corpus.txt` is the exact course corpus.
- `tests/test_next_word_predictor.py` contains automated unit tests.
- `DATASETS.md` records dataset provenance, attribution, the Gutenberg status, and copyright notes.

## Setup and usage

Use Python 3.10 or newer. From this `work/` directory, run:

```bash
python -m unittest discover -s tests -v
```

To run the notebook, start Jupyter from this folder using the existing repository workbench:

```bash
docker compose -f ../lfllm/docker-compose.yml up
```

Then open `http://127.0.0.1:8889/?token=learnfast` and open `next_word_predictor.ipynb`. The repository mounts this folder as the notebook workspace. Alternatively, install Jupyter in your own environment if you prefer; the project code itself has no third-party dependencies.

Example from a Python session in `work/`:

```python
from pathlib import Path
from next_word_predictor import NgramLM, to_sentences

sentences = to_sentences(Path("course_corpus.txt").read_text(encoding="utf-8"))
model = NgramLM(n=2, k=0.5).fit(sentences)
print(model.suggest_next("the morning", topk=3))
print(model.generate(seed=3, max_tokens=20))
```

`NgramLM` supports `fit`, `probabilities`, `suggest_next`, `generate`, `cross_entropy`, and `perplexity`. Use `n=2` for a bigram or `n=3` for a trigram. `k=0` is MLE; any non-negative `k` enables add-k smoothing.

## Methods

The tokenizer follows the course rule: lowercase text and extract `[a-z']+`. Each sentence gets `n-1` start markers and an end marker. The count table maps an `n-1` token context to a counter of following tokens.

For context `c` and word `w`, maximum likelihood estimation is:

`P(w | c) = count(c, w) / count(c)`

Add-k smoothing is:

`P(w | c) = (count(c, w) + k) / (count(c) + k × |V|)`

The vocabulary `V` is learned from training sentences only and includes `</s>` and `<unk>`. Words not found in that vocabulary map to `<unk>` when predicting and scoring. For unsmoothed MLE, an unseen event has probability zero.

Cross-entropy and perplexity are computed per predicted token, including the sentence end:

`H = -(1/N) × Σ log(P(wᵢ | contextᵢ))`

`perplexity = exp(H)`

Lower perplexity means the model assigned higher probability to the observed next words. An empty scoring input returns `NaN` because there are no prediction events. If an MLE event has zero probability, perplexity is infinite.

## Course-corpus experiment

The notebook uses seed `42` to shuffle and split the 60 sentences into 48 training and 12 held-out sentences. The split happens before model fitting, so the test vocabulary is not leaked into training. OOV test words become `<unk>`. Test sentences are distinct from training sentences.

The table reports actual results from the executed notebook. Values are perplexity; `∞` means unsmoothed MLE assigned zero probability to at least one held-out event.

| Model | k | Train perplexity | Held-out perplexity |
|---|---:|---:|---:|
| Bigram | 0 (MLE) | 3.54 | ∞ |
| Bigram | 0.1 | 13.61 | 65.28 |
| Bigram | 0.5 | 36.99 | 80.47 |
| Bigram | 1.0 | 55.03 | 92.15 |
| Trigram | 0 (MLE) | 1.88 | ∞ |
| Trigram | 0.1 | 13.83 | 93.48 |
| Trigram | 0.5 | 42.57 | 107.46 |
| Trigram | 1.0 | 63.43 | 116.21 |

On this small split, the trigram fits its training data more closely but has higher smoothed held-out perplexity than the bigram. The two-word context creates sparse counts. Add-k makes held-out scores finite; in these experiments, larger `k` increases held-out perplexity by spreading more mass away from observed continuations. These values are specific to this corpus, tokenizer, split, and implementation; they are not universal model rankings.

Example suggestions and seeded generation are printed in the notebook. They show the interface working, but the generated text is often locally plausible and globally incoherent, as expected for a sparse count model.

## Larger public-domain corpus

The planned comparison is Project Gutenberg's ebook 11, *Alice's Adventures in Wonderland*. The ebook page returned HTTP 503 with a maintenance notice when checked on 2026-11-18, so its text was not downloaded and no larger-corpus results are claimed. `load_gutenberg_text(url)` provides a loader for a verified plain-text URL hosted on `www.gutenberg.org`; it reads text into memory and does not cache it. Check the live download page and terms before running or redistributing the text. Copyright status can vary by jurisdiction, and public-domain status in the United States does not by itself establish worldwide status. See `DATASETS.md` for the source and attribution details.

## Limitations

- The course corpus has only 60 short sentences, so held-out scores vary with the split and trigram contexts are sparse.
- N-gram models use only the previous `n - 1` tokens and cannot represent long-range meaning.
- Sampling can repeat words or combine locally likely phrases into incoherent text.
- Add-k smoothing is a simple teaching method; this project does not implement back-off or interpolation.
- The Project Gutenberg experiment remains pending until its page and text can be verified.
