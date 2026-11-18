# Dataset sources and attribution

## askThili course corpus

`course_corpus.txt` reproduces the 60 everyday sentences in Lesson 2, “Count
the n-grams,” of the supplied course document, *Language Modeling from First
Principles* (askThili / Thili.ai). It is included to reproduce the course
baseline. Tokenization lowercases the text; the source file preserves the
capitalization, wording, punctuation, and sentence order shown in the course.

The course document was supplied by the project owner. This repository does
not assign a separate license to the course text. Keep this attribution with
the corpus if it is redistributed.

## Project Gutenberg experiment (pending)

The intended comparison corpus is *Alice's Adventures in Wonderland* by Lewis
Carroll, [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11).
The source was checked on 2026-11-18, but Project Gutenberg returned HTTP 503
with a maintenance notice, so no text was downloaded, no corpus metrics are
reported, and the experiment remains pending.

Use `load_gutenberg_text(url)` in `next_word_predictor.py` after verifying the
current plain-text download URL on the ebook page. The loader accepts only an
HTTPS URL hosted by `www.gutenberg.org`, reads text into memory, and does not
save a copy. Before redistributing downloaded text, review the current Project
Gutenberg terms and the copyright status for the jurisdiction where it will be
used. Public-domain status in the United States does not settle copyright
status in every country, and the terms for redistributing a Project Gutenberg
edition may apply.

Source page: <https://www.gutenberg.org/ebooks/11>

## Experiment policy

Baseline and held-out scores use only `course_corpus.txt`. The Alice comparison
is reported as pending until its source and text can be verified. No corpus is
silently substituted and no metrics are invented.
