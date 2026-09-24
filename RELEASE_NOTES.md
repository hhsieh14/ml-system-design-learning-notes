# Release notes

## 1.1.0

- Rewrote all five chapters as self-contained reading material: numbers that previously lived only in diagrams are now in the text, and each chapter adds worked calculations (QPS, candidate counts, index memory, normalized entropy, downsampling correction, class-balanced weights) and answered questions.
- Every decision table now states a measurable switch condition.
- Made the rollout order consistent across chapters: offline gate → shadow → canary → A/B → ramp (Chapter 1 diagram redrawn).
- Added material marked "Beyond the book" with citations in [REFERENCES.md](REFERENCES.md): MMoE, sampling-bias correction, calibration for auctions, network interference in A/B tests, perceptual-hash matching, GradNorm vs Gradient Blending, effective-number class weighting.
- Chapter 4's blank worksheets are replaced with a filled decision record.
- Rebuilt the chapter PDFs and the combined handbook from the Markdown (`tools/build_pdfs.py`).

## 1.0.0

First public release: five chapters, a combined PDF, Markdown sources, original diagrams and decision summaries.
