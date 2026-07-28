# Machine Learning System Design Study Handbook

A readable, alternatives-driven set of personal study notes for machine learning system design interviews and design reviews.

The handbook emphasizes a simple rule: **start with the minimum sufficient design, measure its limitations, and add complexity only when evidence justifies it.**

## Read the handbook

- [Complete PDF](handbook/ML_System_Design_Study_Handbook.pdf)
- [Chapter 0 - How to Approach an ML System Design Case](chapters/00_framework/chapter_00_ml_system_design_framework.md)
- [Chapter 1 - Designing a Large-Scale Social Feed Ranking System](chapters/01_ranking_model/chapter_01_design_a_ranking_model.md)
- [Chapter 2 - Predicting Ad Clicks on a Social Media Platform](chapters/02_ad_click_prediction/chapter_02_ad_click_prediction.md)
- [Chapter 3 - People You May Know](chapters/03_people_you_may_know/chapter_03_people_you_may_know.md)
- [Chapter 4 - Harmful Content Detection](chapters/04_harmful_content_detection/chapter_04_harmful_content_detection.md)

Each chapter includes a PDF, GitHub-readable Markdown, diagrams, and a decision summary.

## Primary reference and attribution

These notes are based largely on:

> Ali Aminian and Alex Xu, *Machine Learning System Design Interview*. ByteByteGo, 2023. ISBN 978-1-7360491-2-9.

The chapter structure, case topics, and many alternatives were learned from the book. The restructuring, minimum-scope framework, clarifications, comparison tables, rewritten explanations, original diagrams, interactive decision prompts, and personal interpretations are Hsiang-Yu (Andy) Hsieh's independent study work.

This repository is unofficial, is not affiliated with the authors or publisher, and is not a replacement for the original book. It does not include handwritten source scans or verbatim reproductions of the book's figures.

## Design approach

1. Define scope and requirements.
2. Separate model output from the product or service action.
3. Choose product and offline metrics.
4. Define data, labels, missingness, and time boundaries.
5. Establish an easy-to-implement baseline.
6. Compare richer and production-balanced alternatives.
7. Design training, serving, validation, and monitoring around measured limitations.
8. Record the selected design and the evidence that would change it.

## Repository layout

```text
chapters/   Individual chapter PDFs, Markdown, decision summaries, and diagrams
handbook/   Combined public PDF
linkedin/   Draft posts for sharing the project and chapters
docs/       Attribution, publication, visual, and QA notes
```

## Public-use note

Brief quotations and links to this repository are welcome with attribution. See [LICENSE.md](LICENSE.md) and [DISCLAIMER.md](DISCLAIMER.md).
