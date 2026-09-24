# Machine Learning System Design: Study Handbook

My notes on designing production ML systems: how to go from a vague product goal to a system that can be built, measured and shipped. Each case works through requirements, back-of-envelope scale, metrics, labels, a baseline, richer alternatives, serving, rollout and monitoring. Every design choice comes with **the measured result that would make me change it**.

> **The rule I apply throughout:** start with the minimum sufficient design, measure its limits, and add complexity only when the evidence justifies it.

## Chapters

| # | Case | What it's really about |
|---:|---|---|
| 0 | [How I approach a case](chapters/00_framework/chapter_00_ml_system_design_framework.md) | the reusable procedure, alternatives lens, switch conditions, rollout order |
| 1 | [Social feed ranking](chapters/01_ranking_model/chapter_01_design_a_ranking_model.md) | multi-source retrieval, multi-task ranking, calibrated utility scores, ANN indexing at 10⁹ items |
| 2 | [Ad click prediction](chapters/02_ad_click_prediction/chapter_02_ad_click_prediction.md) | calibration for auctions, normalized entropy, FM/DCN/DeepFM, downsampling correction, leakage |
| 3 | [People You May Know](chapters/03_people_you_may_know/chapter_03_people_you_may_know.md) | link prediction, graph-scale arithmetic, cached serving, temporal GNNs, network interference |
| 4 | [Harmful content detection](chapters/04_harmful_content_detection/chapter_04_harmful_content_detection.md) | multimodal fusion, weak labels, class imbalance, severity-aware policy, shadow deployment |

Each chapter folder also has a one-page **decision summary** and a PDF. The [complete handbook PDF](handbook/ML_System_Design_Study_Handbook.pdf) combines all five. To practice on a new problem, use the [case template](chapters/00_framework/reusable_case_template.md).

## How each case is organized

1. Scope: input, model output, service action, out of scope
2. Requirements with back-of-envelope numbers (QPS, candidate counts, memory)
3. Metrics by stage: retrieval, ranking, calibration, online, guardrails
4. Labels and data: what one example is, label windows, delayed feedback, leakage
5. Baseline → richer alternatives, with what each fixes and what it costs
6. Serving: candidate generation, caching, latency budget
7. Rollout: offline gate → shadow → canary → A/B → ramp; monitoring and retraining triggers
8. A decision table with measurable switch conditions, and answered questions

## Related

- [ML course notes](https://github.com/hhsieh14/AI-courses-notes-reader): the theory underneath (statistical ML, learning theory, temporal models).

## Attribution

These notes are based on my study of:

> Ali Aminian and Alex Xu, *Machine Learning System Design Interview*. ByteByteGo, 2023. ISBN 978-1-7360491-2-9.

The case topics and many of the alternatives come from the book. The framework, explanations, calculations, added material (marked **Beyond the book**, with [references](REFERENCES.md)), diagrams and decision records are my own. This project is unofficial, is not affiliated with the authors or publisher, and is not a substitute for the book. See [DISCLAIMER.md](DISCLAIMER.md).

## License

My original content here is licensed under [CC BY-NC 4.0](LICENSE): you're welcome to share and adapt it for non-commercial use with credit. The book itself is not covered. See the [license notice](LICENSE_NOTICE.md) for scope and suggested attribution.
