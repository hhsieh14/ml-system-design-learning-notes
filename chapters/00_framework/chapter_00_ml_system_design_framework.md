# Chapter 0: How I Approach an ML System Design Case

> Based on my study of *Machine Learning System Design Interview* (Aminian & Xu, 2023). The framework, explanations and diagrams are my own restatement. Notes marked **Beyond the book** are material I added.

Every case in this handbook, whether feed ranking, ad clicks, friend suggestions or content moderation, follows the same arc: pin down what the product needs, choose how to measure it, build the simplest system that could work, and add complexity only where a measurement shows the simple system falls short. This chapter is that arc as a reusable procedure.

## 1. The core principle: minimum sufficient design

> **Choose the smallest architecture that satisfies the current requirements. Add a component only when it fixes a demonstrated quality, latency, scale or operational problem.**

![Minimum sufficient design](diagrams/01_minimum_sufficient_design.png)

This isn't minimalism for its own sake. Every component you add is something to train, serve, monitor, debug and explain. A two-stage retrieval + ranking system is the right answer at 10⁹ items. At 10⁴ items it's overhead, because you can score everything. The interviewer (or the design review) wants to see that you know *which* problem each component solves.

## 2. The design order

![Design order](diagrams/02_design_order.png)

| # | Step | The question I answer |
|---:|---|---|
| 1 | **Scope** | What exactly is the input, the model's output and the product's action? What's out of scope? |
| 2 | **Requirements** | Scale (users, items, QPS), latency budget, freshness, languages and modalities, cost, fairness and safety constraints |
| 3 | **Service decision** | Which decision does the score feed: rank, filter, threshold, route to a human? |
| 4 | **Metrics** | One primary product metric, guardrails, and offline metrics that predict them |
| 5 | **Data and labels** | What is one training example? Where do labels come from, how noisy are they, when do they arrive? |
| 6 | **Baseline** | The simplest credible model: heuristics, logistic regression, popularity, collaborative filtering |
| 7 | **Richer alternatives** | What specific limitation of the baseline does each fix? |
| 8 | **Features and training** | Feature families, freshness, point-in-time correctness, loss, sampling, imbalance |
| 9 | **Serving and scale** | Online vs precomputed, candidate generation, caching, ANN, latency budget per stage |
| 10 | **Deployment and monitoring** | Offline gate → shadow → canary → A/B → ramp; drift, retraining triggers, rollback |
| 11 | **Decision record** | What I chose, and what measured result would make me switch |

### Back-of-envelope scale

Numbers turn "it's big" into architecture. For example, with 500M daily active users making about 10 feed requests a day:

```text
500e6 × 10 / 86,400 s ≈ 58,000 requests/s on average  →  plan for ~2–3× at peak
```

If each request ranks 1,000 candidates, that's about $6\times10^7$ model scores per second. That's why the final ranker only ever sees a small candidate set, and why retrieval uses precomputed embeddings and ANN search.

## 3. Three lenses on alternatives

![Alternatives lens](diagrams/03_alternatives_lens.png)

For each design area I consider three perspectives:

- **Easy to implement:** a credible baseline for fast validation, debugging and cheap serving. It's also the yardstick every other option has to beat.
- **More granular or accurate:** a richer option aimed at a specific gap: temporal dynamics, graph structure, multiple modalities, feature interactions, per-behavior outputs.
- **Production-balanced:** the option that best trades off quality, latency, scale, reliability, cost and maintainability. This is often a *combination*: a cheap stage to narrow the candidates, then a rich model on the few that remain.

These are ways of looking at the problem, not a rule to always build three models.

## 4. How I compare alternatives

![Compare alternatives](diagrams/04_compare_alternatives.png)

For every option I write down six things:

1. **the problem it solves**, e.g. "the LR baseline can't learn user × ad-category interactions";
2. **the expected benefit**, e.g. "+0.5–1% relative normalized entropy";
3. **the cost or risk**: latency, training cost, operational burden, interpretability;
4. **where it fits best**: data size, traffic, freshness needs;
5. **the evidence needed**: an offline metric on a time-based holdout, then an online test;
6. **the switch condition**, stated so it can be measured, e.g. "adopt DCN if it improves offline NE by ≥ 0.5% relative, holds p99 latency under 20 ms, and wins a two-week A/B test on revenue without raising hide rate."

Point 6 is what separates a design decision from a preference.

## 5. Model output ≠ service action

![Model and service separation](diagrams/05_model_service_separation.png)

Keep the model's granular predictions separate from the product's decision:

```text
model:   P(like), P(comment), P(hide), …     or     P(violence), P(nudity), …
policy:  score = Σ wᵢ·pᵢ, thresholds, caps, business rules, overrides
action:  rank / show / demote / send to review / remove
```

This separation lets you change product policy (weights, thresholds, frequency caps) without retraining. It keeps the model's outputs meaningful as probabilities. And it gives you per-component diagnostics when something goes wrong. It only works if those probabilities are **calibrated**: a weighted sum of miscalibrated probabilities is not a meaningful utility.

## 6. Staged production design

![Staged production design](diagrams/06_staged_production_design.png)

Simple and rich models usually coexist as stages:

```text
millions of items → cheap retrieval (ANN, graph, rules) → ~1,000 candidates
                  → rich ranker → ~50 items → policy / diversity / business rules → shown
```

Each stage has its own metric: recall@K for retrieval, nDCG or precision@K for ranking, and the product metric at the end.

## 7. Serving and deployment

![Serving and deployment](diagrams/07_serving_deployment.png)

**Serving mode**, chosen by freshness and latency:

| Mode | Use when | Example |
|---|---|---|
| online | the input only exists at request time, or freshness matters | feed ranking with the latest interactions |
| precomputed (batch) | inputs change slowly and the set of users/items is predictable | nightly friend suggestions for active users |
| hybrid | stable parts precomputed, a light fresh re-rank online | cached candidates re-ranked with fresh features |

**Rollout order.** Each stage answers a different question, so I use them in sequence:

1. **Offline evaluation** on a time-based holdout: *is it better on data it hasn't seen?*
2. **Shadow deployment:** the new model scores live traffic without acting. *Does it run correctly (latency, errors, score distributions)?*
3. **Canary:** real actions for a small slice of traffic. *Is it healthy on real users?*
4. **A/B test:** randomized treatment vs control, e.g. 5% / 95%. *Does it causally improve the product metric without hurting guardrails?*
5. **Ramp to 100%**, with rollback available at every step.

Interleaving is a faster, more sensitive alternative to A/B for comparing *rankers*: it mixes two rankings in one list and counts which one's items get clicked. A/B is still needed for product-level effects.

## 8. Closing statement

> "I'd start with the smallest system that satisfies the core requirements, establish a measurable baseline, and add complexity only when a specific product, quality, latency or scale limitation justifies it, stating in advance the measured result that would make me switch."

## 9. Questions and answers

<details><summary>How do I budget a 45-minute interview?</summary>

About 5 minutes on scope and requirements, 5 on metrics, 5 on data and labels, 15 on the model (baseline → richer option), 10 on serving and scale, and 5 on deployment and monitoring. Say the plan out loud at the start so the interviewer can redirect you.
</details>

<details><summary>How do I pick the primary metric when the product has several goals?</summary>

Choose the one closest to long-term user value that a two-week experiment can still move (e.g. meaningful interactions or accepted connections, not raw clicks). Make the others guardrails with explicit "must not drop more than X%" limits.
</details>

<details><summary>When is a heuristic baseline acceptable in production?</summary>

When it meets the requirement and the ML gain wouldn't pay for its cost. Popularity plus recency, or a rules-based filter, is sometimes the right launch system, and it becomes the control arm of every later experiment.
</details>

<details><summary>What must every design mention, whatever the case?</summary>

Point-in-time features (no leakage), a time-based validation split, a calibration check if scores are combined or thresholded, a rollout plan, and monitoring with a defined retraining or rollback trigger.
</details>

---

[Next: Chapter 1, Feed Ranking →](../01_ranking_model/chapter_01_design_a_ranking_model.md)
