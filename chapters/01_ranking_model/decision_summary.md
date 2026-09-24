# Chapter 1 Decision Summary: Feed Ranking

## Starting architecture

1. Precompute post embeddings and ANN indexes (HNSW or IVF-PQ; PQ cuts 1B × 128-d from ~512 GB to ~16 GB).
2. Retrieve candidates in parallel from two-tower ANN, the social graph and fresh/trending sources, each with a budget.
3. Merge and deduplicate by post ID, keeping which sources found each post as a feature.
4. Hydrate point-in-time user, post, relationship, history and context features.
5. Score with a shared-bottom multi-task DNN: like, comment, share, dwell ≥ T, hide, block.
6. Calibrate each head, then rank by utility $\sum_iw_i\hat P_i$ (negative weights for hide/block).
7. Roll out: offline gate → shadow → canary → A/B (5% / 95%, ~2 weeks) → ramp.

## Switch conditions

| Area | Switch when |
|---|---|
| shared-bottom → MMoE | a head loses > 1% relative PR-AUC when trained jointly |
| multi-source → single-source retrieval | one source reaches ≥ 95% of multi-source recall@K at lower latency |
| serving weights | an A/B test on weight vectors finds a better primary-metric vs hide-rate trade-off |
| daily → weekly retraining | calibration stays within tolerance for several weeks |

## Still open (tuned empirically)

Task-loss weights, dwell threshold T and label windows, source budgets, index refresh cadence, drift thresholds.
