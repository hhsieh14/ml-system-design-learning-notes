# Chapter 3 Decision Summary: People You May Know

## Starting design

- Optimize **accepted connections** (guardrails: rejections, "don't know" feedback, blocks, spam).
- Candidates: 2-hop friends-of-friends capped per user (~2,000 by mutual count), personalized random walks, cold-start profile sources. Exclude existing, pending, blocked and opted-out users.
- Model: pointwise GBDT/MLP on profile, graph (mutual count, Adamic–Adar, recency) and interaction/exposure features, all computed on the graph before prediction time.
- Scoring: cheap narrowing (~2,000 → ~200), then an MLP on the finalists.
- Serving: batch precompute for active users → cache with staleness metadata → light fresh re-rank at request time → lazy refresh.
- Repeated non-action decays the score; new signals reactivate the candidate.
- Negatives: exposure + graph-hard + some random.
- Evaluation: Precision@K / mAP / PR-AUC offline → interleaving → cluster-randomized A/B (network interference).

## Scale arithmetic

1B users, 300M DAU, ~1,000 friends ⇒ ~10⁶ two-hop paths per user; all pairs for all DAU = 3×10¹⁴ scores/day (infeasible), so the pool is capped and scored in stages.

## Switch conditions

| Area | Switch when |
|---|---|
| static → temporal GNN | ≥ ~2% relative mAP offline and a significant online gain in accepts |
| cheap + MLP → cheap only | the MLP adds no measurable lift |
| refresh cadence | staleness measurably hurts acceptance |
