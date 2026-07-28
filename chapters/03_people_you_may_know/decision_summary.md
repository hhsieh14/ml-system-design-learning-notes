# Chapter 3 Decision Summary

## Starting design

- Optimize accepted or completed connections rather than request volume alone.
- Generate bounded 2-hop friends-of-friends plus complementary graph or cold-start candidates.
- Establish a pointwise binary classifier or learning-to-rank baseline.
- Use graph, profile, recency, direct-interaction, and exposure features.
- Narrow candidates cheaply and apply a richer MLP only to finalists when justified.
- Precompute and cache recommendations, then refresh lazily when stale.
- Treat repeated non-action as a gradual penalty and allow new signals to reactivate candidates.
- Promote a temporal GNN only after measurable incremental value over static recency features.

## Decisions that remain evidence-dependent

- Primary online metric and guardrails.
- Two-hop versus broader graph expansion.
- Static graph model versus temporal GNN.
- Node-memory content and state budget.
- Dot product, MLP, or staged scoring.
- Refresh threshold and lightweight-ranker role.
- Missing-value treatment.
- Negative-pair strategy and delayed-feedback attribution.
- Interleaving, A/B testing, or both.
