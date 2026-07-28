# Chapter 1 Decision Summary

## Starting architecture

Use a hybrid retrieval-and-ranking system:

1. precompute reusable content representations and searchable indexes;
2. retrieve candidates from ANN/two-tower, social-graph, and other sources;
3. merge and deduplicate content IDs while preserving retrieval-source signals;
4. hydrate user, content, relationship, history, and context features;
5. apply a shared-bottom multi-task DNN;
6. combine selected behavior probabilities into the serving score;
7. validate through offline gates, a 5% treatment A/B test, and staged canary rollout.

## Main trade-offs

- More granular heads improve analysis and utility construction but add training and serving complexity.
- Two-tower retrieval scales well but models fewer pairwise interactions than the final ranker.
- Multi-source retrieval improves coverage but requires merge, deduplication, feature hydration, and source budgeting.
- Continual learning improves freshness but can create catastrophic forgetting or time-distribution distortion.

## Open decisions

- task-loss weights;
- serving utility weights;
- dwell and attribution thresholds;
- candidate-source budgets;
- index type and refresh cadence;
- offline and online approval gates;
- retraining triggers.
