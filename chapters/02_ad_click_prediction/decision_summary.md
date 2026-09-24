# Chapter 2 Decision Summary: Ad Click Prediction

## Starting design

- Predict a **calibrated** $P(\text{click})$ for eligible ads; the auction ranks by $\text{bid}\times\hat p$.
- Frequency caps and hide/block handling live in the serving policy; "times shown" is also a model feature.
- Labels: click within a window = positive; impression without a click (after a minimum view time) = negative; respect delayed feedback.
- Baseline: logistic regression with a few known crosses. Challengers: DCN, DeepFM.
- Metrics: log loss and normalized entropy (with a named baseline), calibration ratio, PR-AUC/ROC-AUC; online CTR, revenue, conversions, hide rate.
- If negatives are downsampled at rate $w$: $q=\hat p/(\hat p+(1-\hat p)/w)$.
- Two-tower/ANN retrieval only when the eligible inventory is too large to score in full.
- Point-in-time features, explicit label windows, chronological splits.
- Rollout: shadow → canary → A/B (watch shared advertiser budgets) → ramp. Calibrate with Platt, isotonic or temperature scaling.

## Switch conditions

| Area | Switch when |
|---|---|
| LR → DCN/DeepFM | ≥ ~0.5% relative NE gain, calibrated after correction, within p99 latency |
| fixed cap → dynamic fatigue | better revenue at equal or lower hide rate in an A/B test |
| plain negatives → PU/weighted | better calibration on a trusted holdout |
| daily → more frequent updates | NE decays measurably within a day |
