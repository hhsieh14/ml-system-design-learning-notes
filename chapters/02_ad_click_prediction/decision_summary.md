# Chapter 2 Decision Summary

## Starting design

- Predict calibrated click probability for eligible ads.
- Keep repeated-exposure, hide, and block rules in the serving policy.
- Use clicks as positives and begin with a transparent impression-based negative policy.
- Establish Logistic Regression as the baseline.
- Compare DCN and DeepFM only when learned feature interactions are needed.
- At large inventory scale, use two-tower or ANN for candidate retrieval and a richer CTR model for final ranking.
- Validate with point-in-time data, shadow deployment, limited live traffic, A/B testing, and calibration checks.

## Decisions that remain evidence-dependent

- Frequency cap versus time-based or dynamic fatigue.
- Hard negatives versus unlabeled or PU-style treatment.
- Normalized cross-entropy baseline.
- Two-tower placement.
- Continual-learning method.
- Calibration method.
