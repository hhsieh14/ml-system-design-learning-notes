# Harmful Content Detection - Decision Summary

## Starting design

- Late fusion and shared multi-label modeling are credible baselines.
- Early fusion is the preferred richer multimodal design.
- Shared-bottom multi-task learning is the preferred granular architecture.
- User reports provide weak supervision; human-labeled data provides trusted evaluation and calibration.
- Category scores are aggregated into an overall harmful score.
- Policy routing supports keep, demote, review, and remove.
- Severe categories use a real-time path; lower-urgency signals may be asynchronous.
- Shadow deployment is the preferred first rollout stage.

## Switching conditions

Keep the simpler baseline when the richer design does not produce measurable cross-modal or category-level lift that justifies the added complexity.
