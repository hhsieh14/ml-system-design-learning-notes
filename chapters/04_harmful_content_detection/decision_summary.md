# Chapter 4 Decision Summary: Harmful Content Detection

## Starting design

- **Cascade:** perceptual-hash match against known violating media → cheap per-modality classifiers → full multimodal multi-task model → human review for uncertain cases.
- **Model:** multilingual text encoder (e.g. XLM-R), CLIP-style image encoder, frame-sampled video encoder → early fusion → shared layers → one head per category. Late fusion and shared multi-label are the baselines.
- **Missing modalities:** learned missing embedding + presence flag, trained with modality dropout.
- **Labels:** reliability-weighted user reports and moderator decisions for training; an independent trusted human-labeled set for evaluation, calibration and thresholds.
- **Imbalance:** class-balanced weights (effective number); add focal loss if easy negatives dominate. Balance tasks (GradNorm) or modalities (Gradient Blending) only after measuring which is out of balance.
- **Policy:** per-category removal thresholds first (precision ≥ 95%), then a severity-weighted risk for review / demote bands; reviewer capacity sets the review band.
- **Author history:** used for routing and priority, not in the content harm score.
- **Metrics:** impression-weighted prevalence (primary), proactive rate, appeal reversal rate; offline per-category PR-AUC and recall at the target precision.
- **Deployment:** shadow → limited enforcement (one category or region) → full.

## Switch conditions

| Area | Switch when |
|---|---|
| early → late fusion | early fusion gains < ~1 pt PR-AUC on multimodal posts |
| multi-task → multi-label | the baseline is within noise on every category |
| threshold hybrid → learned aggregator | it beats the hybrid on overall labels and passes policy review |
| mean → max-risk comment pooling | alarming single comments are being missed |
