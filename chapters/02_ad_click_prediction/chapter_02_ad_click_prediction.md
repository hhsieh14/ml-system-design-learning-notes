# Chapter 2: Predicting Ad Clicks on a Social Media Platform

> Based on my study of *Machine Learning System Design Interview* (Aminian & Xu, 2023). The explanations, trade-off analysis and diagrams are my own. Notes marked **Beyond the book** are material I added.

**The design in one paragraph.** For each eligible ad, predict a **calibrated** click probability $\hat p=P(\text{click}\mid\text{user},\text{ad},\text{context})$. The auction ranks ads by expected value, $\text{bid}\times\hat p$, so the probability itself, not just its ordering, drives revenue. Start with logistic regression on crossed features. Move to a learned-interaction model (DCN or DeepFM) when it measurably beats LR. Use two-tower retrieval only when the ad inventory is too large to score everything. Keep frequency caps and hide/block rules in the serving policy, not in the model. Validate with point-in-time data, a shadow run, a canary and an A/B test.

![Ad-click prediction design map](diagrams/01_design_map.png)

## 1. Scope and product policy

- **Input:** a user, a slot in their feed, and the set of **eligible** ads (targeting, budget and policy checks already passed).
- **Model output:** $\hat p$ for each eligible ad.
- **Service action:** choose which ad fills the slot.

Why calibration matters more here than in feed ranking: in a cost-per-click auction the platform's expected revenue from showing an ad is

$$\text{eCPM}=\text{bid}_{\text{CPC}}\times\hat p\times1000 .$$

If $\hat p$ is 2× too high for one advertiser's ads, they win auctions they shouldn't, and the pricing is wrong. **Ranking quality (AUC) isn't enough: the numbers have to be right.**

![Repeated exposure policy](diagrams/02_repeated_exposure.png)

**Repeated exposure is a policy decision, not a model decision.** The model estimates the click probability. The serving policy decides how often the same ad may reappear: no cap, a fixed impression cap (e.g. 3 per user per day), or a time-decay fatigue rule. Hide and block actions remove the ad (or advertiser) for that user and also become training signals. Include "times this user has seen this ad" as a model feature so the model can learn fatigue, but enforce hard caps in policy.

## 2. Data and labels

![Data and labels](diagrams/03_data_and_labels.png)

| Family | Examples |
|---|---|
| ad / campaign | ad ID, advertiser ID, category, creative text/image embeddings, campaign age |
| user / context | user ID, demographics, language, device, location, time of day |
| interaction history | user's CTR on this category or advertiser, recent clicks, conversions, dwell |
| impression | position, times already shown, surface |

**One training example = one impression**, with features as they were at impression time.

- **Positive:** a click within a window (e.g. 30 min) after the impression.
- **Negative:** an impression with no click in the window. This is the transparent default.
- **The ambiguity:** "viewed but not clicked" isn't the same as "not interested". The user may have scrolled past without seeing the ad, or clicked a day later. Options:
  - require a minimum on-screen time before counting a negative;
  - down-weight uncertain negatives;
  - treat non-clicks as *unlabeled* and use positive-unlabeled (PU) learning.

  Start transparent and add these only if hidden positives measurably hurt calibration or ranking.
- **Delayed feedback:** clicks and especially conversions arrive late. Training on too-recent data mislabels future positives as negatives. Wait for the window, or model the delay explicitly.

## 3. Metrics

![Metrics](diagrams/04_metrics.png)

| Question | Metric |
|---|---|
| are the probabilities accurate? | **log loss** (binary cross-entropy); calibration ratio $\sum\hat p/\sum y$ (should be ≈ 1) |
| better than a named baseline? | **normalized entropy** |
| does the score separate clicks from non-clicks? | ROC-AUC; PR-AUC because clicks are rare |
| does the business improve? | CTR, conversions, revenue per 1,000 impressions |
| is the user experience protected? | hide/block rate, ad load, session length (guardrails) |

**Normalized entropy** (He et al., 2014) divides the model's log loss by the log loss of always predicting the average CTR $\bar p$:

$$\mathrm{NE}=\frac{-\frac1N\sum_i\big[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\big]}{-\big[\bar p\log\bar p+(1-\bar p)\log(1-\bar p)\big]} .$$

Lower is better, and NE < 1 means the model beats the constant predictor. Example: with $\bar p=2\%$ the denominator is 0.098 nats, so a model with log loss 0.090 has NE = 0.918. Always say which baseline you normalize by: the constant CTR, the production model, or the LR baseline.

> [!NOTE]
> **Beyond the book: correcting for negative downsampling**
>
> Clicks are ~1–2% of impressions, so it's common to keep only a fraction $w$ of negatives (e.g. $w=0.1$). The model then learns inflated odds, by a factor $1/w$. Recover the true probability with
>
> $$q=\frac{\hat p}{\hat p+(1-\hat p)/w} .$$
>
> For example, $\hat p=0.10$ with $w=0.1$ becomes $q\approx0.011$. Skipping this step breaks the eCPM auction even when AUC is unchanged.

## 4. Model alternatives

![Model progression](diagrams/05_model_progression.png)

| Model | How it handles feature interactions | Strength | Cost |
|---|---|---|---|
| **logistic regression** | none unless you add them by hand | fast, interpretable, calibrated by construction, easy online updates | misses interactions |
| LR + feature crosses | hand-made crosses (user_country × ad_category), hashed into buckets | captures the crosses you know about | combinatorial feature explosion; only crosses you thought of |
| GBDT | tree splits learn nonlinear interactions | strong on dense features | awkward with sparse IDs; slow to update incrementally |
| GBDT + LR | tree leaves become one-hot features for LR (He et al., 2014) | automatic cross discovery + LR's easy updates | two-stage pipeline |
| **factorization machine** | every pair $i,j$ gets weight $\langle\mathbf v_i,\mathbf v_j\rangle$ from low-rank embeddings | pairwise interactions even for rare pairs, $O(kn)$ time | only second-order |
| **DeepFM** | FM component + a deep MLP, sharing embeddings | low- and high-order interactions, no manual crosses | more latency and tuning |
| **DCN** | cross layers $\mathbf x_{l+1}=\mathbf x_0\mathbf x_l^\top\mathbf w_l+\mathbf b_l+\mathbf x_l$ build explicit crosses of degree $l+1$, plus a deep path | explicit, bounded-degree crosses at low cost | more complex than LR |
| xDeepFM | compressed interaction network: explicit vector-wise crosses | richer explicit interactions | heavier still |

![Interaction models](diagrams/06_interaction_models.png)

**Factorization machine:**

$$\hat y=w_0+\sum_iw_ix_i+\sum_{i<j}\langle\mathbf v_i,\mathbf v_j\rangle x_ix_j .$$

The pairwise term can be rewritten as $\tfrac12\sum_f\big[(\sum_iv_{if}x_i)^2-\sum_iv_{if}^2x_i^2\big]$, which costs $O(kn)$ instead of $O(kn^2)$. Because each feature has its own embedding, the model can score a pair (user_country, ad_category) it never saw together in training.

**My plan:** LR (with a few known crosses) as the anchor and the control arm. DCN and DeepFM as the challengers. Adopt one if it improves NE by a meaningful margin (e.g. ≥ 0.5% relative, which is significant at ad scale), stays calibrated after correction, and fits the latency budget.

## 5. Where the two-tower model belongs

![Two-tower placement](diagrams/07_two_tower_placement.png)

A two-tower model scores with a dot product between a user embedding and an ad embedding. That makes it great for **retrieval** (ad embeddings precomputed, ANN search over millions of ads) and weak as the **final CTR model**: there are no cross-features, and its scores aren't calibrated probabilities. So:

- **small inventory** (thousands of eligible ads per request): skip retrieval and score them all with the CTR model;
- **large inventory:** two-tower / ANN retrieval → a few hundred candidates → the rich, calibrated CTR model.

![Candidate generation](diagrams/08_candidate_generation.png)

Candidate sources (ANN, collaborative filtering, eligibility/campaign rules, others) are merged and deduplicated, and each ad keeps a feature for which sources found it.

## 6. Keeping the model fresh

![Continual learning](diagrams/09_continual_learning.png)

Ads churn fast: new campaigns every day, and user interests and seasonality shift. Stale models lose calibration first. Triggers for an update: calibration ratio drifting from 1, campaign turnover, feature distribution shift, falling online metrics. Update options, from lightest to heaviest:

1. **frequent retraining / warm-start fine-tuning** on the latest window (often daily or more);
2. **regularization** toward the previous weights, to limit forgetting;
3. **replay** a sample of older data;
4. **adapters / new capacity** for new ad types.

Use the lightest option that fixes the measured drift.

## 7. Validation, rollout and calibration

![Validation and calibration](diagrams/10_validation_calibration.png)

- **Shadow:** the new model scores live requests without affecting auctions. Compare score distributions, calibration and latency with production.
- **Canary:** it serves a small slice. Check for errors, spend anomalies and advertiser complaints.
- **A/B test:** measures causal impact on CTR, revenue and hide rate. Watch **budget effects**: treatment and control share advertiser budgets, which can bias results, so budget-split or advertiser-level randomization may be needed.
- **Calibration methods**, all fitted on a recent holdout:

| Method | Fits | Use when |
|---|---|---|
| Platt scaling | $\sigma(a\cdot\text{logit}+b)$ | miscalibration is a smooth shift or scale |
| isotonic regression | any monotone map | lots of data, irregular miscalibration |
| temperature scaling | a single $T$ dividing logits | neural models that are over-confident |

## 8. Leakage

![Leakage prevention](diagrams/11_leakage.png)

Every feature and label is computed relative to the **impression time**:

- features use only events before the impression (e.g. "user's clicks in the last 7 days *before* this impression");
- labels use an explicit window after it;
- train/validation/test are split **chronologically**, e.g. train on days 1–28, validate on day 29, test on day 30.

A random split lets the model see tomorrow's behavior of the same campaign, and the offline metrics come out better than what you'll ever see in production.

## 9. Reference architecture

![Reference architecture](diagrams/12_reference_architecture.png)

**Online:** request → eligibility + frequency policy → candidate retrieval (if the inventory is large) → feature hydration → CTR model + calibration → auction ($\text{bid}\times q$) → ad served.
**Offline:** impression/click logs → label construction with windows → point-in-time training data → training + chronological evaluation → model registry → staged rollout.

## 10. Decisions and when I'd change them

| Area | Starting choice | Switch when (measured) |
|---|---|---|
| exposure | fixed frequency cap + hide/block rules | a dynamic fatigue model improves revenue at an equal or lower hide rate in an A/B test |
| negatives | all non-clicked impressions (with a minimum view time) | PU or weighting improves calibration on a trusted holdout |
| baseline | LR + known crosses | – |
| richer model | DCN or DeepFM | ≥ ~0.5% relative NE gain, calibrated, within p99 latency |
| retrieval | none (score all eligible ads) | eligible ads per request exceed what the CTR model can score in budget |
| calibration | Platt on a daily holdout | the reliability curve is non-sigmoid → isotonic |
| freshness | daily warm-start retraining | NE decays measurably within a day → more frequent or online updates |
| leakage | point-in-time features, chronological splits | never optional |

## 11. Questions and answers

<details><summary>Why optimize log loss rather than AUC?</summary>

AUC ignores calibration: multiplying every prediction by 2 leaves AUC unchanged and breaks the auction. Log loss (and NE) penalizes both poor ranking and wrong probabilities.
</details>

<details><summary>How do I handle a brand-new ad with no history?</summary>

Use content features (creative embeddings, category, advertiser history) so the model can generalize, add a small exploration budget (e.g. Thompson sampling on an uncertainty estimate) so new ads get impressions, and back off to advertiser- or category-level CTR priors.
</details>

<details><summary>Which normalized-entropy baseline should I report?</summary>

State it explicitly. The constant average CTR is standard (NE < 1 means "better than knowing nothing about the user"). For launch decisions, compare with the current production model.
</details>

<details><summary>Should frequency be a feature or a rule?</summary>

Both. The feature lets the model learn that the 5th showing is less likely to be clicked. The rule enforces a hard limit that the product guarantees users, whatever the model says.
</details>

<details><summary>What does the A/B test need beyond CTR?</summary>

Revenue and advertiser value (conversions, cost per acquisition), user guardrails (hide/block rate, session time), and a check that treatment and control didn't compete for the same budgets.
</details>

---

[← Chapter 1: Feed Ranking](../01_ranking_model/chapter_01_design_a_ranking_model.md) · [Chapter 3: People You May Know →](../03_people_you_may_know/chapter_03_people_you_may_know.md)
