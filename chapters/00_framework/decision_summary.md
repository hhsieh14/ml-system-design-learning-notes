# Chapter 0 Decision Summary

**Principle:** use the minimum sufficient design. Add a component only when a requirement needs it, a measured limitation justifies it, or production constraints force it.

**Design order:** scope → requirements (with back-of-envelope numbers) → service decision → metrics → data and labels → baseline → richer alternatives → features and training → serving and scale → deployment and monitoring → decision record.

**Alternatives lens:** easy to implement (the baseline and control arm), more granular or accurate (fixes a named gap), production-balanced (often a cheap stage followed by a rich stage). These are perspectives, not a quota of three models.

**For each alternative, record:** the problem solved, the expected benefit, the cost or risk, where it fits, the evidence needed, and a *measurable* switch condition.

**Model output ≠ service action:** calibrated granular predictions → policy (weights, thresholds, caps) → action.

**Rollout:** offline gate → shadow → canary → A/B test → ramp to 100%, with rollback at every step.

> "I'd start with the smallest system that satisfies the core requirements, establish a measurable baseline, and add complexity only when a specific product, quality, latency or scale limitation justifies it, stating in advance the result that would make me switch."
