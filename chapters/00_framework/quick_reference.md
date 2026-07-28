# ML System Design Quick Reference

```text
1. Scope
2. Requirements
3. Service decision
4. Metrics
5. Data and labels
6. Easy baseline
7. Richer alternative
8. Industrial preference
9. Features
10. Training
11. Serving
12. Scale
13. Deployment
14. Decision record
```

## Alternatives rule of thumb

### Easy to implement

Use for:

- baseline;
- fast validation;
- debugging;
- low-cost serving.

### More granular or accurate

Use when the baseline misses:

- detailed outputs;
- temporal patterns;
- graph relationships;
- multimodal interactions;
- difficult feature interactions.

### Industrial preference

Use when balancing:

- quality;
- latency;
- scale;
- reliability;
- cost;
- maintainability.

These are perspectives, not a requirement to create exactly three models.

## Minimal-scope rule

```text
Add a component only when:
1. a requirement needs it;
2. a measured limitation justifies it; or
3. production constraints require it.
```

## Final interview sentence

> Start with the smallest system that satisfies the core requirements, establish a measurable baseline, and add complexity only when a specific product, quality, latency, or scaling limitation justifies it.
