# Chapter 0 Decision Summary

## Core principle

Use the minimum sufficient design that fulfills the current requirements.

## Default design sequence

Scope -> requirements -> service decision -> metrics -> data and labels -> baseline -> alternatives -> features and training -> serving and scale -> deployment -> decision record.

## Alternatives lens

- **Easy to implement:** credible baseline and fast validation.
- **More granular or accurate:** complexity that solves a measured limitation.
- **Production-balanced:** quality, latency, scale, reliability, cost, and maintainability together.

These are perspectives, not a requirement to create exactly three models.

## Decision rule

For each alternative, record the problem solved, expected benefit, cost, best-fit scenario, evidence needed, and switching condition.

## Interview sentence

> I would begin with the smallest system that satisfies the core requirements, establish a measurable baseline, and add complexity only when a specific product, quality, latency, or scaling limitation justifies it.
