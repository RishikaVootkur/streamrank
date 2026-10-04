# 1. Record architecture decisions

Date: 2026-10-04

## Status

Accepted

## Context

This project makes many design choices across data, modeling, serving, and infrastructure. Reviewers and future contributors need to know what was chosen, what else was considered, and why.

## Options considered

1. No written record; rely on commit history and pull requests.
2. A single design document updated over time.
3. Architecture Decision Records: one short file per decision.

## Decision

Use Architecture Decision Records in `docs/adr/`, numbered `NNNN-short-title.md`. Each record has Context, Options considered, Decision, Consequences, and Sources.

## Consequences

- Each significant change links to the ADR that explains it.
- Superseded decisions stay in place and point to their replacement.

## Sources

- Michael Nygard, "Documenting Architecture Decisions": https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- ADR GitHub organization: https://adr.github.io/
