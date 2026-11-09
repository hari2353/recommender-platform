# Borrowed ideas — recommender-platform (personal AI agent reordering)

Source: Instagram post `DfKqdH8gSMj` — "Today every app personalises you in
isolation. Netflix knows what you like. Spotify knows what you like. A new paper
studies what happens next: a personal AI agent that sits between you and those
apps, reordering their recommendations using the ..."

## Concept

A personal AI agent that sits between the user and multiple apps, reordering
their recommendations based on cross-app context. This is the natural extension
of graph-memory-platform (per-user memory graph → personalized context).

## Where it lands in recommender-platform

| Idea | Current state | Borrowed change |
|---|---|---|
| Cross-app personalization | Single-app recommender (CMAB over skills) | Add a `cross_app_context` input to the recommender that pulls from graph-memory-platform MCP (user's memory graph) |
| Personal agent reordering | Recommendations served directly | Add a re-ranking layer that adjusts scores based on cross-app signals (e.g., "user just searched for X on another app") |
| Hypothesis-driven | Already has hypothesis layer (depth_in_role, next_role_prep, etc.) | Add a `cross_app_interest` hypothesis type derived from graph-memory signals |

## Adoption order

1. Add `graph-memory-platform` MCP client to recommender-platform (optional, feature-flagged).
2. Add `cross_app_context` to the recommender input (pulls from MCP `get_context`).
3. Add `cross_app_interest` hypothesis type.
4. A/B test: measure lift from cross-app context vs single-app.

## What is NOT borrowed

- Replacing the existing CMAB recommender — the cross-app layer is additive.
- Real-time cross-app data sharing — privacy concerns; user must opt in.
EOF