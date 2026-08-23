# Security Notes

A recommendation platform processes behavioral data at scale — the threat model here is broader than a
typical CRUD service: untrusted text flows into LLM prompts, event streams shape user profiles, and
admin endpoints mutate learning state.

## Controls in place

- **Prompt-injection containment (two layers).** Course text and RAG questions are never concatenated
  into commands — they are data for extraction/answer prompts whose outputs are parsed as strict JSON or
  plain strings, then used only as *lookup labels* against the taxonomy. An injected "skill" cannot
  escalate past `confidence < 0.5` filtering into the index. LLM-tagged content enters as
  `content_version`-stamped, re-derivable metadata.
- **Event stream hygiene.** Bounded dedup deque (idempotent replay protection), 7-day late-event
  watermark, per-request event cap (500), dwell bonus capped (+25%) so a single inflated event cannot
  dominate a profile.
- **Input validation.** Pydantic models with length bounds on every body field; `k` params range-bounded;
  `strategy` restricted to an enum; path params length-checked; unknown feedback actions rejected (422).
- **Profile integrity.** Per-user normalization prevents power-user/bot domination; dislikes are hard
  filters scoped to that user only — no cross-user write path exists.
- **Admin surface.** `/admin/*` routes enforce an optional `RECSYS_ADMIN_TOKEN` header when configured —
  set it in any shared deployment.
- **Premium entitlement** is checked per request server-side (`402`), not trusted from client flags.
- **Privacy / GDPR:** `DELETE /v1/users/{id}` drops profiles immediately; hard filters and affinities are
  per-tenant keyed; impression logs are in-memory in this scope (no PII sink by design).
- **Security headers** (`nosniff`) on all responses.

## Known gaps (deliberate for open-source demo scope)

- No authN on user-facing endpoints — deploy behind [auth-gateway](https://github.com/hari2353/auth-gateway)
  or an API gateway; set `RECSYS_ADMIN_TOKEN` for admin routes
- In-memory stores (profiles, impressions, bandit arms) — swap for Postgres/Redis behind the same
  interfaces for persistence; multi-node deployments need shared state
- Impression→feedback join is in-memory with 24h lazy resolution — production needs durable attribution
- No rate limiting — front with a proxy
