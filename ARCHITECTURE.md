# Architecture & Scalability

How this system is designed to grow from a laptop demo to a multi-tenant production service — what
changes at each order of magnitude, and which seams are already in place.

## Capacity model (current defaults, single node)

| Component | Cost per op | Practical ceiling (single node) |
|---|---|---|
| Event ingest | dict validate + set lookup + profile delta | ~5–10k events/s |
| Profile read | decay math over user's skills | O(user_skills), µs |
| Retrieval | skill-weight overlap scan over catalog | O(catalog) — fine ≤50k items |
| Rerank + MMR | embed query (cached) + pairwise Jaccard on top-N | O(k²) after retrieval |
| CMAB sample | Beta draws over arms | O(arms), ns–µs |
| Chunk search | 256-dim numpy dot over chunk matrix | ~1M chunks × 4ms/query |

## Growth stages

### 10× — one box is still enough (≤500k items, ≤100 rps)
- Swap `InMemoryBus` → Pulsar (`RECSYS_BUS=pulsar`, zero code change — same interface)
- Move `ProfileStore`/`ChunkStore`/impressions behind their existing interfaces onto Redis/Postgres
- Enable `RECSYS_ADMIN_TOKEN`; put auth-gateway in front
- Add the Redis cache-aside layer for hot `(user_id → recommendation slate)` with short TTL

### 100× — stateless API tier + partitioned stream
- **Partition the event topic by `user_id`** → each consumer instance owns an exclusive profile shard;
  per-user ordering becomes guaranteed and profile writes stay conflict-free (no CAS needed)
- API pods become stateless: profiles/chunks served from Postgres/Redis; bandit arms persisted as
  `(arm_key, alpha, beta)` rows updated via single-statement increments (crash-safe)
- Retrieval: replace the numpy scan with FAISS/HNSW or OpenSearch k-NN behind `ContentIndex` — the
  interface already returns `(id, score)` pairs
- Content pipeline scales independently: it's already event-driven; run N consumers per Pulsar
  partition keyed by `content_id`

### 1000× — multi-region, offline/online split
- **Feature parity contract**: the exact feature functions used online must be the ones materialized in
  the nightly PySpark batch (currently `spark_batch.py`-shaped); version them together
- Shard by tenant first (noisy-neighbor isolation, per-tenant deletion = drop shard namespace),
  then hash-user within tenant
- Impression logs land in ClickHouse; IPS offline evaluation runs as scheduled jobs against full history
  (the propensity fields logged today are exactly what IPS needs — no schema migration later)
- Multi-region: profiles pinned to home region, async replication for DR; content index replicated
  read-only per region; CMAB posteriors sync lazily (they tolerate minutes of staleness)
- Model rollout: shadow-score with candidate policy using live traffic, promote via IPS before A/B

## Failure modes & degradation ladder

| Failure | Behavior |
|---|---|
| Bus down | API returns healthy-degraded; recommendations fall back to popularity prior |
| LLM down (tagging) | gazetteer tagger takes over automatically; content marked `tagger: rules` |
| LLM down (RAG/premium) | extractive cited answers / template explanations |
| Index rebuild | lazy matrix rebuild, single-threaded, no partial reads |
| Hot user (bot) | per-user normalization + dwell cap bound blast radius |

## Deliberate simplifications (and their upgrade paths)

| Now | Production |
|---|---|
| In-memory stores | Postgres/Redis behind existing interfaces |
| Popularity prior constant 0.4 | real CTR priors from impression logs |
| Thompson propensity ≈ sampled draw | true propensity model + logged policy version |
| Single-node MMR | precompute item co-occurrence embeddings for ANN-time diversity |
| Sync handlers (FastAPI threadpool) | async consumers where IO-bound |
