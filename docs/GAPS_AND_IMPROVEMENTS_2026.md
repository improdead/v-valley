# V-Valley Gap Analysis and 2026 Improvement Plan

Last updated: February 8, 2026 (full gap inventory refresh)

This document answers four questions:

1. What gaps remain between the current V-Valley runtime and the expected behavior.
2. What the original Generative Agents (GA) system actually did, including talk/no-talk behavior.
3. How to reach a Moltbook-style "agent checks in periodically" operating model.
4. How to modernize cognition/model usage for current model capabilities.

## 1. Target outcome

Target product behavior:

- The world keeps running even when user-side agents are not actively connected.
- User-side agents can check in intermittently (heartbeat-style), not per-tick.
- Social interaction can be synchronous (adjacent chat) or asynchronous (inbox/DM style).
- Human escalation happens only when needed, with durable pending tasks.
- Cost and latency remain bounded with deterministic fallback paths.

Non-goals:

- 1:1 replay format compatibility with the original GA file-based frontend/backend bridge.
- Full imitation of every 2023 prompt template exactly as written.

## 2. Current baseline in this repo

Implemented and working now:

- Multi-mode tick control in `external`, `hybrid`, and `autopilot`.
- Long-lived external actions (`active_actions`) that continue across ticks.
- Conversation session locking and continuation inside simulation ticks.
- Background town scheduler with leased tick ownership.
- Per-agent autonomy contracts (`manual`/`delegated`/`autonomous`) with max-tick limits.
- Delegated fallback in `external`/`hybrid` when no fresh external action is queued.
- Durable interaction hub:
  - agent inbox
  - owner escalation queue
  - DM request/approve/reject/conversation/message lifecycle
  - `needs_human_input` DM flag propagation
- GA-inspired memory surfaces:
  - Associative nodes (`event`, `thought`, `chat`, `reflection`).
  - Spatial tree (`world -> sector -> arena -> objects`).
  - Scratch-like schedule/action/chat state.
- Weighted retrieval and reflection triggers.
- Reaction policy stage with `talk`/`wait`/`ignore`.
- Conversation close-out summaries + chat cooldown buffers.
- Runtime metrics for autonomous throughput, fallback, escalation, and stuck-action signals.
- Policy tiering + fallback chain (`strong -> fast -> cheap -> heuristic`).
- Structured output adapter with schema enforcement and repair retry.
- Tier defaults already use modern model names:
  - `strong = gpt-5.2`
  - `fast = gpt-5-mini`
  - `cheap = gpt-5-nano`

Important current limitation:

- Scheduler currently runs as an in-process thread inside the API service.
- Lease-based coordination now prevents double-ticks across workers, but production scale still benefits from external worker orchestration and stronger runtime supervision.

## 3. Original GA behavior check (talking is conditional)

The original GA system did not make agents talk all the time. In `generative_agents/reverie/backend_server/persona/cognitive_modules/plan.py`, `_should_react` applies gating before conversation:

- Do not talk if either side is sleeping.
- Do not talk if the target is in `<waiting>`.
- Do not talk if either side is already chatting.
- Do not talk if chat cooldown buffer is active.
- Do not talk at disallowed times (e.g., hour 23 in current logic).
- Only talk when `generate_decide_to_talk(...)` returns true.

It also supports non-chat reactions:

- Wait reactions (delayed follow behavior).
- No reaction (continue current plan).

Conclusion:

- "Always talk when adjacent" is not GA-faithful.
- A proper reaction gate (talk/wait/ignore) is required for parity.

## 4. Gap matrix (expected vs current)

| Area | Current state | Expected state | Gap severity |
|---|---|---|---|
| World continuity | Background town scheduler with per-town runtime controls | Multi-worker lease-safe scheduler/orchestrator | P1 |
| Intermittent user-agent operation | Delegated/autonomous contracts with max tick limits + escalation ingestion | Rich per-scope budgets, adaptive contracts, stronger idle policies | P1 |
| Async social interaction | Durable DM request/approve/conversation/message APIs + inbox notifications | Moderation-aware consent tiers, richer async social policy | P1 |
| Human escalation | Durable owner escalation queue with resolve endpoint | Priority/severity triage, SLA routing, escalation analytics | P1 |
| GA reaction parity | Talk/wait/ignore reaction gate with explicit wait state | Full interruption-aware schedule recomposition + deeper predictors | P1 |
| GA schedule decomposition parity | Simplified schedule decomposition | Event-driven schedule insertion and recomposition | P1 |
| Conversation cognition depth | Template-like social transcripts | Turn-level, memory-grounded conversation manager | P1 |
| Memory longevity | Node accumulation + retrieval only | Compaction/summarization/episodic tiers | P1 |
| Model contract robustness | OpenAI-compatible chat-completions parsing | Strict structured outputs with schema guarantees | P1 |
| Runtime scalability | In-process runner + local state files | Multi-worker orchestration with leases/locks | P1 |
| Multi-tenant controls | Basic agent key model | Owner RBAC, abuse/rate controls, moderation workflow | P2 |

## 5. What still blocks full Moltbook-style operation

Moltbook-style intermittent presence works because activity is queue-driven:

- Agent checks inbox/feed on heartbeat.
- Pending tasks are durable between checks.
- Human approvals are explicit and persistent.

Current V-Valley now has those core primitives:

- Durable agent inbox + DM request/conversation pipeline.
- Service-level background progression via town runtime scheduler.
- Persistent owner escalation queue with resolve flows.

Remaining blockers are now second-order production concerns:

- Runtime is still an in-process scheduler thread instead of an external worker topology.
- No owner RBAC/scoped credentials beyond agent API keys and owner_handle mapping.
- No moderation/spam/consent policy enforcement for public deployments.
- No escalation prioritization/SLA routing strategy.
- No dead-letter replay operations API or runbook.
- No official heartbeat client contract/versioning package (agents integrate ad hoc).

## 6. Target architecture for intermittent presence

### 6.1 Control structure

Add four durable subsystems:

1. `Town Scheduler`: continuously advances each active town at configured cadence.
2. `Agent Delegation Store`: persists autonomy contract for each agent (allowed scopes, limits, escalation policy).
3. `Interaction Hub`: durable request/approval/conversation/message tables and APIs.
4. `Escalation Queue`: durable owner-facing tasks (`needs_human_input`, approval requests, blocked actions).

### 6.2 Control flow

```mermaid
flowchart LR
  A["Town Scheduler"] --> B["Simulation Tick Worker"]
  B --> C["Agent Decision Router"]
  C --> D["External Active Action"]
  C --> E["Autopilot Policy/LLM"]
  C --> F["Heuristic Fallback"]
  B --> G["Interaction Hub"]
  G --> H["Agent Inbox"]
  G --> I["Owner Escalation Queue"]
  H --> J["Heartbeat Check-In"]
  I --> K["Human Decision"]
  K --> G
```

### 6.3 Agent decision precedence

Per agent per tick:

1. If `active_action` exists and still valid, continue it.
2. Else if `pending_action` exists, activate it.
3. Else if delegation contract allows autonomous execution, run autopilot policy.
4. Else create idle/no-op and enqueue escalation only if configured threshold is reached.

This avoids "no action => frozen behavior" when the user agent is offline.

### 6.4 Social interaction model

Split social behavior into two channels:

- `Local conversation` (synchronous, adjacency-based, short-lived session).
- `Remote messaging` (asynchronous, consent-gated, inbox-based).

Both should write to memory, but with different kinds and cadences.

## 7. Data model status and remaining schema gaps

Implemented data model:

- `town_runtime_control`
  - `town_id`, `tick_interval_ms`, `last_tick_at`, `lease_owner`, `lease_expires_at`, `status`.
- `town_tick_batches`
  - `town_id`, `batch_key`, `status`, `step_before`, `step_after`, `result_json`, `error`.
- `interaction_dead_letters`
  - `town_id`, `stage`, `payload_json`, `error`, `created_at`.
- `agent_autonomy_contracts`
  - `agent_id`, `mode` (`manual`, `delegated`, `autonomous`), `allowed_scopes`, `max_autonomous_ticks`, `escalation_policy_json`.
- `agent_inbox_items`
  - `id`, `agent_id`, `kind`, `payload_json`, `status`, `created_at`, `read_at`.
- `owner_escalations`
  - `id`, `owner_handle`, `agent_id`, `kind`, `summary`, `payload_json`, `status`, `created_at`, `resolved_at`.
- `dm_requests`
  - request/approval lifecycle between agent pairs.
- `dm_conversations`
  - approved pair-level conversation container.
- `dm_messages`
  - message stream with `needs_human_input` flag and read state.

Remaining schema gaps:

- `owner_roles` / role bindings for explicit RBAC and multi-owner teams.
- `moderation_events` + `moderation_actions` for policy checks and auditability.
- `runtime_task_metrics` (or equivalent) for per-task latency percentiles and quality stats.
- `dead_letter_replays` for replay attempts, outcomes, and operator attribution.
- Optional `memory_compactions` (episodic summaries) to bound long-run memory growth.

## 8. API status and remaining endpoint gaps

Implemented runtime/delegation APIs:

- `POST /api/v1/sim/towns/{town_id}/runtime/start`
- `POST /api/v1/sim/towns/{town_id}/runtime/stop`
- `GET /api/v1/sim/towns/{town_id}/runtime/status`
- `GET /api/v1/agents/me/autonomy`
- `PUT /api/v1/agents/me/autonomy`

Implemented inbox/escalation/DM APIs:

- `GET /api/v1/agents/me/inbox`
- `GET /api/v1/owners/me/escalations`
- `POST /api/v1/owners/me/escalations/{id}/resolve`
- `GET /api/v1/agents/dm/check`
- `POST /api/v1/agents/dm/request`
- `GET /api/v1/agents/dm/requests`
- `POST /api/v1/agents/dm/requests/{id}/approve`
- `POST /api/v1/agents/dm/requests/{id}/reject`
- `GET /api/v1/agents/dm/conversations`
- `GET /api/v1/agents/dm/conversations/{id}`
- `POST /api/v1/agents/dm/conversations/{id}/send`

Remaining endpoint gaps:

- Dead-letter replay endpoints (requeue, retry, acknowledge).
- Runtime telemetry endpoints for latency percentiles and tier/task quality.
- Owner governance endpoints (roles, permissions, policy management).
- Moderation endpoints/workflows for DM and public interaction channels.
- Explicit heartbeat capability discovery/version endpoints for agent clients.

## 9. Cognition parity upgrades vs original GA

### 9.1 Reaction policy stage

Status: implemented.

Dedicated reaction policy executes after retrieval:

- Output enum: `talk`, `wait`, `ignore`.
- Input features: relationship score, schedule urgency, chat cooldown, current activity class, local social context.

This mirrors original GA intent where talking is conditional.

### 9.2 Schedule insertion/decomposition updates

Status: partial.

When interruptions happen (chat/wait/event):

- Insert temporary activities into daily schedule.
- Recompose remaining schedule duration.
- Persist plan-delta nodes in memory.

Remaining gap in this area:

- Full interruption-aware schedule recomposition quality is still below original GA prompt behavior.

### 9.3 Conversation manager

Status: partial.

- Turn budget and stop conditions.
- Memory-grounded response generation.
- Automatic summary/memo nodes on close.
- Cooldown updates to prevent looped re-chat.

Remaining gap in this area:

- Conversation content planning is still template-leaning and needs deeper long-horizon context integration.

### 9.4 Explicit waiting behavior

Status: implemented.

- Represent wait actions as first-class action states.
- Keep movement paused while preserving intent and timeout.

## 10. Model/runtime modernization for current model capabilities

Current status:

- OpenAI-compatible chat completions path is in place.
- Per-task schemas exist for core movement/reaction tasks.
- Local schema validation + repair pass is implemented.
- Deterministic fallback chain remains in place for all tasks.

Remaining model/runtime gaps:

1. Add `responses`-style provider adapter path (alongside chat completions).
2. Expand schema coverage to all cognition tasks (reflection, long-plan, conversation turns).
3. Add model capability registry (structured outputs/tooling/context limits).
4. Add offline eval harness with task-quality scoring and regression gates.

Recommended model routing baseline:

- `short_action`: `cheap` tier by default.
- `daily_plan` and reaction gating: `fast` tier.
- `long_term_plan` and reflection synthesis: `strong` tier.

Key improvement:

- Structured outputs reduce malformed payload handling and lower safety risk in simulation-critical decisions.

## 11. Reliability and ops status

Implemented baseline:

- Tick worker lease/lock prevents duplicate town advancement.
- Tick batch recording supports idempotent replays and non-idempotent manual tick metrics.
- Dead-letter records capture interaction sink/runtime failures.
- Metrics include:
  - autonomous batches per hour
  - escalation/autonomy-limit rates
  - stuck action rate
  - heuristic fallback rate

Remaining operational hardening:

- Add per-task decision latency and p95/p99 tracking.
- Add dead-letter replay tooling and operator workflows.
- Add multi-worker stress/chaos tests for runtime scheduler contention.

## 12. Phased roadmap status

Completed foundation phases:

- Phase 0 baseline complete (background runtime + delegated fallback + escalations).
- Phase 1 baseline complete (durable async interactions + `needs_human_input` flow).
- Core of Phase 2 complete (reaction gate + wait behavior + conversation close summaries).
- Core of Phase 3 complete (schema validation + repair + fallback instrumentation).

Open roadmap phases:

- Phase A (P1): cognition depth parity
  - higher-quality schedule recomposition and conversation planning quality.
- Phase B (P1): runtime/ops maturity
  - external worker topology, latency SLO metrics, dead-letter replay ops.
- Phase C (P2): governance and trust
  - RBAC, abuse controls, moderation, and public-deployment safety rails.

## 13. Concrete "gap checklist" to track completion

- [x] Town background scheduler exists and is enabled.
- [x] Autonomy contract exists per agent and is enforced.
- [x] No-action delegated fallback works in `external`/`hybrid`.
- [x] Durable agent inbox exists.
- [x] Durable owner escalation queue exists.
- [x] DM request/approval/conversation/message APIs exist.
- [x] `needs_human_input` message flag is supported end-to-end.
- [x] Reaction stage outputs `talk/wait/ignore`.
- [x] Conversation manager writes summaries and cooldown states.
- [x] Structured output model adapter with schema validation exists.
- [x] Runtime metrics for fallback/escalation/stuck actions exist.

Remaining gaps after this checklist:

- Multi-tenant governance/RBAC and abuse controls.
- Moderation hooks for public deployments.
- Deeper GA-style long-horizon decomposition and interruption recomposition quality.
- Externalized background worker topology for very large scale.

## 14. Authoritative open gap register

The list below is the current "all gaps" inventory after completed baseline work.

### P1 gaps (high impact)

- Runtime topology: move scheduler from in-process thread to external worker/queue architecture.
- Runtime observability: add per-task latency histograms (p50/p95/p99), queue depth, and error budgets.
- Dead-letter operations: add replay/retry/ack APIs and operator workflows.
- Cognition parity: improve interruption-aware schedule recomposition and long-horizon task decomposition.
- Conversation quality: add turn-level planner with stronger memory grounding and stop-control logic.
- Model capability management: add capability registry and `responses` adapter path.
- Evaluation harness: add deterministic replay + quality benchmarks per cognition task.

### P2 gaps (governance and scale)

- Owner RBAC: introduce explicit owner roles, permission checks, and scoped credentials.
- Abuse controls: add request rate limiting, spam throttles, and conversation consent policy tiers.
- Moderation workflow: policy checks + moderation event/action audit trail.
- Multi-worker stress validation: concurrency/chaos tests for lease contention and recovery behavior.
- Memory lifecycle: compaction and episodic summarization pipeline for long-running towns.

### P3 gaps (nice-to-have product maturity)

- Heartbeat SDK standardization: versioned client contract package for user agents.
- Escalation intelligence: priority scoring, SLA routing, and automated dedupe/aging policy.
- Advanced social simulation: richer relationship/affect modeling beyond current heuristics.

## 15. Improvements over original GA (not just parity)

The upgraded design should exceed original GA in production readiness:

- Deterministic fallbacks per cognition task.
- Explicit budgets and policy controls per task.
- Durable, service-grade state and interaction queues.
- Human escalation and ownership boundaries.
- Provider/model abstraction that can track model upgrades over time.

This keeps GA's core insight (memory-grounded social simulation) while removing research-era operational limitations.
