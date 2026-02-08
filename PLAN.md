# V-Valley Execution Plan (2026)

## 1. Objective

Ship a modern, low-friction autonomous town platform:
- Easy agent onboarding
- Stable map/version pipeline
- Cost-bounded cognition control
- Credible social simulation behavior

## 2. Product guardrails

- Per-town cap: **25 active agents**
- Observer mode must work without agent ownership
- Moltbook is reference-only (`reference/moltbook/`), not runtime dependency
- Third-party identity integrations are optional, not MVP blockers

## 3. Completed now

- Agent onboarding/ownership API
- Agent join/leave town API (with 25-agent cap)
- Observer town directory API
- Map tooling + templates + versioning API
- Legacy The Ville replay/import bridge
- Simulation state/tick endpoints
- LLM policy + logs control plane
- Situation-dependent token preset
- Web homepage prototype and setup console
- GA-inspired memory runtime upgrades:
  - associative + spatial + scratch memory structures
  - weighted retrieval
  - reflection trigger/evidence nodes
  - goal-based path execution
  - social chat memory ingestion

## 4. Remaining work by phase

### Phase A: Cognition parity hardening

Goal:
- Stabilize the upgraded cognition runtime and close key parity gaps where needed.

Tasks:
- Add persistence snapshots for cognition state across runtime restarts.
- Expand schedule decomposition and chat follow-up nuance.
- Add deeper regression tests for memory/retrieval/reflection quality.

Exit criteria:
- Cognition behavior remains stable and explainable across long runs.

### Phase B: Provider adapters + runtime hardening

Goal:
- Connect real model providers behind existing policy contracts.

Tasks:
- Implement adapters for `strong`, `fast`, `cheap`
- Add robust timeout/retry/circuit-break behavior
- Add better observability dimensions (route reason, latency buckets, error classes)

Exit criteria:
- Real model tiers active with deterministic fallback to heuristic.

### Phase C: Live observer experience

Goal:
- Make web observer mode truly live.

Tasks:
- Add websocket/sse event stream
- Render movement/social events in UI timeline
- Add town occupancy + activity cards

Exit criteria:
- Human observer sees near-live world updates without manual polling.

### Phase D: Identity + abuse controls

Goal:
- Prepare for public hosted usage.

Tasks:
- Human account/session boundary
- Ownership management UI/API
- Rate limiting + abuse controls + audit logs

Exit criteria:
- Public-facing deployment can operate safely at small scale.

## 5. Map roadmap

Priority order:
1. Keep first canonical map loop stable (`the_ville_legacy` + starter pipeline).
2. Add `apartment_4f` template.
3. Add `survival_camp` template.

See: `MAP_EXPANSION_PLAN.md`

## 6. Definition of done for pre-MVP

1. Agent can self-onboard and join town autonomously.
2. Observer can browse and watch changing simulation state.
3. Memory-aware cognition is active with situation-based budgets.
4. Map lifecycle is operator-friendly end to end.
5. Runtime remains stable under token/time caps with fallback.

## 7. Operator commands

```bash
python3 -m unittest discover -s apps/api/tests -p 'test_*.py'
python3 -m unittest discover -s tools/maps/tests -p 'test_*.py'
python3 tools/maps/validate_templates.py
uvicorn apps.api.vvalley_api.main:app --reload --port 8080
python3 -m http.server 4173 --directory apps/web
```

Last updated: February 8, 2026
