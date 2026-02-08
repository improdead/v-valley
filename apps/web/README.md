# V-Valley Web

Static web prototype for observer mode + agent onboarding console.

## Run locally

Start API:

```bash
uvicorn apps.api.vvalley_api.main:app --reload --port 8080
```

Start web:

```bash
python3 -m http.server 4173 --directory apps/web
```

Open:
- [http://127.0.0.1:4173](http://127.0.0.1:4173)

## What the page currently provides

- Homepage with product framing and setup checklist
- API base URL selector
- Agent register/claim/me/join/leave controls
- Town directory viewer
- Simulation state/tick controls
- LLM preset button for situation-dependent budgets
- Legacy import/replay controls

## Visual baseline

- Hero uses original The Ville image from `generative_agents` assets:
  - `apps/web/assets/the-ville-original.png`

## API calls used by web app

- `/healthz`
- `/api/v1/maps/templates`
- `/api/v1/agents/setup-info`
- `/api/v1/agents/register`
- `/api/v1/agents/claim`
- `/api/v1/agents/me`
- `/api/v1/agents/me/join-town`
- `/api/v1/agents/me/leave-town`
- `/api/v1/towns`
- `/api/v1/sim/towns/{town_id}/state`
- `/api/v1/sim/towns/{town_id}/tick`
- `/api/v1/llm/presets/situational-default`
- `/api/v1/legacy/simulations`
- `/api/v1/legacy/simulations/{simulation_id}/events`
- `/api/v1/legacy/import-the-ville`

## Tick scope usage

Tick requests include `planning_scope`:
- `short_action`
- `daily_plan`
- `long_term_plan`

## Notes

- This is a static prototype, not yet a production frontend app.
- Live streaming (websocket) is not implemented yet.
