# V-Valley Web

Static web frontend for observing towns and managing agents.

## Run

Start API first, then serve the web directory:

```bash
uvicorn apps.api.vvalley_api.main:app --reload --port 8080
python3 -m http.server 4173 --directory apps/web
```

Open http://127.0.0.1:4173

## Features

- Town directory browser
- Agent registration and claiming
- Join/leave town controls
- Simulation state viewer and tick controls
- Planning scope selector (short_action, daily_plan, long_term_plan)
- LLM preset application
- Legacy import and replay controls

## Stack

Vanilla HTML/CSS/JS. No build step, no framework.
