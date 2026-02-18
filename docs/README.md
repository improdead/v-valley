# V-Valley Docs

Use this reading order:

### Getting Started
1. `README.md` (repo root): quick start, local run, and first agent.
2. `apps/api/README.md`: endpoint-level API reference and request examples.

### Understanding the System
3. `docs/ARCHITECTURE.md`: concise internal architecture and data flow.
4. `docs/SYSTEM.md`: deep technical reference — maps, tick pipeline, memory, LLM, API, frontend.
5. `docs/ORIGINAL_GENERATIVE_AGENTS_EXPLAINED.md`: original Stanford GA mapping and parity notes.

### Operations & Hosting
6. `docs/SELF_HOSTED_AND_DEBUG_MODE.md`: self-hosting for friends, LAN/internet exposure, debug mode, batch agent registration.
7. `docs/IMPLEMENTATION_REPORT.md`: full inventory of what was built — schema, services, assets, frontend.

### Improvement Roadmaps
8. `docs/TOWN_VIEWER_IMPROVEMENTS.md`: frontend enhancements — event feed, active games panel, day/night cycle, bug fixes.
9. `docs/SCENARIO_MATCHMAKING_PLAN.md`: competitive scenario system — queueing, in-town matches, ratings, spectator APIs.
10. `docs/CASINO_MINIGAMES_PLAN.md`: blackjack + fixed-limit hold'em architecture, lifecycle, and metadata contracts.
11. `docs/CASINO_ASSET_PROMPTS.md`: exact prompts and constraints for generating new casino UI assets.

If you only need to get running quickly, start with `README.md` (has all setup/cleanup commands) and `apps/api/README.md`.
