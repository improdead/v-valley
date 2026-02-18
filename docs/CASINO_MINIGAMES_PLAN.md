# Casino Mini-Games Plan (Implemented)

## Scope

This document covers the casino expansion layered on top of existing scenarios.

Implemented scenarios:
- `werewolf_6p` (unchanged)
- `anaconda_standard` (unchanged)
- `blackjack_tournament` (new)
- `holdem_fixed_limit` (new)

Constraints preserved:
- No admin agent was introduced.
- Matches are formed from existing town members (25-agent town cap remains).
- Scenario participation is temporary; after match resolution/cancel/expiry, agents return to normal base-town simulation.

## Metadata Contract (`rules_json`)

Scenario definitions now use metadata-driven dispatch and UI rendering.

Required keys:
- `engine`: `werewolf|anaconda|blackjack|holdem`
- `ui_group`: `social|casino`
- `spectator_kind`: `werewolf|anaconda|blackjack|holdem`

The backend uses `rules_json.engine` as the primary dispatch key, with scenario-key fallback only when metadata is absent.

## Scenario Seeds

Default seeds now include:
- `blackjack_tournament`
- `holdem_fixed_limit`

Storage init behavior now also:
- Inserts missing default definitions even when `scenario_definitions` is already non-empty.
- Merges missing `rules_json` keys into existing definitions without overwriting existing configured values.

This prevents local DB resets when new scenarios are introduced.

## API Surface

No route paths were changed.

### `GET /api/v1/scenarios`
Returns seeded definitions with metadata under `rules_json`.

### `GET /api/v1/scenarios/servers`
Server rows now include:
- `scenario_name`
- `scenario_category`
- `scenario_kind`
- `ui_group`

Existing server fields are preserved.

### `GET /api/v1/scenarios/matches/{match_id}/spectate`
`public_state` now includes:
- `scenario_name`
- `scenario_kind`
- `scenario_category`
- `ui_group`

Plus engine-specific fields:
- Blackjack: `dealer_cards`, `dealer_upcard`, `player_hands`, `player_bets`, `hand_results`
- Hold'em: `board_cards`, masked/revealed `hole_cards`, button/SB/BB markers, `street_raise_count`, `current_bet`, contributions

## Engine Behavior

## Blackjack (`engine=blackjack`)

Default rules:
- `min_players: 2`, `max_players: 6`
- `buy_in: 50`
- `max_hands: 8`
- `min_bet: 5`, `max_bet: 25`
- `dealer_soft17_stand: true`
- `allow_double: true`, `allow_split: false`, `allow_insurance: false`

State machine:
- `deal -> player_turns -> dealer_turn -> settle` (loops per hand)

Completion:
- Ends after `max_hands` or bankroll-survivor condition.
- Winner set uses highest chip total (ties allowed).

## Hold'em (`engine=holdem`)

Default rules:
- `min_players: 2`, `max_players: 6`
- `buy_in: 100`
- `small_blind: 5`, `big_blind: 10`
- `bet_units: [10,10,20,20]`
- `max_raises_per_street: 3`
- `hands_per_match: 1`

State machine:
- `preflop -> flop -> turn -> river -> showdown`

Rules implemented:
- Fixed-limit actions: `fold|check|call|raise`
- Side-pot resolution reused from existing card resolver
- Early finish when all but one player fold/forfeit

## Cognition Task Extensions

Added planner tasks:
- `blackjack_choose_bet`
- `blackjack_choose_action`
- `holdem_choose_action`

Added provider support:
- Default task policies
- JSON output schema contracts
- Task-specific prompt templates

Heuristic fallbacks remain active if LLM is unavailable.

## Frontend Integration

Landing page (`apps/web`):
- Scenario list now groups by metadata:
  - Social Deduction
  - Mini-Games / Casino
- Server cards now use metadata labels (`scenario_name`, `scenario_kind`) instead of hardcoded key substring checks.

Town spectator (`town.js`):
- New board renderers:
  - `renderBlackjackBoard(state)`
  - `renderHoldemBoard(state)`
- Existing werewolf/anaconda board renderers preserved.
- Modal title and phase tracking now use metadata (`scenario_name` / `scenario_kind`).

## Match Lifecycle and Cleanup

User-visible flow:
1. User opens website.
2. User sees available live servers and scenario cards.
3. User joins queue for a scenario.
4. Match forms in the agent's current town.
5. Agent is temporarily marked with scenario status and assigned scenario context.
6. Match runs and resolves.
7. Server disappears from active list when match is no longer active.
8. Scenario status flags are cleaned; agent resumes normal town simulation.

This lifecycle is the same model for werewolf, anaconda, blackjack, and hold'em.

## Asset Plan

Reuse:
- Processed Anaconda card/chip/table assets are reused for casino board rendering.

New required source assets (for custom art):
- `apps/web/assets/scenarios/blackjack/icon.png`
- `apps/web/assets/scenarios/blackjack/ui/actions_sheet.png`
- `apps/web/assets/scenarios/holdem/icon.png`
- `apps/web/assets/scenarios/holdem/ui/markers_sheet.png`

Pipeline updates:
- `scripts/extract_scenario_sheet_assets.py` now includes optional extraction specs for these files.
- `scripts/build_scenario_assets.py` now includes deterministic `blackjack` and `holdem` fallback packs.
- Processed outputs target:
  - `apps/web/assets/scenarios/processed/blackjack/...`
  - `apps/web/assets/scenarios/processed/holdem/...`

## Validation

Automated API coverage now includes:
- `test_scenarios_list_includes_blackjack_and_holdem`
- `test_blackjack_queue_match_wallet_and_cleanup`
- `test_holdem_queue_match_wallet_and_cleanup`
- `test_servers_endpoint_labels_are_metadata_driven`
- `test_spectate_payload_contains_blackjack_public_fields`
- `test_spectate_payload_contains_holdem_public_fields`

Existing werewolf/anaconda tests remain green.
