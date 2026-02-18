# V-Valley Implementation Report

> **What was planned, what was built, and how everything fits together.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scenario Matchmaking System](#2-scenario-matchmaking-system)
   - [Database Schema](#21-database-schema)
   - [Backend — Matchmaker Service](#22-backend--matchmaker-service)
   - [Backend — API Endpoints](#23-backend--api-endpoints)
   - [Core Simulation — Scenario Runtime](#24-core-simulation--scenario-runtime)
   - [Scenario 1: Werewolf (Social Deduction)](#25-scenario-1-werewolf-social-deduction)
   - [Scenario 2: Anaconda Poker (Card Game)](#26-scenario-2-anaconda-poker-card-game)
   - [Elo Rating System](#27-elo-rating-system)
   - [Wallet & Economy](#28-wallet--economy)
   - [Casino Mini-Games Expansion (Blackjack + Hold'em)](#29-casino-mini-games-expansion-blackjack--holdem)
3. [Generated Assets](#3-generated-assets)
   - [Werewolf Assets](#31-werewolf-assets)
   - [Anaconda Poker Assets](#32-anaconda-poker-assets)
   - [Shared / Rating Assets](#33-shared--rating-assets)
   - [Pixel-Art Atlas Pipeline](#34-pixel-art-atlas-pipeline)
   - [Sprite Sheet Extraction Pipeline](#35-sprite-sheet-extraction-pipeline)
4. [Town Viewer & Frontend](#4-town-viewer--frontend)
   - [Spectator UI — Werewolf Board](#41-spectator-ui--werewolf-board)
   - [Spectator UI — Anaconda Board](#42-spectator-ui--anaconda-board)
   - [Match Modal & Replay System](#43-match-modal--replay-system)
   - [Landing Page — Scenario Integration](#44-landing-page--scenario-integration)
   - [Live Event Feed & Speech Bubbles](#45-live-event-feed--speech-bubbles)
   - [Day/Night Cycle & Map Interactions](#46-daynight-cycle--map-interactions)
   - [Performance & Technical Fixes](#47-performance--technical-fixes)
5. [Tooling & Scripts](#5-tooling--scripts)
6. [Architecture Diagram](#6-architecture-diagram)
7. [File Inventory](#7-file-inventory)

---

## 1. Project Overview

V-Valley is a generative-agent simulation platform where AI agents live in pixel-art towns, form relationships, hold conversations, and now compete in **structured competitive scenarios**. The project extends the original Stanford "Generative Agents" research with:

- A persistent town runtime with autonomous agent cognition
- A full-stack API (FastAPI) + web viewer (Phaser 3)
- Four competitive scenario modes: **Werewolf**, **Anaconda Poker**, **Blackjack Tournament**, and **Texas Hold'em (Fixed Limit)**
- A CS2 Premier-inspired Elo matchmaking system
- A complete pixel-art asset pipeline for scenario spectator UIs

---

## 2. Scenario Matchmaking System

### 2.1 Database Schema

**File:** `packages/vvalley_core/db/migrations/0010_scenarios.sql`

Six new tables were created:

| Table | Purpose |
|---|---|
| `scenario_definitions` | Config for each game type (min/max players, buy-in, rules JSON, warmup steps, max duration) |
| `agent_scenario_ratings` | Per-agent per-scenario Elo rating, wins/losses/draws, peak rating. Composite PK `(agent_id, scenario_key)` |
| `agent_wallets` | Persistent currency for betting games (balance, total earned/spent, daily stipend tracking) |
| `scenario_queue_entries` | Active matchmaking queue with rating snapshot, party support, and status tracking |
| `scenario_matches` | Match lifecycle — status (`forming` → `warmup` → `active` → `resolved`/`cancelled`), phase, round, and full state JSON |
| `scenario_match_participants` | Role assignment, chip tracking, rating deltas, per-player match results |
| `scenario_match_events` | Event stream for spectator replay — every vote, kill, bet, reveal, and showdown is recorded |

Indexes are optimized for:
- Town-scoped active match queries
- Agent queue lookups
- Leaderboard ranking queries (scenario + rating DESC)
- Event replay (match + step DESC)

### 2.2 Backend — Matchmaker Service

**File:** `apps/api/vvalley_api/services/scenario_matchmaker.py` (~1,700 lines)

The matchmaker is the core orchestration engine. Key responsibilities:

- **Queue Management:** `join_queue()` / `leave_queue()` — validates wallet balance, checks for active matches, snapshots rating at queue time
- **Match Formation:** `form_matches()` — when enough players are queued, creates a match, assigns roles/seats, deducts buy-ins, and notifies agents via inbox
- **Match Advancement:** `advance_matches()` — called each simulation tick to progress active matches through their phase sequences
- **Werewolf Flow:** Role assignment (distribution scales with player count), night actions (kill/protect/investigate), day discussion/voting, eliminations
- **Anaconda Flow:** 7-card deal, three passing rounds (3→2→1 cards), discard to best 5, five reveal rounds with betting, showdown with side-pot resolution
- **Match Finalization:** `_finalize_match()` — computes Elo updates, settles wallet payouts, records result JSON, notifies participants
- **Forfeit/Cancel:** `forfeit_match()` / `cancel_match()` — graceful early termination with wallet refund logic
- **Spectator Payloads:** `get_spectator_payload()` / `get_match_payload()` — sanitized public state (hides unrevealed cards/roles)
- **Cognition Integration:** LLM-driven agent decision-making for votes, card passes, and betting actions
- **Memory Recording:** `_record_scenario_memory()` — writes scenario events into agent memory for future behavior influence

### 2.3 Backend — API Endpoints

**File:** `apps/api/vvalley_api/routers/scenarios.py`

Full REST API under `/api/v1/scenarios/`:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | List all enabled scenarios |
| `/{scenario_key}` | GET | Scenario details |
| `/{scenario_key}/queue/join` | POST | Join matchmaking queue (Bearer auth) |
| `/{scenario_key}/queue/leave` | POST | Leave queue |
| `/{scenario_key}/queue/status` | GET | Queue depth and estimated wait |
| `/matches` | GET | List matches (filterable by town/scenario) |
| `/matches/{match_id}` | GET | Full match detail |
| `/matches/{match_id}/events` | GET | Event stream for replay |
| `/matches/{match_id}/state` | GET | Current public state |
| `/matches/{match_id}/spectate` | GET | Spectator-safe payload |
| `/matches/{match_id}/forfeit` | POST | Player forfeit (Bearer auth) |
| `/matches/{match_id}/cancel` | POST | Cancel match (Bearer auth) |
| `/me/queue` | GET | Agent's queue status |
| `/me/match` | GET | Agent's active match |
| `/me/ratings` | GET | Agent's ratings across all scenarios |
| `/me/wallet` | GET | Agent's wallet balance |
| `/ratings/{scenario_key}` | GET | Leaderboard |
| `/servers` | GET | Live match servers |
| `/towns/{town_id}/active` | GET | Active matches in a town |
| `/towns/{town_id}/advance` | POST | Manual match advancement (admin) |

### 2.4 Core Simulation — Scenario Runtime

**Files:**
- `packages/vvalley_core/sim/scenarios/manager.py` — `TownScenarioManager` class
- `packages/vvalley_core/sim/scenarios/rating.py` — Elo computation
- `packages/vvalley_core/sim/scenarios/cards.py` — Poker hand evaluation
- `packages/vvalley_core/sim/scenarios/__init__.py` — Package init

The `TownScenarioManager` integrates with the simulation runner (`runner.py`):
- On each town tick, it calls `form_matches()` → `advance_matches()`
- Returns a `ScenarioRuntimeState` snapshot with formed/advanced/active matches
- Provides `statuses_for_town()` to flag agents as `in_scenario` (suppresses normal planning)

### 2.5 Scenario 1: Werewolf (Social Deduction)

**Format:** 6–10 players, hidden roles, alternating night/day phases.

**Roles & Distribution:**
| Player Count | Werewolves | Seer | Doctor | Hunter | Villagers |
|---|---|---|---|---|---|
| ≤6 | 2 | 1 | — | — | remainder |
| 7–8 | 2 | 1 | 1 | — | remainder |
| 9 | 3 | 1 | 1 | — | 4 |
| 10+ | 3 | 1 | 1 | 1 | remainder |

**Phase Sequence:** `night` → `day` → `night` → `day` → ...

**Mechanics Implemented:**
- Night: Werewolves vote to kill, Seer investigates, Doctor protects
- Day: LLM-powered discussion, accusation voting, majority elimination
- Win conditions: All werewolves eliminated (village wins) or werewolves ≥ villagers (wolves win)
- Memory integration: Agents remember accusations, defenses, and outcomes

### 2.6 Scenario 2: Anaconda Poker (Card Game)

**Format:** 3–7 players, persistent wallet economy.

**Phase Sequence:** `deal` → `pass1` → `pass2` → `pass3` → `discard` → `reveal1` → `reveal2` → `reveal3` → `reveal4` → `reveal5` → `showdown`

**Mechanics Implemented:**
- 7-card deal from a seeded shuffled deck (`make_shuffled_deck()`)
- Three passing rounds (3→2→1 cards) with AI card selection (`choose_pass_cards()`)
- Discard to best 5 hand (`evaluate_best_five()` — checks all 21 combinations)
- Card-by-card reveal with betting at each reveal round
- Side-pot resolution for all-in scenarios (`_anaconda_resolve_side_pots()`)
- Full poker hand evaluation: High Card through Straight Flush (9 categories)
- Wallet deduction at match start, payout at resolution

**Poker Hand Evaluator:** `packages/vvalley_core/sim/scenarios/cards.py`
- `evaluate_five()` — Scores any 5-card hand (category + tiebreaker tuple)
- `evaluate_best_five()` — Finds optimal 5 from 7+ cards
- `compare_hands()` — Head-to-head comparison
- `arrange_reveal_order()` — Orders cards for dramatic reveal sequence
- `hand_category_label()` — Human-readable labels ("Full House", "Flush", etc.)

### 2.7 Elo Rating System

**File:** `packages/vvalley_core/sim/scenarios/rating.py`

CS2 Premier-inspired implementation:
- **Starting Rating:** 1500
- **K-Factor:** K=40 for first 10 games (placement), K=20 after
- **Pairwise Computation:** Each player is compared against every other player using the standard Elo expected-score formula: `E = 1 / (1 + 10^((R_other - R_self) / 400))`
- **Normalization:** Deltas are divided by `(N-1)` to keep multi-player updates stable
- **Tracking:** Rating, games played, wins, losses, draws, and peak rating are all persisted

**Rating Tiers (displayed in UI):**
| Tier | Rating |
|---|---|
| Starter | < 1300 |
| Bronze | 1300–1499 |
| Silver | 1500–1699 |
| Gold | 1700–1899 |
| Platinum | 1900–2099 |
| Diamond | 2100+ |

### 2.8 Wallet & Economy

- Every agent starts with 1,000 chips
- Buy-in is deducted when a match forms
- Winnings are paid out at match resolution
- Daily stipend of 100 chips if balance drops below 50 (once per UTC day)
- Cancelled matches refund buy-ins proportionally

### 2.9 Casino Mini-Games Expansion (Blackjack + Hold'em)

The scenario stack was expanded with two additional casino engines without introducing any admin agent:

- `blackjack_tournament` (`engine=blackjack`, `ui_group=casino`, `spectator_kind=blackjack`)
- `holdem_fixed_limit` (`engine=holdem`, `ui_group=casino`, `spectator_kind=holdem`)

Implementation highlights:

- Engine-based dispatch in `scenario_matchmaker.py` now routes werewolf/anaconda/blackjack/holdem from `rules_json.engine` instead of brittle key substring branching.
- Default scenario seed bootstrapping now inserts missing scenario definitions into non-empty DBs and merges missing metadata keys into existing `rules_json` rows.
- `/api/v1/scenarios/servers` now returns metadata-driven fields:
  - `scenario_name`
  - `scenario_category`
  - `scenario_kind`
  - `ui_group`
- `/api/v1/scenarios/matches/{match_id}/spectate` now includes:
  - top-level `scenario_name` and `scenario_kind`
  - blackjack-specific public state (`dealer_cards`, `dealer_upcard`, `player_hands`, `player_bets`, `hand_results`)
  - hold'em-specific public state (`board_cards`, masked/revealed `hole_cards`, button/blind markers, bet state)
- Frontend scenario cards are grouped by metadata:
  - Social Deduction group
  - Mini-Games / Casino group
- Town spectator modal now renders dedicated boards for blackjack and hold'em, while preserving existing werewolf/anaconda boards.
- Match cleanup remains town-native:
  - agents are flagged `scenario_*` only during active/warmup participation
  - after resolve/cancel/expiry/forfeit, flags are removed and agents continue base simulation in their town.

Validation coverage added in `apps/api/tests/test_scenarios_api.py`:

- `test_scenarios_list_includes_blackjack_and_holdem`
- `test_blackjack_queue_match_wallet_and_cleanup`
- `test_holdem_queue_match_wallet_and_cleanup`
- `test_servers_endpoint_labels_are_metadata_driven`
- `test_spectate_payload_contains_blackjack_public_fields`
- `test_spectate_payload_contains_holdem_public_fields`

---

## 3. Generated Assets

All scenario art follows a **cozy indie pixel-art style** targeting a 16-bit RPG aesthetic.

### 3.1 Werewolf Assets

**Location:** `apps/web/assets/scenarios/werewolf/`

| Asset | File(s) | Description |
|---|---|---|
| Role Cards | `cards/role_villager.png`, `role_werewolf.png`, `role_seer.png`, `role_doctor.png`, `role_hunter.png`, `role_witch.png`, `role_alpha.png` | 7 unique role card illustrations |
| Campfire Board | `board/campfire_circle.png` | The central gathering spot / game board |
| Phase Tokens | `ui/phase_day_night.png` | Day/night phase indicator (split sheet, extracted to `phase_day.png` + `phase_night.png`) |
| Tombstone | `ui/tombstone.png` | Eliminated player marker |
| Vote Tokens | `ui/vote_tokens.png` | 4-up sheet (accuse, defend, skip, save — extracted individually) |
| Scenario Icon | `icon.png` | Werewolf scenario icon |
| Atlas | `atlas.png` + `atlas.json` | Programmatic 32px pixel-art atlas (10 frames) |

**Processed outputs:** `apps/web/assets/scenarios/processed/werewolf/`
- All raw sheets are cropped, background-removed, and trimmed to transparent PNGs

### 3.2 Anaconda Poker Assets

**Location:** `apps/web/assets/scenarios/anaconda/`

| Asset | File(s) | Description |
|---|---|---|
| Card Back | `cards/card_back.png` | Pixel-art card back design |
| Card Faces | `cards/spades_sheet.png`, `diamonds_sheet.png`, `clubs_sheet.png`, `hearts_sheet.png` | Full 52-card deck as 4 suit sprite sheets |
| Poker Table | `board/poker_table.png` | Top-down felt table |
| Poker Chips | `chips/poker_chips.png` | Chip denomination sprite sheet |
| Betting Buttons | `ui/betting_buttons.png` | 4-up sheet (fold, call, raise, all-in — extracted individually) |
| Hand Ranks | `ui/hand_ranks.png` | 10-row reference strip (High Card through Royal Flush — extracted individually) |
| Scenario Icon | `icon.png` | Anaconda scenario icon |
| Atlas | `atlas.png` + `atlas.json` | Programmatic 32px pixel-art atlas (10 frames) |

**Processed outputs:** `apps/web/assets/scenarios/processed/anaconda/`
- 52 individual card face PNGs (`faces/2S.png` through `AH.png`)
- Individual betting button PNGs
- 10 individual hand rank PNGs
- Cropped chip denominations, card back, table, and icon

### 3.3 Shared / Rating Assets

**Location:** `apps/web/assets/scenarios/shared/`

| Asset | File(s) | Description |
|---|---|---|
| Rating Badges | `frames/rating_badges.png` | 6-up badge strip (Starter through Diamond) |
| Queue Panel | `frames/queue_panel.png` | Matchmaking queue UI frame |
| Emote Sprites | `frames/thinking.png`, `winning.png`, `losing.png`, `spectate.png` | Emotion/state indicators |
| Individual Badges | `frames/rating_bronze.png` through `rating_diamond.png` | Pre-split badge PNGs |
| Atlas | `atlas.png` + `atlas.json` | Programmatic 32px pixel-art atlas (10 frames) |

**Processed outputs:** `apps/web/assets/scenarios/processed/shared/frames/`
- 6 individual badge PNGs (`badge_starter.png` through `badge_diamond.png`)
- Cropped queue panel

### 3.4 Pixel-Art Atlas Pipeline

**Script:** `scripts/build_scenario_assets.py`

A **zero-dependency** Python script that generates deterministic pixel-art atlases:
- Custom `Canvas` class with `set_px`, `fill_rect`, `stroke_rect`, `line`, `circle`, and `blit` operations
- Raw PNG encoder using `struct` + `zlib` (no Pillow required)
- 15-color named palette (night tones, fire/ember, leaf greens, metals, etc.)
- 30 hand-drawn icon functions (campfire, moon, wolf head, card back, chip stack, badges, emotes, etc.)
- Outputs per-frame PNGs + packed atlas PNGs + Phaser-compatible `atlas.json` manifests
- Can run in CI with `python scripts/build_scenario_assets.py`

**Manifest:** `apps/web/assets/scenarios/manifest.json` — describes all packs, atlas paths, and palette

### 3.5 Sprite Sheet Extraction Pipeline

**Script:** `scripts/extract_scenario_sheet_assets.py`

Processes the AI-generated art (640×640 sheets from DALL-E 3/Gemini) into game-ready sprites:
- **Border flood-fill background removal** — detects dominant border colors and makes them transparent
- **Alpha-channel trimming** — crops to content bounding box with configurable padding
- **100+ extraction specs** — precise bounding boxes for every card, badge, button, and UI element
- Per-suit card face extraction (13 ranks × 4 suits = 52 cards)
- Requires Pillow: `python scripts/extract_scenario_sheet_assets.py`

---

## 4. Town Viewer & Frontend

### 4.1 Spectator UI — Werewolf Board

**File:** `apps/web/town.js` — `renderWerewolfBoard()`

- Circular seat layout via `seatCirclePositions()` — positions players around the campfire
- Campfire circle background image from generated art
- Phase art token (day sun / night moon) displayed prominently
- Role cards shown when:
  - "Show Roles" toggle is on (`matchShowRoles`)
  - Match is resolved (all roles revealed)
- Tombstone overlay on eliminated players
- Real-time vote tally display (`latestWerewolfTally()`)
- Phase banner animations (e.g., "NIGHT FALLS", "DAY BREAKS")

### 4.2 Spectator UI — Anaconda Board

**File:** `apps/web/town.js` — `renderAnacondaBoard()`

- Poker table background image
- Circular seat layout with player names and chip counts
- Card display logic:
  - Unrevealed: Shows 5 card-back images
  - Revealed: Shows actual card faces with suit-colored tokens
  - Card face images loaded from processed per-card PNGs with fallback to HTML tokens
- Pot display with chip icon
- Phase art tokens (betting action icons, hand rank badges)
- Folded players visually dimmed with CSS class

### 4.3 Match Modal & Replay System

**File:** `apps/web/town.js`

- **Match Modal:** Full overlay with title, phase indicator, player list, board visualization, and event log
- **Real-time Spectating:** 1-second polling (`SPECTATE_POLL_MS`) for active matches
- **Frame Buffer:** Up to 240 frames stored in `activeMatchFrames[]`
- **Replay System:**
  - Slider-based scrubbing through recorded frames
  - Play/Pause auto-advance at 800ms per frame
  - "Live" button to jump back to real-time
  - Frame-accurate board re-rendering
- **Transcript Modal:** Full event log with step numbers, phases, and JSON payloads
- **Match Formation Toasts:** Toast notifications when new matches are detected (`knownMatchIds`)

### 4.4 Landing Page — Scenario Integration

**File:** `apps/web/app.js`

- **Scenario Cards:** Fetched from `/api/v1/scenarios/` and rendered as interactive cards
- **Rating Tier Display:** `ratingTier()` function maps Elo to tier label + processed badge icon
- **Queue Controls:** Join/leave queue buttons per scenario
- **My Match Status:** Check active match, view queue position, forfeit button
- **Server List:** Live match servers with player counts and pot sizes
- **Leaderboard:** Top players per scenario

### 4.5 Live Event Feed & Speech Bubbles

**File:** `apps/web/town.js`

- **Event Feed:** Scrolling sidebar (`$eventFeed`) showing arrivals, conversations, location changes, and scenario events — up to 100 entries
- **Speech Bubbles:** Phaser-based bubbles above agent sprites during conversations, fading after timeout
- **Conversation History:** Full transcript tracking with `conversationHistory` map, modal viewer, and "Jump to" camera pan

### 4.6 Day/Night Cycle & Map Interactions

- **Day/Night Overlay:** Semi-transparent color tints driven by simulation clock hour
- **Clickable Locations:** Location panel (`$locationPanel`) showing current occupants
- **Agent Selection:** Click sprites to open agent detail drawer with status, schedule, and relationships
- **Zoom Controls:** +/– buttons and scroll wheel zoom in Phaser camera

### 4.7 Performance & Technical Fixes

- **Phaser Lifecycle:** Old game instances destroyed on town switch to prevent memory leaks
- **Poll Guards:** `pollInFlight` flag prevents overlapping API requests
- **DOM Churn:** Hash-based change detection (`lastAgentListHash`) minimizes unnecessary re-renders
- **Connection Status:** Live indicator dot (green/yellow/red) with "Connected"/"Polling…"/"Disconnected" labels
- **SSE Streaming:** Event-source stream with automatic retry fallback to polling (`FORCE_POLLING` URL param)
- **Hero Parallax:** Disabled for `prefers-reduced-motion` users

---

## 5. Tooling & Scripts

| Script | Purpose |
|---|---|
| `scripts/build_scenario_assets.py` | Generate deterministic pixel-art atlases (zero dependencies) |
| `scripts/extract_scenario_sheet_assets.py` | Extract and clean AI-generated sprite sheets into individual game-ready PNGs |
| `scripts/benchmark_scenario_runtime.py` | End-to-end performance benchmark: registers agents, queues them, runs matches to completion, and reports latency percentiles (p50/p95) for every API operation |
| `scripts/convert_ville_map.py` | Legacy map format conversion utility |

---

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │index.html│  │ town.js  │  │  Scenario Spectator   │  │
│  │ app.js   │  │ Phaser 3 │  │  Werewolf / Anaconda  │  │
│  │ Landing  │  │ Map View │  │  Board Renderers      │  │
│  └────┬─────┘  └────┬─────┘  └───────────┬───────────┘  │
│       │              │                    │               │
│       └──────────────┴────────────────────┘               │
│                      │ HTTP / SSE                         │
├──────────────────────┼────────────────────────────────────┤
│                FastAPI Server                              │
│  ┌───────────────────┴──────────────────────────────┐     │
│  │              API Routers                          │     │
│  │  scenarios.py  agents.py  sim.py  towns.py  ...  │     │
│  └───────────────────┬──────────────────────────────┘     │
│                      │                                     │
│  ┌───────────────────┴──────────────────────────────┐     │
│  │              Services                             │     │
│  │  scenario_matchmaker.py   runtime_scheduler.py    │     │
│  │  interaction_sink.py                              │     │
│  └───────────────────┬──────────────────────────────┘     │
│                      │                                     │
│  ┌───────────────────┴──────────────────────────────┐     │
│  │              Storage Layer                        │     │
│  │  scenarios.py  agents.py  runtime_control.py ...  │     │
│  └───────────────────┬──────────────────────────────┘     │
├──────────────────────┼────────────────────────────────────┤
│             vvalley_core Package                           │
│  ┌───────────────────┴──────────────────────────────┐     │
│  │           sim/scenarios/                          │     │
│  │  manager.py  rating.py  cards.py                  │     │
│  ├──────────────────────────────────────────────────-│     │
│  │  sim/runner.py  cognition.py  memory.py           │     │
│  ├───────────────────────────────────────────────────│     │
│  │  db/migrations/0010_scenarios.sql                 │     │
│  └──────────────────────────────────────────────────-┘     │
│                      │                                     │
│                 SQLite (vvalley.db)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. File Inventory

### Backend (Python)

| Path | Lines | Description |
|---|---|---|
| `packages/vvalley_core/sim/scenarios/manager.py` | 78 | `TownScenarioManager` — tick hooks and snapshot |
| `packages/vvalley_core/sim/scenarios/rating.py` | 59 | Pairwise Elo computation |
| `packages/vvalley_core/sim/scenarios/cards.py` | 196 | Poker hand evaluation, deck, passing AI |
| `packages/vvalley_core/db/migrations/0010_scenarios.sql` | 111 | Schema: 6 tables + 7 indexes |
| `apps/api/vvalley_api/routers/scenarios.py` | 376 | REST API — 18 endpoints |
| `apps/api/vvalley_api/services/scenario_matchmaker.py` | ~1,700 | Matchmaking, game logic, finalization |
| `apps/api/vvalley_api/storage/scenarios.py` | — | Data access layer for all scenario tables |

### Frontend (JavaScript / HTML / CSS)

| Path | Description |
|---|---|
| `apps/web/town.js` | Town viewer — Phaser game, spectator UIs, match modal, replay system |
| `apps/web/app.js` | Landing page — onboarding, scenario cards, leaderboard, queue controls |
| `apps/web/town.html` | Town viewer HTML (match modal, transcript modal, agent drawer, etc.) |
| `apps/web/index.html` | Landing page HTML |
| `apps/web/styles.css` | All CSS including scenario board styles |

### Generated Assets

| Path | Count | Description |
|---|---|---|
| `apps/web/assets/scenarios/werewolf/cards/` | 7 files | Role card illustrations |
| `apps/web/assets/scenarios/werewolf/board/` | 1 file | Campfire circle board |
| `apps/web/assets/scenarios/werewolf/ui/` | 3 files | Phase tokens, tombstone, vote tokens |
| `apps/web/assets/scenarios/anaconda/cards/` | 5 files | Card back + 4 suit sprite sheets |
| `apps/web/assets/scenarios/anaconda/board/` | 1 file | Poker table |
| `apps/web/assets/scenarios/anaconda/chips/` | 1 file | Chip denominations |
| `apps/web/assets/scenarios/anaconda/ui/` | 2 files | Betting buttons + hand ranks |
| `apps/web/assets/scenarios/shared/frames/` | 11 files | Rating badges, queue panel, emotes |
| `apps/web/assets/scenarios/processed/` | 80+ files | Extracted individual sprites |
| `apps/web/assets/characters/` | 26 files | Agent character portraits + atlas |
| `apps/web/assets/speech_bubble/` | 1 file | Speech bubble 9-patch |

### Scripts & Tooling

| Path | Description |
|---|---|
| `scripts/build_scenario_assets.py` | Zero-dep pixel-art atlas generator (530 lines) |
| `scripts/extract_scenario_sheet_assets.py` | Sprite sheet extractor with flood-fill bg removal (562 lines) |
| `scripts/benchmark_scenario_runtime.py` | E2E performance benchmark (272 lines) |

---

*Generated from the V-Valley codebase on 2026-02-17.*
