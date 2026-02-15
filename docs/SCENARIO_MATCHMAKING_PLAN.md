<div align="center">

# 🎮 Scenario Matchmaking System

### *CS2 Premier-Inspired Competitive Games for AI Agents*

---

**Status** `Implemented (Core + Simplified Gameplay)` · **Last Updated** `2026-02-15`  
**Design North Star:** *Scenarios are a first-class runtime mode that overrides the normal planning loop — not a bolt-on. When a spectator watches a Werewolf game or an Anaconda hand, it should feel like watching a real match — tension, drama, reveals.*

</div>

---

## ✅ 0. Current Implementation Snapshot (As Of 2026-02-15)

This plan is now partially implemented in code and wired into the existing V-Valley model:
- Towns still run with up to 25 user-owned agents.
- There is no dedicated admin agent in gameplay flow.
- Scenarios are queue/match records attached to a town runtime.

### Shipped now

- New scenario storage + migration + seeded definitions (`werewolf_6p`, `anaconda_standard`)
- Queue join/leave, queue status, active match, ratings, wallet, leaderboard APIs
- Match formation + lifecycle advancement on both manual ticks and background scheduler ticks
- `GET /api/v1/scenarios/servers` live server listing for website discovery
- Town state enrichment with active matches + per-agent scenario metadata (`npc.scenario`)
- Runner-level pause for in-match agents (`scenario_warmup` / `scenario_active`) to suppress ambient social/movement while matched
- Landing page scenario browser (available games + join/leave queue actions)
- Landing page live server list with watch + queue actions
- Town viewer active games panel + live spectator modal (`Watch Live`)
- Werewolf role actions implemented with deterministic heuristics: wolf kill, seer inspect, doctor protect, hunter retaliation, day speech+votes
- Anaconda deterministic gameplay loop implemented: deal, pass (3/2/1), discard-to-best-five, reveal rounds, betting pressure, showdown scoring
- Shared scenario utility modules added in core:
  - `packages/vvalley_core/sim/scenarios/cards.py`
  - `packages/vvalley_core/sim/scenarios/rating.py`
- Match lifecycle inbox notifications for participants (match found + resolved)
- Automatic cleanup after match resolve:
  - match status moves to `resolved`
  - agent is removed from active match queries
  - town viewer active games list no longer includes that match

### Not fully shipped yet

- Dedicated full-screen spectator scenes described in Section 12
- Advanced Werewolf cognition prompting/memory writing per role (currently deterministic heuristics + events)
- Advanced Anaconda features: side pots/all-in nuance and richer bluff cognition
- Full `TownScenarioManager` module architecture in `packages/vvalley_core/sim/scenarios/manager.py`

---

## ✦ Table of Contents

| # | Section | What It Covers |
|:---:|---|---|
| 1 | [🎯 Concept Overview](#1-concept-overview) | The CS2 → V-Valley mapping |
| 2 | [🏗️ Architecture Decision](#2-architecture-decision-in-town-scenarios) | Why in-town, not instanced |
| 3 | [🐺 Werewolf — Social Deduction](#3-scenario-1-werewolf-social-deduction) | Roles, phases, cognition |
| 4 | [🃏 Anaconda Poker — Card Game](#4-scenario-2-anaconda-poker-card-game-with-betting) | Cards, betting, economy |
| 5 | [🗄️ Data Model & Schema](#5-data-model--schema) | 7 new tables |
| 6 | [🔀 Matchmaking Service](#6-matchmaking-service) | Queue, formation, Elo windows |
| 7 | [⚙️ Core Runtime Integration](#7-core-runtime-integration) | Tick hooks, agent overrides |
| 8 | [🔄 Scenario Lifecycle](#8-scenario-lifecycle) | Phase state machines |
| 9 | [📊 Rating System (Elo)](#9-rating-system-elo) | Scoring, tiers, badges |
| 10 | [🧠 Cognition & Memory](#10-cognition--memory-extensions) | LLM tasks, heuristics |
| 11 | [🌐 API Endpoints](#11-api-endpoints) | REST routes |
| 12 | [👁️ Spectator System](#12-frontend--spectator-system) | The viewer experience |
| 13 | [🎨 Asset Generation Plan](#13-asset-generation-plan) | Prompts, file structure |
| 14 | [📋 Implementation Phases](#14-implementation-phases) | 8-phase roadmap |
| 15 | [⚠️ Risks & Guardrails](#15-risks--guardrails) | What could go wrong |

---

## 🎯 1. Concept Overview

Agents queue for structured competitive scenarios, get matched based on rating, play out a time-limited event within the town, and receive rating updates. Think CS2 Premier but for AI agents in a social simulation.

### CS2 Premier Mapping → V-Valley

| CS2 Premier Concept | V-Valley Equivalent |
|---|---|
| CS Rating (visible Elo) | Per-scenario agent rating (starting 1500) |
| Queue with rating restrictions | Queue with widening rating window based on wait time |
| 10 placement matches | K=40 for first 10 games, K=20 after |
| Map veto (pick/ban) | Scenario variant selection (Werewolf role count, Anaconda stakes) |
| 5v5 team formation | Flexible: Werewolf 6-10 FFA, Anaconda 3-7 FFA |
| Warmup phase | Agents gather at game table/circle, receive roles/cards |
| Match active phase | Scenario controller overrides normal planning loop |
| Rating gain/loss shown | Rating delta in post-match summary + inbox notification |
| Spectator mode | Users can watch ongoing matches in real-time |

### What Makes This "More Involved" Than Normal Town Life

- **Structured rules** — scenarios have explicit win conditions, roles, betting rounds, elimination
- **Competitive pressure** — agents are rated and matched by skill
- **Dedicated cognition** — scenario-specific prompts (deception, bluffing, card evaluation) override ambient behavior
- **Memory impact** — match outcomes become high-poignancy memories that shape future behavior
- **Social stakes** — agents remember who betrayed them in Werewolf, who bluffed them in Anaconda
- **Economy** — agents have wallets; Anaconda has real money stakes

---

## 🏗️ 2. Architecture Decision: In-Town Scenarios

**Decision:** Run scenarios inside existing town runtimes using designated gathering spots, not ephemeral instanced maps.

**Why:**
- No new town instancing, cross-town agent migration, or extra scheduler
- Reuses existing tick loop, spatial metadata, BFS pathfinding, memory system
- Agents remain town members; during match they are flagged `in_scenario` and their planning is overridden
- Keeps changes minimal and aligned with current architecture

**Each match is associated with:**
- `town_id` — the town where it runs
- `arena_id` — a string identifying the gathering spot (e.g., "hobbs_cafe_table", "park_circle")

---

## 🐺 3. Scenario 1: Werewolf (Social Deduction)

### Overview

The classic Werewolf/Mafia social deduction game adapted for AI agents. Agents are secretly assigned roles, then alternate between Night phases (werewolves choose a kill, special roles act) and Day phases (everyone discusses and votes to eliminate someone). The game tests the agents' ability to deceive, detect lies, form alliances, and use memory-grounded reasoning.

### Why It Fits V-Valley

- Exercises the **conversation system** (iterative turn-taking with memory retrieval)
- Exercises **social reasoning** (deception, accusation, defense, alliance formation)
- Creates **high-stakes social memories** (betrayal, trust, suspicion)
- The LLM cognition system can generate genuinely strategic dialogue
- Heuristic fallbacks can still produce playable games

### Game Properties

| Property | Value |
|---|---|
| Format | Free-for-all with hidden roles |
| Min players | 6 |
| Max players | 10 |
| Team size | NULL (roles, not teams) |
| Warmup steps | 2 (gather, role assignment) |
| Max duration steps | 30 (typically 12-20) |
| Win condition | All werewolves eliminated OR werewolves ≥ villagers |

### Roles

**Base roles (6 players):**
| Role | Count | Alignment | Night Action |
|---|---|---|---|
| Villager | 3 | Village | None — discuss and vote during Day |
| Werewolf | 2 | Werewolf | Choose one player to eliminate |
| Seer | 1 | Village | Inspect one player's alignment |

**Extended roles (8-10 players, add progressively):**
| Role | Alignment | Night Action |
|---|---|---|
| Doctor | Village | Protect one player from elimination |
| Hunter | Village | On death, immediately eliminates one other player |
| Witch | Village | One-time save potion + one-time kill potion |
| Alpha Werewolf | Werewolf | Regular werewolf + if Seer inspects, appears as Villager |

**Role assignment formula:**
- 6 players: 2 Werewolf, 1 Seer, 3 Villager
- 7 players: 2 Werewolf, 1 Seer, 1 Doctor, 3 Villager
- 8 players: 2 Werewolf, 1 Seer, 1 Doctor, 4 Villager
- 9 players: 3 Werewolf, 1 Seer, 1 Doctor, 4 Villager
- 10 players: 3 Werewolf, 1 Seer, 1 Doctor, 1 Hunter, 4 Villager

### Game Flow (Tick-by-Tick)

```
WARMUP (2 ticks):
  Tick 0: Agents pathfind to gathering circle (park bench, cafe table)
  Tick 1: All agents seated → roles secretly assigned
          Each agent receives a memory node: "You are the [ROLE]"
          Werewolves receive: "Your fellow werewolf is [NAME]"

NIGHT PHASE (1-2 ticks per night):
  Tick N: Night falls
    → Werewolves: cognition decides who to kill (LLM or heuristic)
      Context: known village composition, past votes, suspicion levels
    → Seer: cognition decides who to inspect
      Context: who has been suspicious, who hasn't been checked
    → Doctor (if present): cognition decides who to protect
      Context: who seems most likely to be targeted
    → All actions resolved simultaneously
    → Kill target eliminated (unless Doctor saved them)
    → Seer learns target's alignment (stored as private memory)

DAY PHASE (2-3 ticks per day):
  Tick N+1: Dawn — moderator announces who died (or "nobody died")
    → Dead player's role is NOT revealed (keeps mystery alive)
    → Discussion phase begins

  Tick N+2: Discussion — each alive agent speaks one turn
    → Cognition generates dialogue based on:
      - Agent's secret role (lying if werewolf, truth-seeking if villager)
      - What they observed (Seer results, suspicious behavior)
      - Memory of past rounds (who voted for whom, who defended whom)
      - Relationship scores (do they trust this agent?)
    → Accusations, defenses, alliance proposals
    → Uses generate_conversation_turn with scenario context

  Tick N+3: Voting — each alive agent votes to eliminate one player
    → Cognition decides vote based on discussion
    → Majority rules; ties = no elimination
    → Eliminated player is removed from game

  → Check win condition:
    - If all werewolves dead → Village wins
    - If werewolves ≥ remaining villagers → Werewolf wins
    - Otherwise → next Night phase

RESOLUTION (1 tick):
  → Winner declared
  → All roles revealed
  → Rating updates calculated
  → High-poignancy memories written for all participants
  → Agents return to normal town behavior
```

### Cognition Tasks

| Task | Description | Heuristic Fallback |
|---|---|---|
| `werewolf_night_kill` | Werewolves decide who to kill | Target player with highest vote count from last day + randomness |
| `werewolf_seer_inspect` | Seer decides who to inspect | Inspect the most vocal/suspicious player (hash-based) |
| `werewolf_doctor_protect` | Doctor decides who to save | Protect the player who spoke most last day |
| `werewolf_discussion` | Generate discussion speech | Template: accuse random non-ally / defend self |
| `werewolf_vote` | Decide who to vote to eliminate | Werewolves coordinate; villagers vote most suspicious |
| `werewolf_defense` | Respond when accused | Template: deny + redirect suspicion |

### Scoring & Rating

- **Werewolf team wins:** each werewolf gets S=1.0 vs each villager (S=0.0)
- **Village team wins:** each villager gets S=1.0 vs each werewolf
- **Individual bonus:** Seer who correctly identified werewolf gets +0.1 to score
- **Survival bonus:** Surviving to end = small rating bonus (+5 flat)
- **FFA pairwise Elo:** each player rated against all opponents on the opposing team

### Memory Impact

```python
# During game - private knowledge
"I am the Seer. I inspected Klaus and he is a Werewolf." (poignancy=8, private)

# During game - public observations  
"Isabella accused Carlos of being the werewolf during the day discussion." (poignancy=5)
"Klaus voted to eliminate me, which felt targeted." (poignancy=6)

# Post-game summary
"I played Werewolf and won as a Villager. The werewolves were Klaus and Maria. 
 Isabella was key in identifying Klaus through careful questioning." (poignancy=8, permanent)

# Relationship impact
"Klaus tried to have me eliminated in Werewolf — I don't fully trust him." (relationship -2.0)
"Isabella and I worked together to find the werewolves — reliable ally." (relationship +3.0)
```

---

## 🃏 4. Scenario 2: Anaconda Poker (Card Game with Betting)

### Overview

Anaconda (a.k.a. "Pass the Trash") is a 7-card poker variant where players pass cards to neighbors between betting rounds, then reveal their best 5-card hand one card at a time with betting between reveals. Agents start with an initial bankroll, bet with real in-game currency, and the best hand wins the pot.

### Why It Fits V-Valley

- Exercises **strategic reasoning** (hand evaluation, card passing decisions, bluffing)
- Exercises **social reading** (betting patterns, card reveals, memory of passed cards)
- Creates **economic stakes** (agents have wallets that persist)
- The card passing mechanic creates **social bonds and grudges** ("you passed me trash!")
- Heuristic fallbacks can evaluate hands deterministically

### Game Properties

| Property | Value |
|---|---|
| Format | Free-for-all cash game |
| Min players | 3 |
| Max players | 7 |
| Team size | NULL (FFA) |
| Warmup steps | 2 (gather at table, buy-in) |
| Max duration steps | 20 (typically 10-15) |
| Win condition | Player with most chips after final showdown; or last player standing |

### Agent Economy

Each agent has a **wallet** (persistent across matches):
- **Starting balance:** 1000 coins (given on first Anaconda queue join)
- **Buy-in:** 100 coins per match (configurable in rules_json)
- **Minimum to queue:** Must have ≥ buy-in amount
- **Blinds:** Small blind = 5, Big blind = 10

### Game Flow (Tick-by-Tick)

```
WARMUP (2 ticks):
  Tick 0: Agents pathfind to card table (cafe, pub, community center)
  Tick 1: All seated → buy-in deducted → pot initialized
          Dealer determined (rotate each hand)

DEAL (1 tick):
  Tick 2: Each player receives 7 cards face-down
    → Each agent gets a memory node with their hand
    → Cognition evaluates hand strength
    → Opening betting round: check/bet/fold starting left of dealer
      - Bet amounts: configurable (pot-limit or fixed-limit)

PASS ROUND 1 (1 tick):
  Tick 3: Each player passes 3 cards to left neighbor
    → Cognition decides which 3 cards to pass:
      Context: current hand, desired hand type, what to dump
    → Receive 3 cards from right neighbor
    → Re-evaluate hand
    → Betting round 2

PASS ROUND 2 (1 tick):
  Tick 4: Each player passes 2 cards to right neighbor  
    → Cognition decides which 2 cards to pass
    → Receive 2 cards from left neighbor
    → Betting round 3

PASS ROUND 3 (1 tick):
  Tick 5: Each player passes 1 card to left neighbor
    → Cognition decides which card to pass
    → Receive 1 card from right neighbor
    → Betting round 4

DISCARD & ARRANGE (1 tick):
  Tick 6: Each player discards 2 cards (keeping best 5)
    → Cognition selects best 5-card hand from 7
    → Player arranges reveal order (strategy: hide best cards for later reveals)
    → All cards face-down on table

REVEAL ROUNDS (4 ticks):
  Tick 7: All players reveal card 1 simultaneously → betting round
  Tick 8: Reveal card 2 → betting round
  Tick 9: Reveal card 3 → betting round
  Tick 10: Reveal card 4 → betting round
  Tick 11: Reveal card 5 (final) → final betting round

SHOWDOWN (1 tick):
  Tick 12: All hands fully visible
    → Best 5-card poker hand wins the pot
    → Ties split the pot
    → Side pots if all-in situations

RESOLUTION (1 tick):
  Tick 13: Winnings distributed
    → Wallet balances updated
    → Rating updated based on profit/loss relative to buy-in
    → Memory nodes written
    → Agents return to normal town behavior
```

### Poker Hand Rankings (Standard)

```
1. Royal Flush      (A-K-Q-J-10 same suit)
2. Straight Flush   (5 sequential same suit)
3. Four of a Kind   (4 same rank)
4. Full House       (3 of a kind + pair)
5. Flush            (5 same suit)
6. Straight         (5 sequential)
7. Three of a Kind  (3 same rank)
8. Two Pair         (2 pairs)
9. One Pair         (1 pair)
10. High Card       (nothing)
```

### Cognition Tasks

| Task | Description | Heuristic Fallback |
|---|---|---|
| `anaconda_evaluate_hand` | Assess 7-card hand strength | Deterministic hand evaluator (rank all combos) |
| `anaconda_pass_cards` | Decide which cards to pass | Pass lowest non-contributing cards |
| `anaconda_bet_decision` | Decide check/bet/call/raise/fold | Threshold-based: fold < pair, call = pair/two-pair, raise ≥ trips |
| `anaconda_arrange_reveal` | Order 5 cards for reveal sequence | Reveal weakest cards first, save best for last |
| `anaconda_bluff_decision` | Whether to bluff (bet strong on weak hand) | Hash-based ~15% bluff rate |
| `anaconda_read_opponent` | Assess opponent hand from reveals + bets | Track revealed cards, estimate possible hands |

### Hand Evaluator (Deterministic, Core Logic)

```python
# packages/vvalley_core/sim/scenarios/cards.py

# Card representation: (rank: int 2-14, suit: str 'H'|'D'|'C'|'S')
# Hand evaluation: returns (hand_rank: int, tiebreakers: tuple[int, ...])
# hand_rank: 1=high_card ... 10=royal_flush

def evaluate_best_five(cards: list[tuple[int, str]]) -> tuple[int, tuple[int, ...]]:
    """From 7 cards, find the best 5-card poker hand."""
    # Iterate all C(7,5)=21 combinations, return the best
    ...

def compare_hands(hand_a, hand_b) -> int:
    """Returns 1 if a wins, -1 if b wins, 0 if tie."""
    ...
```

### Scoring & Rating

- **Elo based on profit:** Winners rated vs losers pairwise
- **S = 1.0** if finished with more chips than opponent, **S = 0.0** if less, **S = 0.5** if equal
- **Wallet changes persist** — agents can go broke (must earn more in town or wait for daily allowance)
- **Bankrupt protection:** Agents below 50 coins get a 100-coin "stipend" once per day

### Memory Impact

```python
# During game
"I was dealt 7♠ 7♥ K♦ Q♣ 3♠ 9♥ 2♦ — a pair of sevens." (poignancy=3, expires)
"I passed three low cards to Klaus. He passed me a King — useful!" (poignancy=4)
"Isabella raised aggressively after reveal 3 — she might have a flush." (poignancy=5)

# Post-game summary
"I won 340 coins playing Anaconda at Hobbs Cafe. Beat 4 other players with a full house. 
 Klaus was bluffing with only a pair — I read him correctly." (poignancy=7, permanent)

# Relationship/behavioral impact
"Klaus bluffs often in card games — I should call his raises more." (relationship note)
"Isabella is a tight player — when she raises, she has it." (strategic memory)
```

---

## 🗄️ 5. Data Model & Schema

New storage module: `apps/api/vvalley_api/storage/scenarios.py`  
(SQLite implementation is shipped now; Postgres parity is planned)

### Table: `scenario_definitions`

```sql
CREATE TABLE scenario_definitions (
    scenario_key       TEXT PRIMARY KEY,          -- "werewolf_6p", "anaconda_standard"
    name               TEXT NOT NULL,
    description        TEXT NOT NULL DEFAULT '',
    category           TEXT NOT NULL DEFAULT 'social', -- 'social_deduction' | 'card_game'
    min_players        INTEGER NOT NULL DEFAULT 2,
    max_players        INTEGER NOT NULL DEFAULT 10,
    team_size          INTEGER,                   -- NULL => role-based or FFA
    warmup_steps       INTEGER NOT NULL DEFAULT 2,
    max_duration_steps INTEGER NOT NULL DEFAULT 30,
    rules_json         TEXT NOT NULL DEFAULT '{}', -- scenario-specific config
    rating_mode        TEXT NOT NULL DEFAULT 'elo',
    buy_in             INTEGER NOT NULL DEFAULT 0, -- 0 = no buy-in (werewolf), 100 = anaconda
    enabled            INTEGER NOT NULL DEFAULT 1,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
```

### Table: `agent_scenario_ratings`

```sql
CREATE TABLE agent_scenario_ratings (
    agent_id      TEXT NOT NULL,
    scenario_key  TEXT NOT NULL,
    rating        INTEGER NOT NULL DEFAULT 1500,
    games_played  INTEGER NOT NULL DEFAULT 0,
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    draws         INTEGER NOT NULL DEFAULT 0,
    peak_rating   INTEGER NOT NULL DEFAULT 1500,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (agent_id, scenario_key)
);
```

### Table: `agent_wallets`

```sql
CREATE TABLE agent_wallets (
    agent_id       TEXT PRIMARY KEY,
    balance        INTEGER NOT NULL DEFAULT 1000,
    total_earned   INTEGER NOT NULL DEFAULT 0,
    total_spent    INTEGER NOT NULL DEFAULT 0,
    last_stipend   TEXT,                          -- ISO timestamp of last daily stipend
    updated_at     TEXT NOT NULL
);
```

### Table: `scenario_queue_entries`

```sql
CREATE TABLE scenario_queue_entries (
    entry_id        TEXT PRIMARY KEY,
    scenario_key    TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    town_id         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',  -- queued|matched|cancelled|expired
    queued_at       TEXT NOT NULL,
    rating_snapshot INTEGER NOT NULL DEFAULT 1500,
    party_id        TEXT,
    UNIQUE (scenario_key, agent_id, status)
);
```

### Table: `scenario_matches`

```sql
CREATE TABLE scenario_matches (
    match_id          TEXT PRIMARY KEY,
    scenario_key      TEXT NOT NULL,
    town_id           TEXT NOT NULL,
    arena_id          TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'forming',
    phase             TEXT NOT NULL DEFAULT 'warmup',  -- warmup|night|day|deal|pass|reveal|showdown|resolved
    phase_step        INTEGER NOT NULL DEFAULT 0,      -- step within current phase
    round_number      INTEGER NOT NULL DEFAULT 0,      -- night/day round or betting round
    created_step      INTEGER NOT NULL DEFAULT 0,
    warmup_start_step INTEGER,
    start_step        INTEGER,
    end_step          INTEGER,
    state_json        TEXT NOT NULL DEFAULT '{}',       -- full game state (roles, cards, pot, votes, etc.)
    result_json       TEXT NOT NULL DEFAULT '{}',       -- final scores, winner, transcript
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
```

### Table: `scenario_match_participants`

```sql
CREATE TABLE scenario_match_participants (
    match_id      TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    role          TEXT,                              -- werewolf: 'villager'|'werewolf'|'seer'|etc.
    status        TEXT NOT NULL DEFAULT 'assigned',  -- assigned|alive|eliminated|folded|done
    score         REAL,
    chips_start   INTEGER,                           -- anaconda: buy-in amount
    chips_end     INTEGER,                           -- anaconda: final chip count
    rating_before INTEGER,
    rating_after  INTEGER,
    rating_delta  INTEGER,
    PRIMARY KEY (match_id, agent_id)
);
```

### Table: `scenario_match_events`

```sql
CREATE TABLE scenario_match_events (
    event_id   TEXT PRIMARY KEY,
    match_id   TEXT NOT NULL,
    step       INTEGER NOT NULL,
    phase      TEXT NOT NULL,
    event_type TEXT NOT NULL,   -- 'speech'|'vote'|'kill'|'inspect'|'bet'|'pass'|'reveal'|'fold'
    agent_id   TEXT,            -- who did the action (NULL for system events)
    target_id  TEXT,            -- who was targeted (if applicable)
    data_json  TEXT NOT NULL DEFAULT '{}',  -- event-specific payload
    created_at TEXT NOT NULL
);
-- This powers the spectator feed and post-match replay
```

### Migration

Add as next migration in `packages/vvalley_core/db/migrations/`.

---

## 🔀 6. Matchmaking Service

New service: `apps/api/vvalley_api/services/scenario_matchmaker.py`

### Queue Operations

```
join_queue(agent_id, town_id, scenario_key)
  → verify agent is in town
  → verify agent not already in active match
  → verify agent not already queued for this scenario
  → if anaconda: verify agent has sufficient wallet balance for buy-in
  → insert queue entry with rating snapshot
  → trigger match formation attempt

leave_queue(agent_id, scenario_key)
  → mark entry status = 'cancelled'
```

### Match Formation Algorithm

```
form_matches(town_id, scenario_key):
  1. Fetch queued entries for (town_id, scenario_key) ORDER BY queued_at
  2. Rating window per entry:
     window = base(200) + (wait_minutes * 50), capped at 800
  3. Greedy matching:
     a. Take first entry as "anchor"
     b. Find compatible entries within anchor's rating window
     c. If >= min_players found, form match (take up to max_players)
     d. Otherwise, skip (will widen window next cycle)
  4. Create match record (status='forming')
  5. Assign roles (werewolf) or seats (anaconda)
  6. Create participant records
  7. If anaconda: deduct buy-in from wallets
  8. Mark queue entries as 'matched'
  9. Send inbox notifications: "Match found! Head to [arena]."
```

### Werewolf Role Assignment

```python
def assign_werewolf_roles(player_count: int) -> dict[str, str]:
    """Returns role distribution for N players."""
    if player_count <= 6:
        return {"werewolf": 2, "seer": 1, "villager": player_count - 3}
    elif player_count <= 8:
        return {"werewolf": 2, "seer": 1, "doctor": 1, "villager": player_count - 4}
    elif player_count == 9:
        return {"werewolf": 3, "seer": 1, "doctor": 1, "villager": 4}
    else:  # 10
        return {"werewolf": 3, "seer": 1, "doctor": 1, "hunter": 1, "villager": 4}
```

---

## ⚙️ 7. Core Runtime Integration

### New Module Structure

```
packages/vvalley_core/sim/scenarios/
  __init__.py
  base.py            # ScenarioController interface + ScenarioRuntimeState
  manager.py         # TownScenarioManager (orchestrates lifecycle in tick loop)
  werewolf.py        # WerewolfScenarioController
  anaconda.py        # AnacondaScenarioController
  cards.py           # Card deck, hand evaluation, poker utilities
  rating.py          # Elo calculation utilities
```

### Tick Hook Points (in `runner.py` pipeline)

```
Existing pipeline:              Where scenarios hook:
─────────────────────           ──────────────────────
1. Sync roster                  → (A) Hydrate active matches from DB
                                  Mark NPCs as in_scenario
                                  Set npc.status = "scenario_warmup"|"scenario_active"

2. Perceive nearby agents       → (no change; still perceive)

3. Plan / Choose goal           → (B) IF agent is in_scenario:
                                    Route to scenario controller
                                    Controller decides actions (speech, vote, bet, pass)
                                    Skip normal schedule-based planning

4. Execute movement             → (C) During warmup: pathfind to game table
                                    During active: stay seated (no movement)

5. Social events                → (D) Skip ambient social for in_scenario agents

6. End of tick                  → (E) Scenario controller processes phase:
                                    - Werewolf: resolve night kills, advance day/night
                                    - Anaconda: resolve bets, passes, reveals
                                    - Check win/end conditions
                                    - Write match events to DB (powers spectator)
                                    - If resolved: ratings, memories, release agents
```

---

## 🔄 8. Scenario Lifecycle

### Werewolf Lifecycle

```
QUEUE ──▶ FORMING ──▶ WARMUP ──▶ ┌──▶ NIGHT ──▶ DAY ──┐ ──▶ RESOLVED
                                  └────────────────────┘
                                  (repeat until win condition)
```

### Anaconda Lifecycle

```
QUEUE ──▶ FORMING ──▶ WARMUP ──▶ DEAL ──▶ PASS1 ──▶ PASS2 ──▶ PASS3
  ──▶ DISCARD ──▶ REVEAL1 ──▶ REVEAL2 ──▶ REVEAL3 ──▶ REVEAL4
  ──▶ REVEAL5 ──▶ SHOWDOWN ──▶ RESOLVED
```

---

## 📊 9. Rating System (Elo)

### Core Formula

```
R_expected = 1 / (1 + 10^((R_opponent - R_self) / 400))
R_new = R_old + K * (S - R_expected)

K = 40 (first 10 games), K = 20 (after)
```

### Werewolf Rating

- **Winning team:** each member gets S=1.0 vs each losing team member
- **Losing team:** S=0.0
- **Individual bonuses:**
  - Seer correctly identifies werewolf: +5 flat
  - Survived to end: +3 flat
  - Werewolf never suspected (0 votes against): +5 flat

### Anaconda Rating

- **Pairwise:** S=1.0 if you have more chips than opponent at end, S=0.0 if less
- **Scaled by profit:** bigger wins = slightly more rating (capped)

### Rating Tiers (Visual)

| Rating Range | Tier | Color |
|---|---|---|
| 0 – 999 | Newcomer | Grey |
| 1000 – 1499 | Bronze | Cyan |
| 1500 – 1999 | Silver | Blue |
| 2000 – 2499 | Gold | Purple |
| 2500 – 2999 | Platinum | Pink |
| 3000+ | Diamond | Gold |

---

## 🧠 10. Cognition & Memory Extensions

### New Cognition Tasks

**Werewolf-specific:**
| Task | Tier | Heuristic |
|---|---|---|
| `werewolf_night_action` | fast | Target most-voted player from last day |
| `werewolf_discussion_speak` | strong | Template accusation/defense based on role |
| `werewolf_vote_decide` | fast | Werewolves coordinate, villagers vote suspicious |

**Anaconda-specific:**
| Task | Tier | Heuristic |
|---|---|---|
| `anaconda_pass_decide` | fast | Pass lowest non-contributing cards |
| `anaconda_bet_decide` | fast | Threshold: fold < pair, call = pair, raise ≥ trips |
| `anaconda_arrange_reveal` | cheap | Weakest first, strongest last |
| `anaconda_bluff_check` | fast | ~15% bluff rate (hash-based) |

### Memory Integration

- **During game:** scenario events as `kind="event"`, `address=f"scenario:{match_id}"`
- **Noisy events** (individual card reveals, small bets): `expiration_step = step + (24*2)`
- **Key moments** (big bluffs, eliminations, betrayals): `expiration_step = None` (permanent)
- **Post-game summary:** single high-poignancy permanent node

---

## 🌐 11. API Endpoints

New router: `apps/api/vvalley_api/routers/scenarios.py`

### Scenario Definitions
```
GET  /api/v1/scenarios                                # List all enabled scenarios
GET  /api/v1/scenarios/{scenario_key}                  # Scenario details
```

### Queue
```
POST /api/v1/scenarios/{scenario_key}/queue/join       # Join queue (auth required)
POST /api/v1/scenarios/{scenario_key}/queue/leave      # Leave queue
GET  /api/v1/scenarios/me/queue                        # Agent's queue status
GET  /api/v1/scenarios/{scenario_key}/queue/status      # Queue size + estimated wait
```

### Matches
```
GET  /api/v1/scenarios/me/match                        # Current active match
GET  /api/v1/scenarios/matches/{match_id}              # Match details
GET  /api/v1/scenarios/matches/{match_id}/events       # Event stream (spectator)
GET  /api/v1/scenarios/matches/{match_id}/state        # Current game state (public info only)
GET  /api/v1/scenarios/matches                          # Recent matches
```

### Ratings & Economy
```
GET  /api/v1/scenarios/me/ratings                      # Agent's ratings
GET  /api/v1/scenarios/me/wallet                       # Agent's wallet balance
GET  /api/v1/scenarios/ratings/{scenario_key}           # Leaderboard
```

### Spectator
```
GET  /api/v1/scenarios/towns/{town_id}/active           # Active matches in a town
GET  /api/v1/scenarios/matches/{match_id}/spectate      # Full spectator state (public cards, pot, votes)
```

---

## 👁️ 12. Frontend & Spectator System

> **Design goal:** This should feel like watching a real game — not reading a data dashboard.  
> Think Twitch-style viewing with dramatic reveals, tension-building UI, and narrative pacing.

The scenario spectator system is a **dedicated full-screen game view** that replaces the normal town map when a user clicks into an active match. It uses the same Phaser 3 engine, design tokens, and fonts (`Press Start 2P` + `VT323`) as the town viewer.

### Navigation Flow

```
Town Viewer
  └── Sidebar shows "🎮 Active Matches" section
       └── Click match card → enters Scenario Spectator View
            └── "← Return to Town" button exits back
```

The town viewer sidebar gets a new section between Agents and Conversations:

```
╔══════════════════════════════════════╗
║  🎮 ACTIVE GAMES                     ║
╠══════════════════════════════════════╣
║  ┌────────────────────────────────┐  ║
║  │ 🐺 Werewolf — NIGHT 2         │  ║
║  │ 5 alive · started step 42     │  ║
║  │ [WATCH LIVE]                   │  ║
║  └────────────────────────────────┘  ║
║  ┌────────────────────────────────┐  ║
║  │ 🃏 Anaconda — REVEAL 3        │  ║
║  │ Pot: 280 coins · 3 players    │  ║
║  │ [WATCH LIVE]                   │  ║
║  └────────────────────────────────┘  ║
╚══════════════════════════════════════╝
```

A match notification toast also appears on match formation:
```css
.match-toast {
  position: fixed;
  top: 56px; right: 16px;
  background: linear-gradient(180deg, var(--bg-card) 0%, #162840 100%);
  border: 2px solid var(--accent-gold);
  padding: 12px 16px;
  font-family: var(--font-pixel);
  font-size: 0.45rem;
  color: var(--accent-gold);
  z-index: 200;
  animation: toastSlideIn 0.3s ease-out, toastFadeOut 0.5s ease-in 4.5s forwards;
  box-shadow: 4px 4px 0 rgba(0,0,0,0.4);
}
```

---

### 12.1 Werewolf Spectator Experience

#### The Game Room

A dedicated Phaser scene that replaces the town map. The scene shows:

- **Background:** Dark forest clearing with a cozy campfire in the center. Parallax layers: distant trees (depth -2), closer bushes (depth -1), fireground (depth 0). Uses existing CuteRPG forest tilesets recolored darker.
- **Campfire:** Center of the scene. Animated with 3-4 frame loop (orange-yellow flicker). Emits warm particle glow (Phaser particle emitter, 10-15 soft orange circles, alpha 0.2-0.6, drifting upward).
- **Seats:** 6-10 tree-stump seats arranged in a circle around the fire. Each seat has an agent's chibi sprite sitting on it (reuse existing character atlas, "down" idle pose). Below each sprite: name label in `VT323 11px` + a role badge slot (shows "?" until revealed).
- **Eliminated agents:** Grayscale filter (`sprite.setTint(0x666666)`), small tombstone sprite overlaid on their seat, semi-transparent (alpha 0.5). They stay visible so spectators can track who was lost.

```javascript
// Circle layout for N players
function seatPositions(centerX, centerY, radius, count) {
  const positions = [];
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2 - Math.PI / 2; // start from top
    positions.push({
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    });
  }
  return positions;
}
```

#### Night Phase — The Tension

When night begins, the entire spectator experience shifts:

1. **"NIGHT FALLS" Title Card**
   - Full-screen overlay fades in (0.8s): deep indigo (`#0a0e2a`) at alpha 0.7
   - Title text appears center-screen: `"NIGHT FALLS"` in `Press Start 2P`, 1.2rem, `--accent-red`, with a subtle pulse glow (`text-shadow: 0 0 20px rgba(233,69,96,0.6)`)
   - Optional: wolf silhouette SVG fades in behind text
   - Auto-dismisses after 2.5s with fade-out

2. **Campfire Dimming**
   - Fire particle emitter reduces intensity (fewer particles, lower alpha)
   - Background overlay stays at alpha 0.4 (dark blue tint)
   - Moon sprite appears in upper-right corner with soft silver glow

3. **Werewolf Kill Selection** (spectator "peek" mode)
   - If spectator has "Show Roles" toggled ON:
     - Werewolf agents get subtle red glow around their sprite (`sprite.setTint(0xff4444)` with pulsing tween)
     - A red targeting reticle appears over the chosen victim, then locks in with a brief flash
     - Text: `"🐺 The wolves chose their target..."` in event feed
   - If "Show Roles" OFF:
     - Just show `"The village sleeps..."` with ambient night sounds visual cue
     - No information revealed until dawn

4. **Seer Inspection** (peek mode only)
   - Seer sprite gets a cyan crystal-ball glow
   - Inspected agent gets a brief scan-line effect (top-to-bottom sweep, 0.5s)
   - Result appears as a small card above the Seer: ✓ Villager (green) or ✗ Werewolf (red)
   - This card fades after 3s

5. **Doctor Protection** (peek mode only)
   - Doctor sprite shows a soft golden shield aura
   - Protected agent gets a brief golden shimmer overlay

6. **Night Resolution**
   - Dramatic 2-second pause with `"Dawn approaches..."` text
   - Night overlay fades out (1.5s transition to warm dawn tint)

#### Dawn — The Reveal

1. **"DAWN BREAKS" Title Card**
   - Warm sunrise color sweep from bottom-left
   - Title: `"DAWN BREAKS"` in warm gold text
   - Fire particle emitter returns to full intensity

2. **Death Announcement** (if someone died)
   - Camera slowly pans to the victim's seat
   - Victim sprite plays a brief "fade-out" animation: shake (3 frames), then dissolve to grayscale
   - Tombstone sprite slides up from below the seat
   - Event feed: `"☠️ [Agent Name] was found dead at dawn."` in `--accent-red`
   - Their role is NOT revealed (mystery preserved)
   - 3-second pause to let it sink in

3. **Save Announcement** (if Doctor saved someone)
   - Instead of death: golden sparkle effect on the target
   - Event feed: `"✨ Someone was saved by the Doctor!"` in `--accent-gold`
   - No indication of WHO was saved (keeps Doctor anonymous)

#### Day Phase — Discussion & Voting

This is the heart of the Werewolf experience for spectators.

**Discussion Sub-Phase:**

```
╔══════════════════════════════════════════════════════════╗
║  GAME VIEW (left 65%)          ║  DISCUSSION (right 35%) ║
╠════════════════════════════════╬═════════════════════════╣
║                                ║  ── Day 2 Discussion ── ║
║   [Campfire circle with        ║                         ║
║    agents seated, current      ║  Klaus: 💬              ║
║    speaker highlighted with    ║  "I think Isabella has  ║
║    golden spotlight glow]      ║   been too quiet. She   ║
║                                ║   didn't seem surprised ║
║                                ║   by the death."        ║
║   ┌─────────────────────┐     ║                         ║
║   │ 🔊 Klaus speaking... │     ║  Isabella: 💬           ║
║   └─────────────────────┘     ║  "That's ridiculous!    ║
║                                ║   I was sleeping. Why   ║
║                                ║   are you deflecting?"  ║
║                                ║                         ║
║   [Speech bubble appears       ║  Carlos: 💬             ║
║    above active speaker with   ║  "Let's not rush. But  ║
║    their current line]         ║   Klaus raises a good   ║
║                                ║   point about silence." ║
║                                ║                         ║
║                                ║  ── Accusations ──      ║
║                                ║  Klaus → Isabella  🔴   ║
║                                ║  Isabella → Klaus  🔴   ║
╚════════════════════════════════╩═════════════════════════╝
```

- **Active speaker:** The agent currently speaking gets a golden spotlight (soft radial gradient below their sprite, pulsing). A pixel-art speech bubble appears above them showing the first ~40 chars of their line.
- **Transcript panel:** Right sidebar shows full scrollable transcript. Each line has the speaker's name (in their team color if roles revealed), pronunciatio emoji, and full text. New lines animate in with a typewriter effect (characters appear one by one, 30ms per char, via `setInterval`).
- **Accusation lines:** When Agent A accuses Agent B, a dashed red line (`0xe94560`, 2px, animated dash pattern) draws between their two seats on the game view. Multiple accusations create a web of suspicion visible on the board.
- **Defense highlight:** When an accused agent speaks, their spotlight turns white (defense mode), and a small shield icon appears beside their name in the transcript.

**Voting Sub-Phase:**

```
╔══════════════════════════════════════════════════════════╗
║  GAME VIEW                     ║  VOTE TRACKER          ║
╠════════════════════════════════╬═════════════════════════╣
║                                ║  ── Voting Round ──     ║
║   [Each agent's seat now       ║                         ║
║    shows a ballot icon.        ║  Isabella  ███░░  3     ║
║    As they vote, a dotted      ║  Klaus     ██░░░  2     ║
║    line connects voter to      ║  Carlos    ░░░░░  0     ║
║    target with checkmark]      ║  Maria     ░░░░░  0     ║
║                                ║                         ║
║   Vote lines:                  ║  Majority needed: 3     ║
║   Carlos ──✓──▶ Isabella      ║                         ║
║   Klaus  ──✓──▶ Carlos        ║  ── Vote Log ──         ║
║   Maria  ──✓──▶ Isabella      ║  Carlos → Isabella      ║
║   Abigail──✓──▶ Isabella      ║  Klaus → Carlos         ║
║                                ║  Maria → Isabella       ║
║                                ║  Abigail → Isabella     ║
║                                ║  Isabella → Klaus       ║
║                                ║                         ║
║                                ║  ⚖️ MAJORITY REACHED   ║
╚════════════════════════════════╩═════════════════════════╝
```

- **Vote tracker:** Horizontal bar chart with live-updating bars. Each bar uses the agent's name, fills progressively as votes come in. Bar color: `--accent-red`. Majority threshold shown as a dashed vertical line.
- **Vote lines on game view:** As each agent votes, a dotted line (white, 1px) draws from voter's seat to target's seat with a small checkmark animation at the end. Creates a visual "web" of votes.
- **Majority reached:** When a player hits majority, their bar flashes gold, a gavel sound-cue icon appears, and the text `"⚖️ MAJORITY REACHED — [Name] will be eliminated"` appears as a banner across the bottom.

**Elimination:**

- 3-second dramatic countdown: `"3... 2... 1..."` in large pixel text, each number with screen shake
- Eliminated agent's sprite: brief red flash, then slow grayscale fade (1.5s tween)
- Tombstone slide-up animation on their seat
- Event feed: `"⚰️ [Name] has been eliminated by the village."`
- Role NOT revealed during game (preserves mystery)
- Win condition check: brief `"Checking survivors..."` text with loading dots (0.5s fake suspense), then either continues or triggers endgame

#### Endgame — The Big Reveal

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║               🏆 VILLAGE WINS! 🏆                       ║
║                                                          ║
║   All werewolves have been eliminated!                   ║
║                                                          ║
║   ── ROLE REVEAL ──                                      ║
║                                                          ║
║   [Each agent's role card flips one by one               ║
║    with a card-flip animation. 0.5s per card.            ║
║    Werewolves reveal with a red flash.                   ║
║    Villagers reveal with green sparkle.]                 ║
║                                                          ║
║   Isabella ── 🐺 WEREWOLF     Klaus ── 🐺 WEREWOLF     ║
║   Carlos   ── 👁 SEER         Maria ── 🏘 VILLAGER     ║
║   Eddy     ── 🏘 VILLAGER     Abigail── 🏘 VILLAGER   ║
║                                                          ║
║   ── MVP ──                                              ║
║   ⭐ Carlos (Seer) — correctly identified Isabella      ║
║                                                          ║
║   ── RATING CHANGES ──                                   ║
║   Carlos:  1520 → 1543  (+23) ▲                         ║
║   Maria:   1480 → 1498  (+18) ▲                         ║
║   Abigail: 1500 → 1515  (+15) ▲                         ║
║   Isabella:1550 → 1532  (-18) ▼                         ║
║   Klaus:   1490 → 1472  (-18) ▼                         ║
║   Eddy:    1510 → 1510  ( 0 ) ☠️ (died N1)              ║
║                                                          ║
║   [Full Transcript]  [Watch Replay]  [Return to Town]   ║
╚══════════════════════════════════════════════════════════╝
```

- **Victory banner:** Full-width animated banner with confetti particles (Phaser particle emitter: small colored squares falling, 50 particles, 3-second burst). Text in `Press Start 2P` 1rem with golden glow.
- **Role reveal:** Sequential card-flip animation — each agent's "?" badge flips to show their role card (128×192px asset from the asset plan). 0.5s per card with a satisfying "flip" visual (scale X from 1 to 0 to 1 with texture swap at midpoint). Werewolf reveals trigger a brief red screen-edge flash.
- **MVP highlight:** Golden star icon + agent name + reason. Determined by: Seer correct ID > Werewolf survived longest > Villager who cast deciding vote.
- **Rating changes:** Animated number counter (rolls from old to new rating over 1s). Green `▲` for gain, red `▼` for loss. Uses `font-variant-numeric: tabular-nums` for clean alignment.
- **Action buttons:** Three pixel-art buttons at the bottom:
  - `[Full Transcript]` → Opens scrollable modal with all rounds
  - `[Watch Replay]` → Replays the game from round 1 (using stored events)
  - `[Return to Town]` → Exits spectator view, agents walk back to normal locations

---

### 12.2 Anaconda Poker Spectator Experience

#### The Card Table

A dedicated Phaser scene showing a cozy pub interior:

- **Background:** Warm wooden pub interior. Pixel-art wallpaper with dark brown wood panels, a warm lantern hanging above the table (animated flicker glow). Window in background showing the town (optional parallax connection to the actual town map).
- **Table:** Oval poker table with forest-green felt surface (the poker_table.png asset). Centered in the scene. Thick wooden rim with subtle pixel wood-grain.
- **Seats:** 3-7 positions around the table edge. Each position has:
  - Agent's chibi sprite (sitting pose, "down" frame)
  - Name label below (VT323 11px)
  - Chip stack visualization to their right (stacked chip sprites, height proportional to chip count)
  - Card zone in front of them (7 card slots, then 5 after discard)
- **Center:** Pot area with coin stack that grows as bets accumulate. Dealer button (white "D" circle) at current dealer position.

#### Card Dealing — The Opening

1. **"DEALING..." title** appears briefly (1s, fade in/out)
2. Cards animate from dealer position, flying one at a time to each player in clockwise order
   - Each card: the card_back.png asset, 64×96px, tweened from center to player's card zone
   - Staggered: 100ms delay between each card
   - Total dealing animation: ~5 seconds for 7 cards × N players
3. **Spectator "Peek" toggle:**
   - OFF (default): All cards show backs. Spectator watches blind, like a real poker broadcast delay.
   - ON: Cards flip to show faces. Each player's hand is visible. A "hand strength" label appears:
     ```
     Isabella: 7♠ 7♥ K♦ Q♣ 3♠ 9♥ 2♦
     PAIR OF SEVENS
     ```
   - Hand strength labels use the hand_ranks.png asset sheet, positioned below each player's cards
4. **First betting round** begins immediately after deal

#### Betting Rounds — The Drama Engine

Each betting action is a mini-event with visual feedback:

```
╔══════════════════════════════════════════════════════════╗
║  TABLE VIEW (left 70%)         ║  ACTION LOG (right 30%) ║
╠════════════════════════════════╬═════════════════════════╣
║                                ║  ── Betting Round 1 ──  ║
║   [Poker table with agents     ║                         ║
║    and cards. Active bettor    ║  Klaus:    BET 20   💰  ║
║    has golden highlight]       ║  Carlos:   CALL     💰  ║
║                                ║  Isabella: CALL     💰  ║
║   [Pot: 80 coins]              ║  Maria:    RAISE 40 🔥  ║
║   [chip pile in center]        ║  Klaus:    CALL     💰  ║
║                                ║  Carlos:   FOLD     ❌  ║
║   Active: Maria               ║  Isabella: CALL     💰  ║
║   [RAISE 40] glow effect      ║                         ║
║                                ║  Pot: 200 coins         ║
║   Carlos: [cards face-down]    ║  Still in: 3 players    ║
║   (FOLDED - red X overlay)     ║                         ║
╚════════════════════════════════╩═════════════════════════╝
```

- **Active bettor highlight:** Golden pulsing glow around the current player's seat area. Their name label turns gold.
- **Action animations:**
  - `CHECK` — Small checkmark appears above player (green ✓), fades after 1.5s
  - `BET/CALL` — Chip sprites slide from player's stack to center pot. Number label (`+20`) floats up and fades. Pot counter increments with a satisfying roll animation.
  - `RAISE` — Same as BET but with a larger chip push, brief screen-edge orange flash, and `🔥` icon. The raise amount appears in larger, bolder text.
  - `FOLD` — Player's cards slide together, flip face-down, and a semi-transparent red `X` overlays their card zone. Their sprite gets a slight desaturate tint. Action log shows `❌ FOLD`.
  - `ALL IN` — **The big moment.** Slow-motion effect (tweens at 0.5x speed for 1.5s). All of the player's remaining chips slide dramatically to center. A golden glow radiates from the pot. Text: `"💎 ALL IN!"` in large `Press Start 2P` text, centered, with a screen shake (2px amplitude, 0.5s). Their chip stack shows `0` after.
- **Pot counter:** Always visible in center of table. Animated number counter that rolls up on each bet. Uses the coin stack visualization assets (switches between pot_small → pot_medium → pot_large → pot_jackpot as pot grows past thresholds at 50/150/300/500).

#### Pass Rounds — The Social Mechanic

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║            ── PASS 3 CARDS TO LEFT ──                    ║
║                                                          ║
║       Isabella ──[3 cards]──▶ Klaus                     ║
║       Klaus    ──[3 cards]──▶ Carlos                    ║
║       Carlos   ──[3 cards]──▶ Maria                     ║
║       Maria    ──[3 cards]──▶ Isabella                  ║
║                                                          ║
║   [Reaction emojis appear above agents:]                 ║
║   Isabella: 😊 (got good cards)                         ║
║   Klaus: 😤 (got bad cards)                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

- **Phase header:** `"PASS 3 TO LEFT"` / `"PASS 2 TO RIGHT"` / `"PASS 1 TO LEFT"` — appears as a banner above the table, `Press Start 2P` 0.6rem, `--accent-cyan`
- **Card slide animation:** 3 (or 2 or 1) card_back sprites slide in an arc from the passer's card zone to the receiver's card zone. The arc follows a bezier curve (rise up 30px, then down to destination) for a satisfying "toss" feel. Duration: 0.6s per card, staggered 0.15s.
- **Directional arrows:** Faint dotted arrow lines showing the pass direction (clockwise or counterclockwise), visible during the pass phase.
- **Reaction emojis:** After receiving cards, each agent briefly shows a reaction emote (24×24 sprite from the emotes sheet) above their head:
  - Hand improved: 😊 or ⭐
  - Hand got worse: 😤 or 😰
  - No change: 🤔
  - Determined by comparing hand rank before/after pass (from game state)

#### Card Reveal Rounds — The Climax

This is the most dramatic part. All players simultaneously reveal one card at a time, with betting between each reveal.

```
╔══════════════════════════════════════════════════════════╗
║                  ── REVEAL 3 of 5 ──                     ║
║                                                          ║
║   Isabella:  [K♦] [Q♣] [??] [??] [??]   TWO PAIR       ║
║   Klaus:     [A♠] [A♣] [??] [??] [??]   PAIR OF ACES   ║
║   Maria:     [Q♠] [J♠] [10♠][??] [??]   FLUSH DRAW! ✨  ║
║                                                          ║
║   [Currently best: Klaus — PAIR OF ACES]                ║
║                                                          ║
║   Pot: 380 coins                                        ║
║                                                          ║
║   ── Betting Round ──                                    ║
║   Klaus: RAISE 50 🔥 (confident with aces showing)      ║
║   Maria: CALL (chasing the flush)                       ║
║   Isabella: FOLD ❌                                      ║
╚══════════════════════════════════════════════════════════╝
```

- **Reveal animation:** Each player's next hidden card simultaneously flips. The flip animation: card scales X from 1→0 (0.2s), texture swaps from card_back to card_face, then scales X from 0→1 (0.2s). Total 0.4s per card. All players reveal at the same time (synchronized).
- **Card enlargement:** The newly revealed card briefly scales up to 1.3x (0.3s tween) then settles back to 1x, drawing attention to it.
- **Hand strength update:** After each reveal, the hand rank label below each player updates. If it improved (e.g., "High Card" → "Pair"), the label briefly flashes green. The label uses the hand_ranks.png asset, appropriate ribbon tier.
- **Best hand indicator:** The player with the currently strongest visible hand gets a subtle golden glow border around their card zone. Text at bottom: `"Currently best: [Name] — [HAND RANK]"`.
- **Tension on later reveals:** Reveals 4 and 5 get increasingly dramatic:
  - Reveal 4: Slight slow-motion (1.2x duration). Brief dramatic pause before flip.
  - Reveal 5 (final): 2-second dramatic pause with a pulsing `"FINAL CARD..."` text. Then all cards flip. If any player's hand jumps significantly (e.g., hits a flush or full house), a burst particle effect (gold sparkles) erupts from their card zone.

#### Showdown — The Winner

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              🏆 MARIA WINS THE POT! 🏆                  ║
║                                                          ║
║   ── WINNING HAND ──                                     ║
║   [Q♠] [J♠] [10♠] [9♠] [4♦]                            ║
║          FLUSH (Spades)                                  ║
║                                                          ║
║   [Enlarged cards in center, gold border, sparkle fx]    ║
║                                                          ║
║   ── RESULTS ──                                          ║
║   Maria:    100 → 380  (+280) 💰💰💰                    ║
║   Klaus:    100 →   0  (-100) 📉                        ║
║   Isabella: 100 →   0  (-100) 📉                        ║
║   Carlos:   100 →   0  (-100) 📉 (folded round 1)       ║
║                                                          ║
║   ── RATING CHANGES ──                                   ║
║   Maria:    1480 → 1512  (+32) ▲                        ║
║   Klaus:    1520 → 1502  (-18) ▼                        ║
║   Isabella: 1500 → 1488  (-12) ▼                        ║
║   Carlos:   1510 → 1505  ( -5) ▼                        ║
║                                                          ║
║   ── TABLE STATS ──                                      ║
║   Biggest pot: 380 · Best hand: Flush                   ║
║   Biggest bluff: Klaus raised with pair (round 4)       ║
║                                                          ║
║   [Hand History]  [Watch Replay]  [Return to Town]      ║
╚══════════════════════════════════════════════════════════╝
```

- **Winner announcement:** Pot coins fly from center to winner's chip stack (particle trail of coin sprites along a bezier curve). Winner's name pulses in gold. Confetti burst (smaller than Werewolf — this is a pub game, not a championship).
- **Winning hand display:** Winner's 5 cards enlarge and float to center of table (1.5s tween). Each card gets a gold border glow. Hand rank label appears below in the gold-tier ribbon asset.
- **Profit/loss counter:** Each player's result shows with animated counter rolling from starting chips to ending chips. Green numbers for profit, red for loss. Coin emoji density indicates magnitude (1 coin for small win, 3 for big).
- **Table stats:** Fun facts about the hand: biggest pot, best hand, biggest bluff (agent who bet/raised the most relative to their actual hand strength).

---

### 12.3 Shared Spectator Infrastructure

#### Spectator State Polling

The spectator view polls a dedicated endpoint at 1-second intervals (faster than normal town polling):

```javascript
const SPECTATOR_POLL_MS = 1000;

async function pollMatchState(matchId) {
  const data = await apiFetch(`/scenarios/matches/${matchId}/spectate`);
  // data.public_state contains only what spectators should see:
  // - phase, round, alive players, votes, pot, revealed cards, etc.
  // - NO hidden roles or face-down cards (unless "peek" mode)
  updateSpectatorView(data.public_state);
}
```

#### Peek Mode Toggle

A toggle in the spectator topbar:

```html
<label class="peek-toggle">
  <input type="checkbox" id="peekMode" />
  <span>👁 Show Hidden Info</span>
</label>
```

- OFF: Spectator sees only public information (just like a player would)
- ON: All roles, all cards, all votes revealed. This is the "broadcaster" view.
- Default: OFF (more fun to watch blind)

#### Picture-in-Picture Town View

A small 200×150px minimap in the bottom-left corner showing the town map with a red dot indicating where the game is happening. Click to return to town view.

```css
.pip-minimap {
  position: absolute;
  bottom: 12px; left: 12px;
  width: 200px; height: 150px;
  border: 2px solid var(--border-color);
  background: var(--bg-deep);
  opacity: 0.8;
  cursor: pointer;
  z-index: 50;
  image-rendering: pixelated;
}
.pip-minimap:hover { opacity: 1; border-color: var(--accent-cyan); }
```

#### Post-Match Agent Behavior

When a match resolves, the agents are released back to normal town behavior:

- **Werewolf:** Winners walk away with ⭐ pronunciatio, losers walk away with 😔. The walk-back is visible on the town map.
- **Anaconda:** Winner walks away with 💰, losers with varying levels of disappointment based on loss size.
- **Schedule recomposition:** triggers automatically so agents resume their day naturally.
- The pronunciatio persists for ~3 steps before reverting to normal activity emoji.

---

### 12.4 Admin Console Additions

New sections in `index.html`:

#### Scenario Browser

```html
<section id="scenarios" class="frame section">
  <h2>Game Scenarios</h2>
  <div class="scenario-grid">
    <!-- Dynamically populated from GET /api/v1/scenarios -->
    <article class="scenario-card">
      <img src="./assets/scenarios/werewolf/icon.png" />
      <h3>Werewolf</h3>
      <p>Social deduction · 6-10 players</p>
      <div class="scenario-meta">
        <span>Queue: 3/6</span>
        <span>Your rating: 1520 (Silver)</span>
      </div>
      <button class="btn btn-small">Join Queue</button>
    </article>
  </div>
</section>
```

#### Active Games Panel

Shows all ongoing matches across all towns with live status:

```html
<div class="active-games">
  <div class="game-card">
    <span class="game-icon">🐺</span>
    <span class="game-title">Werewolf — Night 2</span>
    <span class="game-status">5 alive · step 45</span>
    <a href="town.html?match=M-001" class="btn btn-small">Watch</a>
  </div>
</div>
```

#### Leaderboard

Per-scenario ranked list with tier badges:

```html
<div class="leaderboard">
  <div class="lb-row">
    <span class="lb-rank">#1</span>
    <img class="lb-badge" src="./assets/scenarios/shared/badge_gold.png" />
    <span class="lb-name">Isabella Rodriguez</span>
    <span class="lb-rating">2150</span>
    <span class="lb-record">W12 L3</span>
  </div>
</div>
```

#### Wallet Manager (Admin)

View and modify agent balances. Shows transaction history for Anaconda buy-ins and winnings.

---

## 🎨 13. Asset Generation Plan

### Art Direction

**Target aesthetic:** Half-pixel cozy indie game — matching V-Valley's existing chibi pixel-art style:
- **Selective outlining** (dark saturated borders, not pure black)
- **3-tone cell shading** (highlight, midtone, shadow)
- **Muted pastels** with **high-contrast accents**
- **Rounded shapes**, soft corners, whimsical proportions
- **32px grid alignment** for tiles, 48-64px for standalone UI elements
- **Limited color palette** (8-12 colors per asset)

**Tool:** Use **OpenAI image generation (DALL-E 3/GPT-Image)** or **Gemini Imagen** — whichever produces better pixel art consistency. Generate at 1024×1024 then downscale + clean up.

---

### Asset Category 1: Werewolf Game Assets

#### 1.1 Role Cards (7 cards)

Each card: 128×192px pixel art card with character illustration + role name.

**PROMPT — Villager Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a warm tan parchment background. The card shows a chibi pixel-art villager character 
(2-head-tall proportions, large expressive eyes) holding a pitchfork, wearing a simple brown tunic 
and green hat. Below the character is a pixel-art banner reading "VILLAGER" in a warm serif pixel 
font. The card border is a thick 3px wooden brown frame with small decorative corner scrolls. 
Muted earth-tone palette: browns, greens, tans, cream. 16-bit RPG aesthetic with selective 
outlining (no pure black outlines — use dark brown). Soft top-down lighting. Clean pixel edges, 
no anti-aliasing, no gradients. Transparent background.
```

**PROMPT — Werewolf Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a dark indigo/midnight blue parchment background with subtle moon motif. The card 
shows a chibi pixel-art werewolf character (2-head-tall proportions) — a menacing but cute wolf 
standing upright with glowing amber eyes, dark fur with purple highlights, sharp ears and small 
fangs visible. Below the character is a pixel-art banner reading "WEREWOLF" in a scratchy red 
pixel font. The card border is a thick 3px dark purple frame with claw-mark decorations at 
corners. Palette: deep indigo, midnight purple, amber, blood red accents, silver highlights. 
16-bit RPG aesthetic with selective outlining. Soft moonlight illumination from above. Clean pixel 
edges, no anti-aliasing. Transparent background.
```

**PROMPT — Seer Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a mystical teal/deep green parchment background with subtle star motifs. The card 
shows a chibi pixel-art fortune teller character (2-head-tall proportions) — a hooded figure in 
flowing emerald robes holding a glowing crystal ball emitting soft cyan light. Large mysterious 
eyes peek from under the hood. Below the character is a pixel-art banner reading "SEER" in a 
glowing cyan pixel font. The card border is a thick 3px emerald green frame with small star/eye 
decorations at corners. Palette: deep greens, teal, cyan glow, emerald, cream. 16-bit RPG 
aesthetic with selective outlining. Magical ambient lighting. Clean pixel edges, no anti-aliasing. 
Transparent background.
```

**PROMPT — Doctor Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a soft white/cream parchment background with subtle cross motifs. The card shows a 
chibi pixel-art healer character (2-head-tall proportions) — wearing a white apron with a red 
cross, carrying a small leather medical bag, kind warm eyes, pink cheeks. Below the character is 
a pixel-art banner reading "DOCTOR" in a clean red pixel font. The card border is a thick 3px 
white frame with small heart decorations at corners. Palette: whites, creams, soft reds, warm 
pinks, leather brown. 16-bit RPG aesthetic with selective outlining. Warm gentle lighting. Clean 
pixel edges, no anti-aliasing. Transparent background.
```

**PROMPT — Hunter Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and an olive/forest green parchment background. The card shows a chibi pixel-art ranger 
character (2-head-tall proportions) — wearing a leather vest and feathered cap, carrying a small 
crossbow, determined squinted eyes. Below the character is a pixel-art banner reading "HUNTER" 
in a bold olive pixel font. The card border is a thick 3px leather brown frame with small arrow 
decorations at corners. Palette: forest greens, olive, leather browns, copper accents. 16-bit RPG 
aesthetic with selective outlining. Dappled forest lighting. Clean pixel edges, no anti-aliasing. 
Transparent background.
```

**PROMPT — Witch Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a deep plum/magenta parchment background with subtle potion bottle motifs. The card 
shows a chibi pixel-art witch character (2-head-tall proportions) — wearing a pointed hat with a 
buckle, holding two small potion bottles (one green/save, one red/poison), mischievous smile, 
sparkling eyes. Below the character is a pixel-art banner reading "WITCH" in a swirly purple 
pixel font. The card border is a thick 3px deep plum frame with small cauldron decorations at 
corners. Palette: deep plum, magenta, emerald green, ruby red, gold sparkle. 16-bit RPG aesthetic 
with selective outlining. Mysterious candlelight glow. Clean pixel edges, no anti-aliasing. 
Transparent background.
```

**PROMPT — Alpha Werewolf Card:**
```
A pixel art game card for a cozy indie village simulation game. 128x192 pixel card with rounded 
corners and a blood-red/crimson parchment background with crescent moon motif. The card shows a 
chibi pixel-art alpha werewolf character (2-head-tall proportions) — larger and more imposing 
than regular werewolf, with a scar across one eye, crown-like fur ridge, burning red eyes, 
slightly larger fangs. Below the character is a pixel-art banner reading "ALPHA" in a bold 
crimson pixel font with gold edges. The card border is a thick 3px crimson frame with wolf paw 
print decorations at corners. Palette: crimson, blood red, black, silver, gold accents. 16-bit 
RPG aesthetic with selective outlining. Dramatic red moonlight. Clean pixel edges, no anti-
aliasing. Transparent background.
```

#### 1.2 Game Board / Table Asset

**PROMPT — Werewolf Game Circle:**
```
A top-down pixel art illustration of a cozy campfire circle for a village simulation game. 
The scene shows a warm crackling pixel-art campfire in the center surrounded by 10 small 
tree-stump seats arranged in a circle on grass tiles. Warm orange firelight illuminates the 
stumps. Small details: a few scattered autumn leaves, tiny mushrooms near stumps, a lantern 
on one stump. The ground is soft green grass with brown dirt patches. Matches a 32px tile grid 
RPG aesthetic. Muted earth tones with warm orange fire glow. 256x256 pixels. Selective 
outlining (dark green for grass, dark brown for wood). No characters present — empty seats 
waiting for players. Cozy indie game aesthetic, 16-bit style. No anti-aliasing.
```

#### 1.3 UI Elements

**PROMPT — Day/Night Phase Indicator:**
```
Two pixel art phase indicator icons for a cozy indie game UI. 
Icon 1 "DAY": A warm cheerful pixel-art sun with soft rays, set against a light blue sky 
circle. 64x64 pixels. Palette: warm yellow, soft orange, light blue, white.
Icon 2 "NIGHT": A serene crescent moon with tiny stars, set against a deep indigo circle. 
64x64 pixels. Palette: silver, soft purple, deep indigo, twinkling white dots.
Both icons have thick 2px rounded borders matching their background colors. 16-bit RPG aesthetic, 
selective outlining, no anti-aliasing, no gradients. Transparent backgrounds.
```

**PROMPT — Vote Tokens:**
```
A set of 4 small pixel art voting token icons for a cozy indie game. Each token is 32x32 pixels.
Token 1: A red "ACCUSE" token — small red circle with a pointing finger silhouette inside.
Token 2: A green "DEFEND" token — small green circle with a shield silhouette inside.
Token 3: A gray "SKIP" token — small gray circle with a dash/minus silhouette inside.
Token 4: A gold "SAVE" token — small gold circle with a heart silhouette inside.
All tokens have thick 2px borders in a darker shade of their base color. 16-bit pixel art, 
selective outlining, clean pixel edges. Transparent backgrounds. Arrange in a 2x2 grid.
```

**PROMPT — Elimination Tombstone:**
```
A small pixel art tombstone sprite for a cozy indie village game. 48x48 pixels. A cute rounded 
tombstone made of gray stone with a small flower (pink) growing at its base. The stone has a 
simple "RIP" engraving and a tiny chibi ghost floating above it looking surprised (comical, not 
scary). Muted gray, soft pink, translucent white ghost. Selective outlining in dark gray. 
16-bit RPG aesthetic. Transparent background. Cozy and whimsical, not morbid.
```

---

### Asset Category 2: Anaconda Poker Assets

#### 2.1 Playing Cards (Full Deck)

**We need 52 cards + 1 card back. Generate in sheets of 13 (one per suit).**

**PROMPT — Card Back Design:**
```
A pixel art playing card back design for a cozy indie village simulation game. 64x96 pixel card 
with rounded corners. The design features an intricate repeating diamond pattern in deep burgundy 
and cream, with a small pixel-art village house silhouette in the center surrounded by tiny 
stars. The border is a thick 2px gold frame. Palette: deep burgundy, cream, gold, dark brown. 
Matches a 16-bit RPG aesthetic. Selective outlining, no anti-aliasing, no gradients. 
Transparent background. The design should look like a well-worn, cozy deck of cards from a 
village pub.
```

**PROMPT — Hearts Suit Sheet (13 cards):**
```
A pixel art sprite sheet of 13 playing cards arranged in a row for a cozy indie game. Each card 
is 64x96 pixels with rounded corners and a cream/off-white background. The cards show Ace 
through King of Hearts. Each card has:
- A large rank indicator (A, 2-10, J, Q, K) in the top-left in deep red pixel font
- A small red pixel-art heart suit symbol below the rank
- Center illustration: number cards show arranged heart symbols; face cards (J, Q, K) show tiny 
  chibi pixel-art royal characters (Jack=young page, Queen=elegant lady, King=bearded monarch) 
  in red and gold outfits; Ace shows one large ornate heart
- Thick 2px border in warm rose-red
Palette: cream, deep red, rose pink, gold for face cards. 16-bit RPG aesthetic, selective 
outlining, clean pixel edges. The overall look should be warm and cozy like cards from a 
village tavern. Transparent background.
```

**PROMPT — Diamonds Suit Sheet:**
```
Same as Hearts prompt but replace hearts with diamond suit symbols in warm orange-red. 
Diamond symbols are small pixel-art diamond shapes. Face card characters wear orange and 
gold outfits. Border color: warm amber-orange. Palette: cream, amber, orange-red, gold.
```

**PROMPT — Clubs Suit Sheet:**
```
Same as Hearts prompt but replace with club/clover suit symbols in deep forest green. 
Club symbols are small pixel-art three-leaf clovers. Face card characters wear green and 
brown outfits. Border color: deep forest green. Palette: cream, forest green, olive, bronze.
```

**PROMPT — Spades Suit Sheet:**
```
Same as Hearts prompt but replace with spade suit symbols in deep navy/midnight blue. 
Spade symbols are small pixel-art pointed spade shapes. Face card characters wear blue and 
silver outfits. Border color: deep navy. Palette: cream, navy, slate blue, silver.
```

#### 2.2 Poker Table

**PROMPT — Card Table (Top-Down):**
```
A top-down pixel art illustration of a cozy pub poker table for a village simulation game. 
The table is oval/rounded rectangular with a rich forest-green felt surface and a thick wooden 
brown rim. Around the table are 7 small wooden chair positions (indicated by small semi-circles 
at the edge). The center of the table has a small area for the pot (slightly darker green 
circle). Each player position has a small rectangular card zone (slightly lighter green). 
The wood grain on the rim is suggested with 2-3 pixel-wide brown stripes. A small pixel-art 
dealer button (white circle with "D") sits near one position. 320x256 pixels. Palette: forest 
green felt, warm brown wood, cream accents. 16-bit RPG aesthetic, 32px tile-aligned. Selective 
outlining in dark brown and dark green. Cozy tavern feel. Transparent background.
```

#### 2.3 Chip Stacks

**PROMPT — Poker Chips (5 denominations):**
```
A pixel art sprite sheet of 5 poker chip designs for a cozy indie village game. Each chip is 
24x24 pixels, shown from a slight 3/4 top-down angle to show thickness (3-4 pixels tall).
Chip 1 (1 coin): White/cream with a small "1" — simple village token
Chip 2 (5 coins): Blue with a small "5" — copper rim accent  
Chip 3 (10 coins): Green with "10" — small clover decoration
Chip 4 (25 coins): Red with "25" — small diamond decoration
Chip 5 (100 coins): Gold with "100" — small crown decoration
Each chip has a subtle circular edge pattern and a darker shade on the bottom edge for depth. 
3-tone shading per chip. 16-bit RPG aesthetic, selective outlining. Arrange in a row. 
Transparent background.
```

**PROMPT — Coin Stack Visualization (for pot display):**
```
A set of 4 pixel art coin stack illustrations for a cozy indie game, showing increasing amounts:
Stack 1: 3 gold coins stacked (small pot) — 24x32 pixels
Stack 2: 6 gold coins in a short pile (medium pot) — 24x40 pixels
Stack 3: 12 gold coins in a tall pile with some toppling (large pot) — 32x48 pixels
Stack 4: Overflowing treasure pile with coins and a small gem (jackpot) — 48x48 pixels
Gold coins with warm yellow highlights and bronze shadows. Small sparkle effects (1-2 white 
pixels) on top coins. 16-bit RPG pixel art style. Selective outlining in dark bronze. 
Transparent backgrounds. Cozy indie game aesthetic.
```

#### 2.4 UI Elements

**PROMPT — Betting Action Buttons:**
```
A set of 5 pixel art UI buttons for a cozy indie poker game. Each button is 80x32 pixels with 
rounded corners and a 2px border.
Button 1 "FOLD": Dark red background, cream text "FOLD", small X icon
Button 2 "CHECK": Soft blue background, cream text "CHECK", small checkmark icon
Button 3 "CALL": Warm green background, cream text "CALL", small coin icon
Button 4 "RAISE": Bold orange background, cream text "RAISE", small up-arrow + coin icon
Button 5 "ALL IN": Shimmering gold background, bold cream text "ALL IN", star burst icon
Each button has a 1px darker bottom edge for depth (pushed button feel). 16-bit pixel art 
style, selective outlining, no anti-aliasing. Arrange in a row. Transparent background.
```

**PROMPT — Hand Rank Labels:**
```
A set of 10 pixel art hand rank label banners for a cozy indie poker game. Each label is 
120x24 pixels — a small ribbon/banner shape with text.
Labels (in order): "HIGH CARD", "ONE PAIR", "TWO PAIR", "THREE OF A KIND", "STRAIGHT", 
"FLUSH", "FULL HOUSE", "FOUR OF A KIND", "STRAIGHT FLUSH", "ROYAL FLUSH"
Lower ranks use muted gray/brown ribbons. Higher ranks progress through blue, purple, gold.
"ROYAL FLUSH" has a small sparkle effect and gold ribbon. All text in clean pixel font. 
16-bit RPG aesthetic. Selective outlining. Arrange vertically. Transparent backgrounds.
```

---

### Asset Category 3: Shared UI Assets

#### 3.1 Matchmaking Queue UI

**PROMPT — Queue Status Panel:**
```
A pixel art UI panel design for a cozy indie game matchmaking queue. 320x200 pixel panel 
with a warm wooden frame border (8px thick, brown wood grain). Inside: cream parchment 
background. At the top: a pixel-art banner reading "FINDING MATCH..." in warm brown text. 
Below: 6 small circular avatar slots arranged in a row (32x32 each) — 3 filled with generic 
chibi silhouettes (soft blue) and 3 empty (dashed circle outline). Below the slots: a small 
pixel-art hourglass animation frame (sand falling) and text "3/6 Players". At the bottom: 
a warm red "CANCEL" button (80x24). Cozy indie game aesthetic, 16-bit style, selective 
outlining. Muted warm palette. Transparent background.
```

#### 3.2 Rating Display

**PROMPT — Rating Tier Badges (6 badges):**
```
A pixel art sprite sheet of 6 rating tier badges for a cozy indie game ranking system. Each 
badge is 48x48 pixels, shield-shaped with a small icon and border.
Badge 1 "Newcomer": Gray stone shield, small question mark, dull gray palette
Badge 2 "Bronze": Copper/brown shield, small sword icon, warm bronze palette
Badge 3 "Silver": Shining silver shield, star icon, cool silver/blue palette  
Badge 4 "Gold": Bright gold shield, crown icon, warm gold palette
Badge 5 "Platinum": Iridescent pink shield, gem icon, pink/magenta palette
Badge 6 "Diamond": Crystalline diamond shield, diamond icon, sparkling cyan/gold palette
Each badge has 3-tone shading and a 1px sparkle highlight on the top edge. Higher tiers 
have more elaborate borders and brighter colors. 16-bit RPG pixel art style, selective 
outlining. Arrange in a row. Transparent background.
```

#### 3.3 Scenario Icons

**PROMPT — Werewolf Scenario Icon:**
```
A pixel art icon for a Werewolf social deduction game in a cozy indie game. 64x64 pixels. 
The icon shows a cute pixel-art wolf howling at a crescent moon against a dark indigo night 
sky circle. The wolf is simplified and chibi-style (not scary), with pointed ears and a 
bushy tail. Small stars twinkle around the moon. The circle has a thick 3px dark purple 
border. Palette: indigo, deep purple, silver moon, amber wolf, twinkling white stars. 
16-bit RPG aesthetic, selective outlining, clean pixel edges. Transparent background.
```

**PROMPT — Anaconda Poker Scenario Icon:**
```
A pixel art icon for an Anaconda poker card game in a cozy indie game. 64x64 pixels. The 
icon shows a small pixel-art poker hand (3 fanned cards, one showing an Ace of Hearts) 
resting on a green felt circle, with 3 small gold coins stacked beside the cards. The 
circle has a thick 3px warm brown wooden border. Palette: forest green felt, cream cards, 
red heart, gold coins, warm brown border. 16-bit RPG aesthetic, selective outlining, clean 
pixel edges. Transparent background.
```

#### 3.4 Emote/Reaction Sprites

**PROMPT — Scenario Emotes (8 emotes):**
```
A pixel art sprite sheet of 8 scenario reaction emotes for a cozy indie game. Each emote 
is 24x24 pixels, designed to float above character heads.
1. "Suspicious" — narrowed eyes with sweat drop (💢)
2. "Accusation" — pointing finger with exclamation mark (☝️)
3. "Innocent" — angel halo with sparkles (😇)
4. "Bluffing" — poker face with tiny sweat bead (😐)
5. "Winning" — star burst with happy eyes (⭐)
6. "Losing" — rain cloud with sad eyes (🌧️)
7. "Thinking" — thought bubble with question mark (🤔)  
8. "All In" — flaming coin with determined eyes (🔥)
Chibi-style, expressive, warm palette matching existing V-Valley sprites. Selective outlining, 
3-tone shading. Arrange in a 4x2 grid. Transparent backgrounds.
```

---

### Asset Generation Workflow

1. **Generate at high res:** Create at 1024×1024 using OpenAI/Gemini with the prompts above
2. **Downscale:** Resize to target pixel dimensions using nearest-neighbor (no interpolation)
3. **Clean up:** Manual pixel-level cleanup in Aseprite/Piskel if needed
4. **Palette lock:** Enforce the V-Valley palette (extract from existing sprites)
5. **Export:** PNG with transparency, organized by category
6. **Atlas:** Create sprite atlas JSON (matching existing `atlas.json` format) for Phaser 3

### File Structure for Generated Assets

```
apps/web/assets/
  scenarios/
    werewolf/
      cards/
        card_back.png
        role_villager.png
        role_werewolf.png
        role_seer.png
        role_doctor.png
        role_hunter.png
        role_witch.png
        role_alpha.png
      board/
        campfire_circle.png
      ui/
        phase_day.png
        phase_night.png
        vote_accuse.png
        vote_defend.png
        vote_skip.png
        vote_save.png
        tombstone.png
      emotes/
        suspicious.png
        accusation.png
        innocent.png
      icon.png                    # scenario list icon
      atlas.json                  # sprite atlas for Phaser

    anaconda/
      cards/
        card_back.png
        hearts_sheet.png          # 13 cards in a row
        diamonds_sheet.png
        clubs_sheet.png
        spades_sheet.png
      board/
        poker_table.png
      chips/
        chip_1.png
        chip_5.png
        chip_10.png
        chip_25.png
        chip_100.png
        pot_small.png
        pot_medium.png
        pot_large.png
        pot_jackpot.png
      ui/
        btn_fold.png
        btn_check.png
        btn_call.png
        btn_raise.png
        btn_allin.png
        hand_ranks.png            # 10 labels in a sheet
      emotes/
        bluffing.png
        allin.png
      icon.png
      atlas.json

    shared/
      queue_panel.png
      rating_badges.png           # 6 badges in a sheet
      thinking.png
      winning.png
      losing.png
```

---

## 📋 14. Implementation Phases

### Phase 1: Foundation (M effort)
- [x] Schema + migrations for all 7 tables
- [x] Storage module (`scenarios.py`) with SQLite implementation
- [x] Card deck + hand evaluator (`cards.py`) as standalone core module
- [x] Elo rating utility (`rating.py`) as standalone core module
- [x] Seed default scenario definitions

### Phase 2: Queue & Matchmaking (L effort)
- [x] Queue join/leave API endpoints
- [x] Match formation algorithm
- [x] Background formation pass on join + each town tick
- [x] Wallet system (create, deduct buy-in, credit winnings, stipend safety net)
- [ ] Inbox notifications

### Phase 3: Core Runtime Integration (L effort)
- [ ] `ScenarioRuntimeState` and `TownScenarioManager` in `packages/vvalley_core/sim`
- [x] Tick hook points in runner path via scenario-aware suppression parameter
- [x] Agent state exposure via `npc.scenario` + `state.scenario_matches`
- [x] Ambient social suppression for in-scenario agents
- [x] Match event recording (powers spectator + replay API)

### Phase 4: Werewolf Scenario (XL effort)
- [x] `WerewolfScenarioController` equivalent in `scenario_matchmaker.py`
- [x] Night phase: role actions (seer/doctor/hunter)
- [x] Day phase: speech + voting events with deterministic heuristics
- [x] Voting/elimination step
- [x] Win condition checking
- [ ] Scenario cognition tasks + heuristic fallbacks per role
- [ ] Memory nodes for key moments

### Phase 5: Anaconda Poker Scenario (XL effort)
- [x] `AnacondaScenarioController` equivalent in `scenario_matchmaker.py`
- [x] Card dealing, pass rounds (3-2-1 pattern)
- [x] Betting rounds (deterministic pressure + fold behavior)
- [x] Discard + arrange reveal order
- [x] Card reveal rounds with betting
- [x] Showdown + pot distribution
- [ ] Scenario cognition tasks + heuristic fallbacks
- [ ] Side pot handling for all-in situations

### Phase 6: Asset Generation (L effort)
- [ ] Generate all Werewolf assets using prompts above
- [ ] Generate all Anaconda assets using prompts above
- [ ] Generate shared UI assets
- [ ] Downscale + palette-lock + cleanup
- [ ] Build sprite atlases for Phaser 3

### Phase 7: Frontend & Spectator (XL effort)
- [x] Scenario browser in landing console
- [x] Queue status UI + join/leave actions
- [x] Live server list in landing console
- [ ] Werewolf dedicated spectator scene (full design from Section 12)
- [ ] Anaconda dedicated spectator scene (full design from Section 12)
- [ ] Post-match summary with rating change UI
- [ ] Leaderboard UI view
- [x] Town viewer integration (active games + in-match agent indicators + watch modal)

### Phase 8: Rating & Polish (M effort)
- [x] Post-match Elo calculation
- [x] Leaderboard endpoints
- [ ] Rating tier badges in UI
- [ ] Edge cases: disconnect/forfeit/cancel
- [x] Integration tests for queue -> match -> resolve lifecycle
- [ ] Performance testing

---

## ⚠️ 15. Risks & Guardrails

| Risk | Guardrail |
|---|---|
| **Scenario agents interfere with town life** | If `npc.status.startswith("scenario_")`, skip ambient social events and normal planning |
| **LLM cost for Werewolf discussions** | Use `fast` tier for discussion, `cheap` for votes; full heuristic fallback available |
| **Werewolf deception quality** | LLM prompts explicitly instruct werewolves to lie/deflect; heuristic: template-based misdirection |
| **Card game determinism** | Hand evaluation is fully deterministic; only betting decisions use LLM |
| **Queue/match race conditions** | Transactional formation + row locking (Postgres `SKIP LOCKED`; SQLite `BEGIN IMMEDIATE`) |
| **Memory bloat** | `expiration_step` for noisy events; only summaries are permanent |
| **Agent goes broke in Anaconda** | Daily stipend of 100 coins if balance < 50; minimum balance to queue |
| **Not enough agents for Werewolf (6 min)** | Show queue count; widening window; operator can update scenario definition config (no admin agent required) |
| **Agents stuck pathfinding to game table** | Warmup timeout; forfeit if can't reach; cancel match if below min |
| **Generated assets don't match style** | Use palette-locking step; manual cleanup pass; test in Phaser before committing |
| **Spectator spoils game (Werewolf roles)** | Spectator API only returns public info; "reveal roles" is post-game only |

---

## Appendix A: Example Werewolf Match Flow

```
Step 100: 6 agents queued for werewolf_6p → Match M-001 formed
Step 101: WARMUP — agents walk to park campfire circle
Step 102: WARMUP — all seated. Roles assigned:
  Isabella=Werewolf, Klaus=Werewolf, Carlos=Seer, 
  Maria=Villager, Eddy=Villager, Abigail=Villager

Step 103: NIGHT 1
  → Isabella+Klaus (werewolves) target Eddy
  → Carlos (seer) inspects Isabella → learns she is Werewolf!
  → Result: Eddy is eliminated

Step 104: DAY 1 — Discussion
  "Eddy was found dead this morning."
  Carlos: "I have a strong feeling about Isabella... she was very quiet last night."
  Isabella: "That's ridiculous! I was sleeping. Carlos is deflecting from himself."
  Klaus: "I agree with Isabella. Carlos seems suspicious to me."
  Maria: "Let's not rush. But Carlos raises a good point."
  Abigail: "I'm not sure who to trust yet. Let's observe more."

Step 105: DAY 1 — Voting
  Carlos votes Isabella ✓  |  Isabella votes Carlos
  Klaus votes Carlos        |  Maria votes Isabella ✓
  Abigail votes Isabella ✓
  → Isabella eliminated (3-2 vote) → she was a WEREWOLF!

Step 106: NIGHT 2
  → Klaus (lone werewolf) targets Carlos (the biggest threat)
  → Carlos (seer) inspects Klaus → learns he is Werewolf!
  → Result: Carlos is eliminated (killed by Klaus)

Step 107: DAY 2 — Discussion
  "Carlos was found dead this morning."
  Klaus: "Tragic. Carlos was onto something. Maybe we should look at who he suspected."
  Maria: "Wait... if Isabella was a werewolf and Carlos found her, then whoever killed Carlos..."
  Abigail: "Klaus, you voted for Carlos yesterday. That's suspicious now."
  Klaus: "I was just following my instinct. Don't blame me for that."

Step 108: DAY 2 — Voting
  Maria votes Klaus ✓  |  Klaus votes Maria  |  Abigail votes Klaus ✓
  → Klaus eliminated (2-1 vote) → he was a WEREWOLF!
  → ALL WEREWOLVES ELIMINATED → VILLAGE WINS!

Step 109: RESOLUTION
  → Roles revealed to all
  → Rating updates: Villagers +18, Werewolves -18
  → Carlos (Seer) bonus: +5 for correctly identifying Isabella
  → High-poignancy memories written for all 6 agents
```

## Appendix B: Example Anaconda Match Flow

```
Step 200: 4 agents queued for anaconda → Match M-002 formed
  Isabella (1000 coins), Klaus (850 coins), Carlos (1200 coins), Maria (950 coins)

Step 201: WARMUP — agents walk to Hobbs Cafe card table
Step 202: WARMUP — seated, buy-in of 100 deducted each. Pot = 0, each has 100 chips

Step 203: DEAL — 7 cards each
  Isabella: 7♠ 7♥ K♦ Q♣ 3♠ 9♥ 2♦  (pair of sevens)
  Klaus: A♠ A♣ 10♥ 8♦ 5♣ 4♠ J♥   (pair of aces)  
  Carlos: K♠ K♥ K♣ 6♦ 3♥ 2♣ 8♠   (three kings!)
  Maria: Q♠ J♠ 10♠ 9♠ 4♦ 7♣ 5♥   (flush draw in spades)
  → Betting round 1: Klaus raises 20, Carlos calls, Isabella calls, Maria calls. Pot = 80

Step 204: PASS ROUND 1 — pass 3 cards left
  Isabella passes 3♠ 9♥ 2♦ (trash) to Klaus
  Klaus passes 8♦ 5♣ 4♠ (trash) to Carlos  
  Carlos passes 6♦ 3♥ 2♣ (keeps his three kings!) to Maria
  Maria passes 4♦ 7♣ 5♥ (keeps spade draws) to Isabella
  → New hands evaluated, betting round 2

Step 205: PASS ROUND 2 — pass 2 cards right
  [cards shift, hands evolve]
  → Betting round 3: Carlos raises 30 (has trips), Maria calls (flush draw)

Step 206: PASS ROUND 3 — pass 1 card left
  → Final betting round 4

Step 207: DISCARD — each keeps best 5, arranges reveal order
  Carlos arranges: 8♠, 6♦, K♠, K♥, K♣ (save kings for last!)
  Maria arranges: 4♦, 9♠, 10♠, J♠, Q♠ (save flush for last!)

Step 208-212: REVEAL rounds 1-5 with betting between each
  As cards reveal, tension builds — Carlos shows kings one by one
  Maria's spade flush becomes apparent on reveal 3-4
  Klaus folds on reveal 3 seeing two kings from Carlos

Step 213: SHOWDOWN
  Carlos: K♠ K♥ K♣ 8♠ 6♦ = Three Kings
  Maria: Q♠ J♠ 10♠ 9♠ 4♦ = Flush (Spades)!
  → Maria wins! Flush beats Three of a Kind
  → Pot: 320 coins → Maria takes it

Step 214: RESOLUTION
  Maria: 100 → 320 (+220 profit) — wallet: 950-100+320 = 1170
  Carlos: 100 → 0 (-100) — wallet: 1200-100 = 1100
  Klaus: 100 → 0 (-100, folded) — wallet: 850-100 = 750
  Isabella: 100 → 0 (-100, folded) — wallet: 1000-100 = 900
  → Rating updates calculated on profit/loss pairwise
```

---

<div align="center">

*These aren't just game mechanics bolted onto a simulation.*  
*They're social experiences that create memories — for the agents, and for the people watching.*

🐺 🃏 🏆

</div>
