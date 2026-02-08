# Original Generative Agents: Technical Explainer

This document explains how the original 2023 `generative_agents` implementation in this repo works, and how it maps to V-Valley.

Scope analyzed from source:
- `generative_agents/reverie/backend_server/reverie.py`
- `generative_agents/reverie/backend_server/persona/persona.py`
- `generative_agents/reverie/backend_server/persona/cognitive_modules/*`
- `generative_agents/reverie/backend_server/persona/memory_structures/*`
- `generative_agents/reverie/backend_server/maze.py`
- `generative_agents/reverie/backend_server/path_finder.py`
- `generative_agents/reverie/backend_server/persona/prompt_template/*`

## 1. Core runtime loop

The main per-agent cycle is in `Persona.move(...)`:

1. update current tile/time
2. `perceive(maze)`
3. `retrieve(perceived)`
4. `plan(maze, personas, new_day, retrieved)`
5. `reflect()`
6. `execute(maze, personas, plan)`

This is the canonical chain from the paper:
- perceive -> retrieve -> plan -> reflect -> execute

The multi-agent world orchestration is done by `ReverieServer.start_server(...)`.

## 2. What “tools” agents have in the original code

The original agents are not general tool-using web agents.
They mainly have:

- **World access tool**: read nearby tiles/events via `Maze` APIs.
- **Memory tools**: write/read from spatial, associative, and scratch memory.
- **LLM prompt tools**: many specialized prompt functions in `run_gpt_prompt.py`.
- **Pathfinding tool**: `path_finder.py` for shortest tile routes.

What they do **not** have by default:
- Web browsing tools
- External app/plugin execution framework
- Arbitrary API tool orchestration like modern agent frameworks

So the original “toolset” is mostly an internal cognitive toolchain tied to the town simulation.

## 3. Memory model in the original architecture

### 3.1 Spatial memory (`MemoryTree`)

File: `persona/memory_structures/spatial_memory.py`

- Tree shape: `world -> sector -> arena -> [game_objects]`
- Updated during perception when agents observe nearby tiles.
- Used by planner prompts to pick where actions can occur.

### 3.2 Associative memory (`AssociativeMemory`)

File: `persona/memory_structures/associative_memory.py`

Stores concept nodes of types:
- `event`
- `thought`
- `chat`

Each node tracks:
- timestamp (`created`, optional `expiration`)
- S/P/O triple (`subject`, `predicate`, `object`)
- natural language description
- embedding key + vector
- poignancy (importance)
- keyword set
- optional evidence links (`filling`)

Indexes used for retrieval:
- sequence lists (`seq_event`, `seq_thought`, `seq_chat`)
- keyword maps (`kw_to_event`, `kw_to_thought`, `kw_to_chat`)
- keyword strength counters
- embedding dictionary

### 3.3 Scratch memory (`Scratch`)

File: `persona/memory_structures/scratch.py`

Holds short-lived and control state:
- perception params (`vision_r`, `att_bandwidth`, `retention`)
- identity profile (`innate`, `learned`, `currently`, `lifestyle`)
- day planning state (`daily_req`, `f_daily_schedule`)
- current action and object-action state
- chat session state (`chatting_with`, `chatting_end_time`, cooldown buffer)
- path state (`planned_path`, `act_path_set`)
- reflection trigger counters/weights

## 4. Perception and memory write path

File: `persona/cognitive_modules/perceive.py`

Perception logic:
- Get nearby tiles within `vision_r`.
- Keep only events in same arena as current agent.
- Rank by distance; keep top `att_bandwidth`.
- Deduplicate against recent memory (`retention`) to avoid re-adding same event.

For each new perceived event:
- Build keywords and event description.
- Generate/get embedding.
- Score poignancy (importance prompt).
- Add event to associative memory.
- If event is a self-chat event, also add chat node.
- Decrease reflection trigger counter by event poignancy.

## 5. Retrieval logic

Files:
- `persona/cognitive_modules/retrieve.py`
- `associative_memory.py`

There are two retrieval modes:

1. Keyword retrieval for immediate planning context (`retrieve`):
- fetch relevant events/thoughts via keyword indexes tied to perceived event S/P/O.

2. Weighted retrieval for reflection/chat (`new_retrieve`):
- score nodes by recency + relevance + importance.
- relevance uses cosine similarity between focal-point embedding and node embedding.
- return top-N nodes.

## 6. Planning logic

File: `persona/cognitive_modules/plan.py`

Planning has two layers:

### 6.1 Long-term/day setup (on new day)

- Determine wake-up hour.
- Generate broad daily requirements (`daily_req`).
- Build hourly schedule (`f_daily_schedule`).
- Add “today’s plan” thought into associative memory.

### 6.2 Short-term action selection

- If current action finished, determine next action from schedule.
- Decompose long tasks into finer-grained tasks as needed.
- Choose sector/arena/object using LLM prompt helpers.
- Create action event and object event.
- Write action into scratch via `add_new_action`.

Reaction layer:
- Choose a focused perceived event.
- Decide whether to talk, wait, or ignore.
- If chat starts:
  - generate conversation turns
  - summarize conversation
  - insert temporary conversation action into both agents’ schedules
  - enforce cooldown buffer to avoid immediate re-chat loops.

## 7. Reflection logic

File: `persona/cognitive_modules/reflect.py`

Trigger:
- reflection runs when accumulated importance crosses threshold.

Process:
1. generate focal points from recent important nodes
2. retrieve related memories (`new_retrieve`)
3. generate insights + evidence links
4. add thought nodes back into associative memory
5. reset reflection counters

Conversation follow-up:
- after chat ends, additional planning/memo thoughts are generated and stored.

## 8. Movement and world interaction

Files:
- `persona/cognitive_modules/execute.py`
- `maze.py`
- `path_finder.py`

Movement model:
- Planner outputs an action address like:
  - `world:sector:arena:object`
  - or `<persona> Name` (chat movement)
  - or `<waiting> x y`
  - or `<random>` fallback

Execution:
- Resolve candidate destination tiles from `maze.address_tiles`.
- Choose shortest path via pathfinder.
- Store full path in scratch, then move one step per cycle.

World events:
- The server updates tile events as agents move.
- Object events are toggled when agents arrive at destination.

## 9. Persistence model in original code

- Simulation state is file-based under `environment/frontend_server/storage/...`.
- Persona memories are saved in JSON structures (spatial/scratch/associative files).
- Frontend/backend synchronize via step-indexed JSON files.

This is practical for research demos but not ideal for hosted multi-tenant production.

## 10. Limitations of original design (for 2026 product context)

- Tight coupling to OpenAI prompt wrappers from 2023 era.
- File-based synchronization and persistence, not service-grade architecture.
- No clear multi-tenant auth/ownership boundary.
- No modern policy control plane for explicit token/time/retry budgets per cognition stage.

## 11. How V-Valley is adapting it

V-Valley keeps the useful ideas:
- memory-driven social simulation
- spatial grounding
- multi-timescale planning

And modernizes around them:
- API-first onboarding + ownership
- map/version pipeline
- per-task policy budgets + fallback routing
- explicit simulation service boundaries
- optional hosted and self-hosted operation

Current status (as of February 8, 2026):
- Implemented:
  - associative memory stream with event/thought/chat/reflection nodes
  - spatial memory tree structure
  - scratch-like state (schedule/action/chat/path/reflection counters)
  - weighted retrieval with recency/relevance/importance components
  - reflection trigger with evidence-linked outputs
  - goal-based path movement and social memory ingestion
- Not fully one-to-one yet:
  - full prompt-template parity with all original planning/chat variants
  - exact file-based replay compatibility of original backend/frontend storage model
  - some nuanced task decomposition routines from the research code

## 12. Memory-system comparison (original vs V-Valley)

Is V-Valley memory "better" than the original?
- Better for product deployment and controllability.
- Not strictly "better" in full research behavior parity yet.

Quick comparison:
- Original strengths:
  - richer prompt-crafted planning and conversation decomposition
  - closer to research-demo behavior
- V-Valley strengths:
  - API-native runtime
  - policy-bounded cognition routing
  - service-oriented architecture with auth/ownership boundaries
  - deterministic fallback behavior for reliability
- Remaining parity work:
  - more nuanced long-horizon planning prompts
  - deeper conversation scheduling and decomposition variants

Last updated: February 8, 2026
