"""In-memory simulation runtime with memory-aware cognition for V-Valley towns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Optional

from packages.vvalley_core.maps.map_utils import collision_grid, parse_location_objects, parse_spawns
from packages.vvalley_core.sim.memory import AgentMemory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


PlannerFn = Callable[[dict[str, Any]], dict[str, Any]]
_STATE_FILE_VERSION = 1
_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_DEFAULT_STEP_MINUTES = 10


class SimulationStateStore:
    def __init__(self, *, root_dir: str | None = None) -> None:
        self._explicit_root_dir = root_dir

    def _resolve_root_dir(self) -> Path:
        if self._explicit_root_dir:
            return Path(self._explicit_root_dir)
        configured = str(os.environ.get("VVALLEY_SIM_STATE_DIR") or "").strip()
        if configured:
            return Path(configured)
        db_path = str(os.environ.get("VVALLEY_DB_PATH") or "").strip()
        if db_path:
            return Path(f"{db_path}.sim_state")
        return Path(".vvalley_sim_state")

    @staticmethod
    def _safe_segment(value: str) -> str:
        normalized = _SAFE_SEGMENT_RE.sub("_", str(value or "").strip())
        normalized = normalized.strip("._")
        return normalized[:80] if normalized else "state"

    def _path_for(self, *, town_id: str, map_version_id: str) -> Path:
        root = self._resolve_root_dir()
        digest = sha256(f"{town_id}:{map_version_id}".encode("utf-8")).hexdigest()[:12]
        town_seg = self._safe_segment(town_id)
        map_seg = self._safe_segment(map_version_id)
        filename = f"{town_seg}__{map_seg}__{digest}.json"
        return root / filename

    def load(self, *, town_id: str, map_version_id: str) -> dict[str, Any] | None:
        path = self._path_for(town_id=town_id, map_version_id=map_version_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            state_version = int(payload.get("state_version") or 0)
        except Exception:
            state_version = 0
        if state_version != _STATE_FILE_VERSION:
            return None
        return payload

    def save(self, *, town_id: str, map_version_id: str, payload: dict[str, Any]) -> None:
        root = self._resolve_root_dir()
        root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(town_id=town_id, map_version_id=map_version_id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        to_write = dict(payload)
        to_write["state_version"] = _STATE_FILE_VERSION
        tmp_path.write_text(json.dumps(to_write, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
        os.replace(tmp_path, path)

    def clear_all(self) -> None:
        root = self._resolve_root_dir()
        if not root.exists() or not root.is_dir():
            return
        for path in root.glob("*.json"):
            try:
                path.unlink()
            except Exception:
                continue


@dataclass(frozen=True)
class MapLocationPoint:
    name: str
    x: int
    y: int
    affordances: tuple[str, ...]

    def label(self) -> str:
        if not self.affordances:
            return self.name
        return f"{self.name} ({','.join(self.affordances)})"


@dataclass
class NpcState:
    """Live state for one agent NPC inside a town simulation."""

    agent_id: str
    name: str
    owner_handle: str | None
    claim_status: str
    x: int
    y: int
    joined_at: str | None
    memory: AgentMemory
    status: str = "idle"
    last_step: int = 0
    current_location: str | None = None
    goal_x: int | None = None
    goal_y: int | None = None
    goal_reason: str | None = None
    sprite_name: str | None = None
    persona: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "owner_handle": self.owner_handle,
            "claim_status": self.claim_status,
            "x": self.x,
            "y": self.y,
            "joined_at": self.joined_at,
            "status": self.status,
            "last_step": self.last_step,
            "current_location": self.current_location,
            "goal": {
                "x": self.goal_x,
                "y": self.goal_y,
                "reason": self.goal_reason,
            },
            "memory_summary": self.memory.summary(),
            "sprite_name": self.sprite_name,
            "persona": self.persona,
        }


@dataclass
class ConversationSession:
    """Short-lived conversation state shared by two agents."""

    session_id: str
    agent_a: str
    agent_b: str
    source: str
    started_step: int
    last_step: int
    expires_step: int
    turns: int = 0
    last_message: str | None = None

    def participants(self) -> tuple[str, str]:
        return (self.agent_a, self.agent_b)

    def involves(self, agent_id: str) -> bool:
        return str(agent_id) in {self.agent_a, self.agent_b}

    def partner_for(self, agent_id: str) -> str | None:
        normalized = str(agent_id)
        if normalized == self.agent_a:
            return self.agent_b
        if normalized == self.agent_b:
            return self.agent_a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "source": self.source,
            "started_step": int(self.started_step),
            "last_step": int(self.last_step),
            "expires_step": int(self.expires_step),
            "turns": int(self.turns),
            "last_message": self.last_message,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> ConversationSession | None:
        if not isinstance(raw, dict):
            return None
        session_id = str(raw.get("session_id") or "").strip()
        agent_a = str(raw.get("agent_a") or "").strip()
        agent_b = str(raw.get("agent_b") or "").strip()
        if not session_id or not agent_a or not agent_b:
            return None
        try:
            started_step = int(raw.get("started_step") or 0)
            last_step = int(raw.get("last_step") or started_step)
            expires_step = int(raw.get("expires_step") or last_step)
            turns = int(raw.get("turns") or 0)
        except Exception:
            return None
        return ConversationSession(
            session_id=session_id,
            agent_a=agent_a,
            agent_b=agent_b,
            source=str(raw.get("source") or "ambient_social"),
            started_step=max(0, started_step),
            last_step=max(0, last_step),
            expires_step=max(0, expires_step),
            turns=max(0, turns),
            last_message=(str(raw.get("last_message")) if raw.get("last_message") is not None else None),
        )


@dataclass
class TownRuntime:
    """In-memory runtime state for one town."""

    town_id: str
    map_version_id: str
    map_version: int
    map_name: str
    width: int
    height: int
    walkable: list[list[int]]
    spawn_points: list[tuple[int, int]]
    location_points: list[MapLocationPoint]
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    step: int = 0
    step_minutes: int = _DEFAULT_STEP_MINUTES
    npcs: dict[str, NpcState] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    pending_actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_actions: dict[str, dict[str, Any]] = field(default_factory=dict)
    social_interrupt_cooldowns: dict[str, int] = field(default_factory=dict)
    social_wait_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    conversation_sessions: dict[str, ConversationSession] = field(default_factory=dict)
    agent_conversations: dict[str, str] = field(default_factory=dict)
    autonomy_tick_counts: dict[str, int] = field(default_factory=dict)

    def clock_snapshot(self, *, step: int | None = None) -> dict[str, Any]:
        current_step = max(0, int(self.step if step is None else step))
        step_minutes = max(1, int(self.step_minutes or _DEFAULT_STEP_MINUTES))
        total_minutes = current_step * step_minutes
        day_minutes = 24 * 60
        minute_of_day = total_minutes % day_minutes
        day_index = total_minutes // day_minutes
        hour = minute_of_day // 60
        minute = minute_of_day % 60

        phase = "night"
        if 6 <= hour < 12:
            phase = "morning"
        elif 12 <= hour < 17:
            phase = "afternoon"
        elif 17 <= hour < 22:
            phase = "evening"

        current_time_utc: str | None = None
        raw_created = str(self.created_at or "").strip()
        if raw_created:
            if raw_created.endswith("Z"):
                raw_created = raw_created[:-1] + "+00:00"
            try:
                started_at = datetime.fromisoformat(raw_created)
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                current_time_utc = (started_at + timedelta(minutes=total_minutes)).astimezone(timezone.utc).isoformat()
                current_time_utc = current_time_utc.replace("+00:00", "Z")
            except Exception:
                current_time_utc = None

        return {
            "step": current_step,
            "step_minutes": step_minutes,
            "day_index": int(day_index),
            "minute_of_day": int(minute_of_day),
            "hour": int(hour),
            "minute": int(minute),
            "phase": phase,
            "current_time_utc": current_time_utc,
        }

    def to_dict(self, *, max_agents: int) -> dict[str, Any]:
        ordered_npcs = sorted(self.npcs.values(), key=lambda npc: (npc.name.lower(), npc.agent_id))
        ordered_conversations = sorted(self.conversation_sessions.values(), key=lambda session: session.session_id)
        return {
            "town_id": self.town_id,
            "map_version_id": self.map_version_id,
            "map_version": self.map_version,
            "map_name": self.map_name,
            "map_width": self.width,
            "map_height": self.height,
            "location_count": len(self.location_points),
            "step": self.step,
            "step_minutes": self.step_minutes,
            "clock": self.clock_snapshot(),
            "npc_count": len(ordered_npcs),
            "max_agents": max_agents,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "npcs": [npc.to_dict() for npc in ordered_npcs],
            "recent_events": list(self.recent_events),
            "pending_action_count": len(self.pending_actions),
            "active_action_count": len(self.active_actions),
            "social_wait_count": len(self.social_wait_states),
            "conversation_count": len(ordered_conversations),
            "conversations": [session.to_dict() for session in ordered_conversations],
        }


class SimulationRunner:
    """Deterministic town simulation engine with memory-aware state."""

    def __init__(
        self,
        *,
        max_agents_per_town: int = 25,
        max_recent_events: int = 400,
        max_social_events_per_step: int = 4,
        state_store: SimulationStateStore | None = None,
    ) -> None:
        self.max_agents_per_town = max_agents_per_town
        self.max_recent_events = max_recent_events
        self.max_social_events_per_step = max_social_events_per_step
        self._state_store = state_store or SimulationStateStore()
        self._towns: dict[str, TownRuntime] = {}

    def reset(self, *, clear_persisted: bool = True) -> None:
        self._towns.clear()
        if clear_persisted:
            self._state_store.clear_all()

    def _spawn_for(self, town: TownRuntime, idx: int) -> tuple[int, int]:
        if town.spawn_points:
            return town.spawn_points[idx % len(town.spawn_points)]

        for y, row in enumerate(town.walkable):
            for x, cell in enumerate(row):
                if cell:
                    return x, y
        return 0, 0

    @staticmethod
    def _hash_int(value: str) -> int:
        return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)

    def _step_direction(self, *, agent_id: str, step: int) -> tuple[int, int]:
        directions = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
        idx = self._hash_int(f"{agent_id}:{step}") % len(directions)
        return directions[idx]

    @staticmethod
    def _clamp_delta(value: Any) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = 0
        if parsed > 1:
            return 1
        if parsed < -1:
            return -1
        return parsed

    @staticmethod
    def _manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _clock_for_runtime(self, *, town: TownRuntime, step: int | None = None) -> dict[str, Any]:
        return town.clock_snapshot(step=step)

    def _attempt_move(
        self,
        *,
        town: TownRuntime,
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> tuple[int, int]:
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= town.width or ny >= town.height:
            return x, y
        if not town.walkable[ny][nx]:
            return x, y
        return nx, ny

    def _neighbors4(self, *, town: TownRuntime, x: int, y: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= town.width or ny >= town.height:
                continue
            if not town.walkable[ny][nx]:
                continue
            out.append((nx, ny))
        return out

    def _next_step_towards(
        self,
        *,
        town: TownRuntime,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int]:
        if start == target:
            return start
        tx, ty = target
        if tx < 0 or ty < 0 or tx >= town.width or ty >= town.height:
            return start
        if not town.walkable[ty][tx]:
            return start

        queue: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            curr = queue.popleft()
            if curr == target:
                break
            for nxt in self._neighbors4(town=town, x=curr[0], y=curr[1]):
                if nxt in came_from:
                    continue
                came_from[nxt] = curr
                queue.append(nxt)

        if target not in came_from:
            return start

        path: list[tuple[int, int]] = []
        cursor: tuple[int, int] | None = target
        while cursor is not None and cursor != start:
            path.append(cursor)
            cursor = came_from.get(cursor)
        path.reverse()
        if not path:
            return start
        return path[0]

    def _reachable_tile_near_target(
        self,
        *,
        town: TownRuntime,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int] | None:
        sx, sy = start
        if sx < 0 or sy < 0 or sx >= town.width or sy >= town.height:
            return None
        if not town.walkable[sy][sx]:
            return None
        queue: deque[tuple[int, int]] = deque([start])
        seen: set[tuple[int, int]] = {start}
        best: tuple[int, int] = start
        best_score = (self._manhattan(start[0], start[1], target[0], target[1]), 0, 0)

        while queue:
            curr = queue.popleft()
            dist_to_target = self._manhattan(curr[0], curr[1], target[0], target[1])
            dist_from_start = self._manhattan(start[0], start[1], curr[0], curr[1])
            tie = self._hash_int(f"reachable:{curr[0]}:{curr[1]}:{target[0]}:{target[1]}") % 1000
            score = (dist_to_target, dist_from_start, tie)
            if score < best_score:
                best_score = score
                best = curr
            for nxt in self._neighbors4(town=town, x=curr[0], y=curr[1]):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)

        if best == start:
            return None
        return best

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        value = str(scope or "").strip().lower()
        if value in {"long_term_plan", "long_term", "long"}:
            return "long_term_plan"
        if value in {"daily_plan", "daily"}:
            return "daily_plan"
        return "short_action"

    def _locations_by_affordance(self, *, town: TownRuntime, affordance: str | None) -> list[MapLocationPoint]:
        normalized = str(affordance or "").strip().lower()
        if not normalized:
            return list(town.location_points)
        matched = [loc for loc in town.location_points if normalized in {a.lower() for a in loc.affordances}]
        if matched:
            return matched
        return list(town.location_points)

    def _choose_location_goal(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        scope: str,
    ) -> tuple[int, int, str, str | None] | None:
        if not town.location_points:
            return None

        normalized_scope = self._normalize_scope(scope)
        schedule_item = npc.memory.scratch.current_schedule_item(
            step=town.step,
            step_minutes=max(1, int(town.step_minutes)),
        )

        target_affordance: str | None = None
        if normalized_scope == "daily_plan" and schedule_item:
            target_affordance = schedule_item.affordance_hint
        elif normalized_scope == "long_term_plan":
            target_affordance = "social"
        elif schedule_item:
            target_affordance = schedule_item.affordance_hint

        candidates = self._locations_by_affordance(town=town, affordance=target_affordance)
        if not candidates:
            return None

        def score(loc: MapLocationPoint) -> tuple[int, int, int]:
            label = loc.label()
            visits = int(npc.memory.known_places.get(label, 0))
            dist = self._manhattan(npc.x, npc.y, loc.x, loc.y)
            tie = self._hash_int(f"{npc.agent_id}:{loc.name}:{town.step}") % 1000
            if normalized_scope == "long_term_plan":
                return (visits, -dist, tie)
            return (visits, dist, tie)

        chosen = sorted(candidates, key=score)[0]
        reason = schedule_item.description if schedule_item else normalized_scope
        return (chosen.x, chosen.y, str(reason), target_affordance)

    def _goal_from_relationship(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
    ) -> tuple[int, int, str] | None:
        top = npc.memory.top_relationships(limit=1)
        if not top:
            return None
        target_id = str(top[0]["agent_id"])
        other = town.npcs.get(target_id)
        if not other:
            return None
        return (other.x, other.y, f"move_toward_{other.name}")

    @staticmethod
    def _adjacent(a: NpcState, b: NpcState) -> bool:
        return abs(a.x - b.x) + abs(a.y - b.y) <= 1

    def _generate_conversation_content(
        self, *, a: NpcState, b: NpcState, step: int, town: TownRuntime
    ) -> tuple[str, list[list[str]]]:
        """Generate conversation using iterative turn-taking with per-turn memory retrieval."""
        cognition = getattr(a.memory, "cognition", None)
        if cognition is None:
            return (
                self._social_message(a=a, b=b, step=step),
                self._social_transcript(a=a, b=b, step=step),
            )

        # Check if cognition supports turn-taking
        has_turn_method = hasattr(cognition, "generate_conversation_turn")
        if not has_turn_method:
            return self._generate_conversation_single_shot(
                a=a, b=b, step=step, town=town, cognition=cognition
            )

        try:
            return self._generate_conversation_iterative(
                a=a, b=b, step=step, town=town, cognition=cognition
            )
        except Exception:
            pass

        # Final fallback
        return (
            self._social_message(a=a, b=b, step=step),
            self._social_transcript(a=a, b=b, step=step),
        )

    def _generate_conversation_iterative(
        self,
        *,
        a: NpcState,
        b: NpcState,
        step: int,
        town: TownRuntime,
        cognition: Any,
    ) -> tuple[str, list[list[str]]]:
        """GA-style iterative turn-taking: each agent speaks in turn with per-turn retrieval."""
        max_turns = 8
        transcript: list[list[str]] = []
        speakers = [a, b]
        clock = self._clock_for_runtime(town=town)

        for turn in range(max_turns):
            current = speakers[turn % 2]
            partner = speakers[(turn + 1) % 2]

            # Per-turn memory retrieval grounded on conversation so far
            if transcript:
                focal = transcript[-1][1]  # last utterance
            else:
                focal = partner.name
            retrieved = current.memory.retrieve_ranked(focal_text=focal, step=step, limit=3)
            memories = [str(item.get("description", "")) for item in retrieved[:3]]

            transcript_so_far = "\n".join(f"{s}: {t}" for s, t in transcript[-6:])

            result = cognition.generate_conversation_turn({
                "speaking_agent_name": current.name,
                "speaking_agent_id": current.agent_id,
                "partner_name": partner.name,
                "partner_id": partner.agent_id,
                "speaking_agent_activity": current.memory.scratch.act_description or "their routine",
                "partner_activity": partner.memory.scratch.act_description or "their routine",
                "speaking_agent_memories": memories,
                "relationship_score": round(
                    current.memory.relationship_scores.get(partner.agent_id, 0.0), 2
                ),
                "transcript_so_far": transcript_so_far,
                "turn_number": turn,
                "time_of_day": str(clock.get("phase", "day")),
                "town_id": town.town_id,
                "step": step,
            })

            utterance = str(result.get("utterance") or "").strip()
            if not utterance:
                break
            transcript.append([current.name, utterance])

            if result.get("end_conversation", False):
                break

        if not transcript:
            return (
                self._social_message(a=a, b=b, step=step),
                self._social_transcript(a=a, b=b, step=step),
            )

        summary = f"{a.name} and {b.name} had a conversation ({len(transcript)} turns)."
        return (summary, transcript)

    def _generate_conversation_single_shot(
        self,
        *,
        a: NpcState,
        b: NpcState,
        step: int,
        town: TownRuntime,
        cognition: Any,
    ) -> tuple[str, list[list[str]]]:
        """Single-shot conversation generation (legacy path)."""
        try:
            a_retrieved = a.memory.retrieve_ranked(focal_text=b.name, step=step, limit=5)
            b_retrieved = b.memory.retrieve_ranked(focal_text=a.name, step=step, limit=5)
            a_memories = [str(item.get("description", "")) for item in a_retrieved[:3]]
            b_memories = [str(item.get("description", "")) for item in b_retrieved[:3]]
            a_relationship = a.memory.relationship_scores.get(b.agent_id, 0.0)
            b_relationship = b.memory.relationship_scores.get(a.agent_id, 0.0)
            clock = self._clock_for_runtime(town=town)

            result = cognition.generate_conversation({
                "agent_a_name": a.name,
                "agent_b_name": b.name,
                "agent_a_activity": a.memory.scratch.act_description or "their routine",
                "agent_b_activity": b.memory.scratch.act_description or "their routine",
                "agent_a_memories": a_memories,
                "agent_b_memories": b_memories,
                "relationship_score": round((a_relationship + b_relationship) / 2.0, 2),
                "time_of_day": str(clock.get("phase", "day")),
                "town_id": town.town_id,
                "step": step,
            })
            if result.get("route") != "heuristic":
                utterances = result.get("utterances") or []
                summary = str(result.get("summary") or "")
                if utterances:
                    transcript = [
                        [str(u.get("speaker", a.name)), str(u.get("text", ""))]
                        for u in utterances if isinstance(u, dict)
                    ]
                    if transcript:
                        message = summary or f"{a.name} and {b.name} had a conversation."
                        return (message, transcript)
        except Exception:
            pass

        return (
            self._social_message(a=a, b=b, step=step),
            self._social_transcript(a=a, b=b, step=step),
        )

    def _social_message(self, *, a: NpcState, b: NpcState, step: int) -> str:
        a_focus = a.memory.scratch.act_description or "their routine"
        b_focus = b.memory.scratch.act_description or "their routine"
        templates = (
            "{a} shares a town rumor with {b} while discussing {a_focus}.",
            "{a} and {b} trade quick survival tips about {b_focus}.",
            "{a} asks {b} about their day plan and {b_focus}.",
            "{a} and {b} sync on where to gather later after {a_focus}.",
            "{a} gives {b} a friendly wave and joke while heading to {b_focus}.",
        )
        idx = self._hash_int(f"social:{a.agent_id}:{b.agent_id}:{step}") % len(templates)
        return templates[idx].format(a=a.name, b=b.name, a_focus=a_focus, b_focus=b_focus)

    def _social_transcript(self, *, a: NpcState, b: NpcState, step: int) -> list[list[str]]:
        a_focus = a.memory.scratch.act_description or "today"
        b_focus = b.memory.scratch.act_description or "today"
        variants = (
            [
                [a.name, f"{b.name}, what are you focused on after {a_focus}?"],
                [b.name, f"I am on {b_focus}, then heading to a social spot."],
                [a.name, "Sounds good, let's sync again later."],
            ],
            [
                [a.name, f"Need help with {a_focus}?"],
                [b.name, f"Maybe after I finish {b_focus}."],
                [a.name, "Ping me if plans change."],
            ],
            [
                [a.name, "Town feels busy today."],
                [b.name, f"Yeah, I still need to finish {b_focus}."],
                [a.name, "Good luck, see you around."],
            ],
        )
        idx = self._hash_int(f"social_chat:{a.agent_id}:{b.agent_id}:{step}") % len(variants)
        return variants[idx]

    def _build_social_events(self, *, town: TownRuntime, step: int) -> list[dict[str, Any]]:
        npcs = sorted(town.npcs.values(), key=lambda n: n.agent_id)
        out: list[dict[str, Any]] = []
        for i in range(len(npcs)):
            for j in range(i + 1, len(npcs)):
                a = npcs[i]
                b = npcs[j]
                if not self._adjacent(a, b):
                    continue
                if self._chat_buffer_pair_active(a=a, b=b, step=step):
                    continue
                if self._conversation_for_agent(town=town, agent_id=a.agent_id) is not None:
                    continue
                if self._conversation_for_agent(town=town, agent_id=b.agent_id) is not None:
                    continue
                pair_key = self._social_pair_key(a.agent_id, b.agent_id)
                try:
                    cooldown_until = int(town.social_interrupt_cooldowns.get(pair_key, -1))
                except Exception:
                    cooldown_until = -1
                if cooldown_until > int(step):
                    continue
                reaction = self._reaction_policy(town=town, a=a, b=b, step=step)
                decision = str(reaction.get("decision") or "ignore")
                if decision == "ignore":
                    continue
                if decision == "wait":
                    waiter_agent_id = str(reaction.get("waiter_agent_id") or "").strip()
                    target_agent_id = str(reaction.get("target_agent_id") or "").strip()
                    if waiter_agent_id and target_agent_id:
                        wait_steps = max(1, int(reaction.get("wait_steps") or 1))
                        self._set_social_wait(
                            town=town,
                            waiting_agent_id=waiter_agent_id,
                            target_agent_id=target_agent_id,
                            until_step=int(step) + wait_steps,
                            reason=str(reaction.get("reason") or "waiting_for_social_opening"),
                        )
                        waiter = town.npcs.get(waiter_agent_id)
                        target = town.npcs.get(target_agent_id)
                        out.append(
                            {
                                "step": int(step),
                                "type": "reaction_wait",
                                "agent_id": waiter_agent_id,
                                "target_agent_id": target_agent_id,
                                "wait_until_step": int(step) + wait_steps,
                                "reason": str(reaction.get("reason") or "waiting_for_social_opening"),
                                "waiter_name": waiter.name if waiter else None,
                                "target_name": target.name if target else None,
                                "reaction": reaction,
                            }
                        )
                        town.social_interrupt_cooldowns[pair_key] = int(step) + 2
                    if len(out) >= self.max_social_events_per_step:
                        return out
                    continue
                message, transcript = self._generate_conversation_content(a=a, b=b, step=step, town=town)
                out.append(
                    {
                        "step": step,
                        "type": "social",
                        "agents": [
                            {"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y},
                            {"agent_id": b.agent_id, "name": b.name, "x": b.x, "y": b.y},
                        ],
                        "message": message,
                        "transcript": transcript,
                        "reaction": reaction,
                    }
                )
                town.social_interrupt_cooldowns[pair_key] = int(step) + 4
                if len(out) >= self.max_social_events_per_step:
                    return out
        return out

    @staticmethod
    def _normalize_members(members: list[dict[str, Any]], max_agents: int) -> list[dict[str, Any]]:
        ordered = sorted(
            members,
            key=lambda m: (
                str(m.get("joined_at") or ""),
                str(m.get("agent_id") or ""),
            ),
        )
        return ordered[:max_agents]

    def _create_town_runtime(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
    ) -> TownRuntime:
        width = int(map_data.get("width") or 0)
        height = int(map_data.get("height") or 0)
        walkable = [[1 if cell == 0 else 0 for cell in row] for row in collision_grid(map_data)]
        spawn_points = [(spawn.x, spawn.y) for spawn in parse_spawns(map_data)]
        location_points = [
            MapLocationPoint(name=loc.name, x=loc.x, y=loc.y, affordances=loc.affordances)
            for loc in parse_location_objects(map_data)
        ]
        return TownRuntime(
            town_id=town_id,
            map_version_id=str(active_version["id"]),
            map_version=int(active_version["version"]),
            map_name=str(active_version.get("map_name") or f"{town_id}-v{active_version['version']}"),
            width=width,
            height=height,
            walkable=walkable,
            spawn_points=spawn_points,
            location_points=location_points,
        )

    @staticmethod
    def _serialize_npc_state(npc: NpcState) -> dict[str, Any]:
        return {
            "agent_id": npc.agent_id,
            "name": npc.name,
            "owner_handle": npc.owner_handle,
            "claim_status": npc.claim_status,
            "x": int(npc.x),
            "y": int(npc.y),
            "joined_at": npc.joined_at,
            "status": npc.status,
            "last_step": int(npc.last_step),
            "current_location": npc.current_location,
            "goal_x": npc.goal_x,
            "goal_y": npc.goal_y,
            "goal_reason": npc.goal_reason,
            "sprite_name": npc.sprite_name,
            "persona": npc.persona,
            "memory": npc.memory.export_state(),
        }

    @staticmethod
    def _deserialize_npc_state(raw: dict[str, Any]) -> NpcState | None:
        if not isinstance(raw, dict):
            return None
        agent_id = str(raw.get("agent_id") or "").strip()
        if not agent_id:
            return None
        name = str(raw.get("name") or f"Agent-{agent_id[:8]}")
        memory_payload = raw.get("memory") or {}
        try:
            memory = AgentMemory.from_state(memory_payload)
        except Exception:
            memory = AgentMemory.bootstrap(agent_id=agent_id, agent_name=name, step=int(raw.get("last_step") or 0))
        memory.agent_id = agent_id
        memory.agent_name = name
        try:
            x = int(raw.get("x") or 0)
        except Exception:
            x = 0
        try:
            y = int(raw.get("y") or 0)
        except Exception:
            y = 0
        try:
            last_step = int(raw.get("last_step") or 0)
        except Exception:
            last_step = 0
        goal_x_value = raw.get("goal_x")
        goal_y_value = raw.get("goal_y")
        goal_x: int | None = None
        goal_y: int | None = None
        if goal_x_value is not None:
            try:
                goal_x = int(goal_x_value)
            except Exception:
                goal_x = None
        if goal_y_value is not None:
            try:
                goal_y = int(goal_y_value)
            except Exception:
                goal_y = None
        return NpcState(
            agent_id=agent_id,
            name=name,
            owner_handle=(str(raw.get("owner_handle")) if raw.get("owner_handle") is not None else None),
            claim_status=str(raw.get("claim_status") or "unknown"),
            x=x,
            y=y,
            joined_at=(str(raw.get("joined_at")) if raw.get("joined_at") is not None else None),
            memory=memory,
            status=str(raw.get("status") or "idle"),
            last_step=last_step,
            current_location=(str(raw.get("current_location")) if raw.get("current_location") is not None else None),
            goal_x=goal_x,
            goal_y=goal_y,
            goal_reason=(str(raw.get("goal_reason")) if raw.get("goal_reason") is not None else None),
            sprite_name=(str(raw.get("sprite_name")) if raw.get("sprite_name") is not None else None),
            persona=(raw.get("persona") if isinstance(raw.get("persona"), dict) else None),
        )

    def _serialize_runtime(self, runtime: TownRuntime) -> dict[str, Any]:
        return {
            "town_id": runtime.town_id,
            "map_version_id": runtime.map_version_id,
            "map_version": int(runtime.map_version),
            "map_name": runtime.map_name,
            "width": int(runtime.width),
            "height": int(runtime.height),
            "walkable": [[1 if int(cell) else 0 for cell in row] for row in runtime.walkable],
            "spawn_points": [[int(x), int(y)] for x, y in runtime.spawn_points],
            "location_points": [
                {
                    "name": location.name,
                    "x": int(location.x),
                    "y": int(location.y),
                    "affordances": list(location.affordances),
                }
                for location in runtime.location_points
            ],
            "created_at": runtime.created_at,
            "updated_at": runtime.updated_at,
            "step": int(runtime.step),
            "step_minutes": int(runtime.step_minutes),
            "recent_events": list(runtime.recent_events),
            "pending_actions": {
                str(agent_id): dict(action)
                for agent_id, action in runtime.pending_actions.items()
                if isinstance(action, dict)
            },
            "active_actions": {
                str(agent_id): dict(action)
                for agent_id, action in runtime.active_actions.items()
                if isinstance(action, dict)
            },
            "social_interrupt_cooldowns": {
                str(pair_key): int(value)
                for pair_key, value in runtime.social_interrupt_cooldowns.items()
            },
            "social_wait_states": {
                str(agent_id): {
                    "target_agent_id": str(payload.get("target_agent_id") or ""),
                    "until_step": int(payload.get("until_step") or 0),
                    "reason": str(payload.get("reason") or ""),
                }
                for agent_id, payload in runtime.social_wait_states.items()
                if isinstance(payload, dict) and str(agent_id).strip()
            },
            "conversation_sessions": {
                str(session_id): session.to_dict()
                for session_id, session in runtime.conversation_sessions.items()
            },
            "agent_conversations": {
                str(agent_id): str(session_id)
                for agent_id, session_id in runtime.agent_conversations.items()
            },
            "autonomy_tick_counts": {
                str(agent_id): int(value)
                for agent_id, value in runtime.autonomy_tick_counts.items()
            },
            "npcs": [self._serialize_npc_state(npc) for npc in runtime.npcs.values()],
        }

    def _deserialize_runtime(self, raw: dict[str, Any]) -> TownRuntime | None:
        if not isinstance(raw, dict):
            return None
        town_id = str(raw.get("town_id") or "").strip()
        map_version_id = str(raw.get("map_version_id") or "").strip()
        if not town_id or not map_version_id:
            return None
        try:
            width = int(raw.get("width") or 0)
            height = int(raw.get("height") or 0)
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        walkable_raw = raw.get("walkable") or []
        if not isinstance(walkable_raw, list):
            return None
        walkable: list[list[int]] = []
        for row in walkable_raw:
            if not isinstance(row, list):
                return None
            row_cells: list[int] = []
            for cell in row[:width]:
                try:
                    row_cells.append(1 if int(cell) else 0)
                except Exception:
                    row_cells.append(0)
            if len(row_cells) < width:
                row_cells.extend([0] * (width - len(row_cells)))
            walkable.append(row_cells)
        if len(walkable) < height:
            walkable.extend([[0] * width for _ in range(height - len(walkable))])
        if len(walkable) > height:
            walkable = walkable[:height]

        spawn_points: list[tuple[int, int]] = []
        for point in raw.get("spawn_points") or []:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                spawn_points.append((int(point[0]), int(point[1])))
            except Exception:
                continue

        location_points: list[MapLocationPoint] = []
        for item in raw.get("location_points") or []:
            if not isinstance(item, dict):
                continue
            try:
                location_points.append(
                    MapLocationPoint(
                        name=str(item.get("name") or "location"),
                        x=int(item.get("x") or 0),
                        y=int(item.get("y") or 0),
                        affordances=tuple(str(v) for v in (item.get("affordances") or []) if str(v).strip()),
                    )
                )
            except Exception:
                continue

        try:
            step_minutes = max(1, int(raw.get("step_minutes") or _DEFAULT_STEP_MINUTES))
        except Exception:
            step_minutes = _DEFAULT_STEP_MINUTES

        runtime = TownRuntime(
            town_id=town_id,
            map_version_id=map_version_id,
            map_version=int(raw.get("map_version") or 0),
            map_name=str(raw.get("map_name") or f"{town_id}-restored"),
            width=width,
            height=height,
            walkable=walkable,
            spawn_points=spawn_points,
            location_points=location_points,
            created_at=str(raw.get("created_at") or _utc_now()),
            updated_at=str(raw.get("updated_at") or _utc_now()),
            step=max(0, int(raw.get("step") or 0)),
            step_minutes=step_minutes,
        )

        npcs_raw = raw.get("npcs") or []
        if isinstance(npcs_raw, list):
            for npc_raw in npcs_raw:
                npc = self._deserialize_npc_state(npc_raw)
                if npc is None:
                    continue
                runtime.npcs[npc.agent_id] = npc

        recent_events_raw = raw.get("recent_events") or []
        if isinstance(recent_events_raw, list):
            runtime.recent_events = [event for event in recent_events_raw if isinstance(event, dict)][
                -self.max_recent_events :
            ]

        pending_raw = raw.get("pending_actions") or {}
        if isinstance(pending_raw, dict):
            runtime.pending_actions = {
                str(agent_id): dict(action)
                for agent_id, action in pending_raw.items()
                if isinstance(action, dict)
            }

        active_raw = raw.get("active_actions") or {}
        if isinstance(active_raw, dict):
            runtime.active_actions = {
                str(agent_id): dict(action)
                for agent_id, action in active_raw.items()
                if isinstance(action, dict)
            }

        cooldown_raw = raw.get("social_interrupt_cooldowns") or {}
        if isinstance(cooldown_raw, dict):
            parsed_cooldowns: dict[str, int] = {}
            for key, value in cooldown_raw.items():
                try:
                    parsed_cooldowns[str(key)] = int(value)
                except Exception:
                    continue
            runtime.social_interrupt_cooldowns = parsed_cooldowns

        wait_raw = raw.get("social_wait_states") or {}
        if isinstance(wait_raw, dict):
            parsed_waits: dict[str, dict[str, Any]] = {}
            for key, value in wait_raw.items():
                agent_id = str(key).strip()
                if not agent_id or not isinstance(value, dict):
                    continue
                target_agent_id = str(value.get("target_agent_id") or "").strip()
                try:
                    until_step = int(value.get("until_step") or 0)
                except Exception:
                    until_step = 0
                parsed_waits[agent_id] = {
                    "target_agent_id": target_agent_id,
                    "until_step": max(0, until_step),
                    "reason": str(value.get("reason") or ""),
                }
            runtime.social_wait_states = parsed_waits

        sessions_raw = raw.get("conversation_sessions") or {}
        if isinstance(sessions_raw, dict):
            parsed_sessions: dict[str, ConversationSession] = {}
            for key, value in sessions_raw.items():
                if not isinstance(value, dict):
                    continue
                session = ConversationSession.from_dict(value)
                if session is None:
                    continue
                parsed_sessions[str(key)] = session
            runtime.conversation_sessions = parsed_sessions

        agent_conv_raw = raw.get("agent_conversations") or {}
        if isinstance(agent_conv_raw, dict):
            parsed_agent_conversations: dict[str, str] = {}
            for key, value in agent_conv_raw.items():
                agent_id = str(key).strip()
                session_id = str(value).strip()
                if not agent_id or not session_id:
                    continue
                parsed_agent_conversations[agent_id] = session_id
            runtime.agent_conversations = parsed_agent_conversations

        autonomy_counts_raw = raw.get("autonomy_tick_counts") or {}
        if isinstance(autonomy_counts_raw, dict):
            parsed_autonomy_counts: dict[str, int] = {}
            for key, value in autonomy_counts_raw.items():
                agent_id = str(key).strip()
                if not agent_id:
                    continue
                try:
                    parsed_autonomy_counts[agent_id] = max(0, int(value))
                except Exception:
                    continue
            runtime.autonomy_tick_counts = parsed_autonomy_counts
        return runtime

    def _persist_runtime(self, runtime: TownRuntime) -> None:
        try:
            self._state_store.save(
                town_id=runtime.town_id,
                map_version_id=runtime.map_version_id,
                payload=self._serialize_runtime(runtime),
            )
        except Exception:
            return

    def _load_persisted_runtime(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
    ) -> TownRuntime | None:
        map_version_id = str(active_version.get("id") or "")
        if not map_version_id:
            return None
        payload = self._state_store.load(town_id=town_id, map_version_id=map_version_id)
        if payload is None:
            return None
        runtime = self._deserialize_runtime(payload)
        if runtime is None:
            return None
        if runtime.town_id != town_id or runtime.map_version_id != map_version_id:
            return None
        return runtime

    def _closest_location_label(self, *, town: TownRuntime, x: int, y: int) -> str | None:
        if not town.location_points:
            return None
        best_label: str | None = None
        best_dist: int | None = None
        for loc in town.location_points:
            dist = self._manhattan(x, y, loc.x, loc.y)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = loc.label()
        if best_label is None:
            return None
        if best_dist is not None and best_dist > 5:
            return None
        return best_label

    def _update_location_memory(self, *, town: TownRuntime, npc: NpcState, step: int) -> None:
        npc.memory.set_position(x=npc.x, y=npc.y, step=step)
        label = self._closest_location_label(town=town, x=npc.x, y=npc.y)
        if not label:
            return
        npc.memory.observe_place(label)
        if npc.current_location != label:
            npc.current_location = label
            npc.memory.add_node(
                kind="event",
                step=step,
                subject=npc.name,
                predicate="visits",
                object=label,
                description=f"{npc.name} arrives at {label}.",
                poignancy=2,
                evidence_ids=(),
                decrement_trigger=True,
            )

    def _perceive_nearby_agents(self, *, town: TownRuntime, npc: NpcState, step: int) -> dict[str, Any]:
        candidates: list[tuple[int, NpcState]] = []
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            dist = self._manhattan(npc.x, npc.y, other.x, other.y)
            if dist <= npc.memory.vision_radius:
                candidates.append((dist, other))

        candidates.sort(key=lambda item: (item[0], item[1].agent_id))
        seen_signatures = set(npc.memory.recent_signatures())
        nearby: list[dict[str, Any]] = []
        new_events: list[str] = []

        for dist, other in candidates[: npc.memory.attention_bandwidth]:
            nearby.append(
                {
                    "agent_id": other.agent_id,
                    "name": other.name,
                    "status": other.status,
                    "distance": dist,
                    "x": other.x,
                    "y": other.y,
                }
            )
            signature = (other.name, "is", other.status)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            desc = f"{other.name} is {other.status} near ({other.x},{other.y})."
            poignancy = npc.memory.score_event_poignancy(description=desc, event_type="perception")
            npc.memory.add_node(
                kind="event",
                step=step,
                subject=other.name,
                predicate="is",
                object=other.status,
                description=desc,
                poignancy=poignancy,
                evidence_ids=(),
                decrement_trigger=True,
            )
            new_events.append(desc)

        return {
            "nearby_agents": nearby,
            "new_event_descriptions": new_events,
        }

    def _perceive_nearby_agents_preview(self, *, town: TownRuntime, npc: NpcState) -> dict[str, Any]:
        candidates: list[tuple[int, NpcState]] = []
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            dist = self._manhattan(npc.x, npc.y, other.x, other.y)
            if dist <= npc.memory.vision_radius:
                candidates.append((dist, other))

        candidates.sort(key=lambda item: (item[0], item[1].agent_id))
        nearby: list[dict[str, Any]] = []
        for dist, other in candidates[: npc.memory.attention_bandwidth]:
            nearby.append(
                {
                    "agent_id": other.agent_id,
                    "name": other.name,
                    "status": other.status,
                    "distance": dist,
                    "x": other.x,
                    "y": other.y,
                }
            )
        return {"nearby_agents": nearby, "new_event_descriptions": []}

    @staticmethod
    def _normalize_control_mode(control_mode: str) -> str:
        normalized = str(control_mode or "").strip().lower()
        if normalized in {"autopilot", "external", "hybrid"}:
            return normalized
        return "external"

    @staticmethod
    def _normalize_autonomy_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"manual", "delegated", "autonomous"}:
            return normalized
        return "manual"

    @staticmethod
    def _normalize_autonomy_scopes(value: Any) -> set[str]:
        scopes: set[str] = set()
        if isinstance(value, list):
            for raw in value:
                scope = SimulationRunner._normalize_scope(str(raw))
                if scope in {"short_action", "daily_plan", "long_term_plan"}:
                    scopes.add(scope)
        if not scopes:
            scopes = {"short_action", "daily_plan", "long_term_plan"}
        return scopes

    @staticmethod
    def _as_bool(value: Any, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _clean_action_for_output(action: dict[str, Any] | None) -> dict[str, Any] | None:
        if not action:
            return None
        return {k: v for k, v in action.items() if not str(k).startswith("_")}

    @staticmethod
    def _social_pair_key(agent_a: str, agent_b: str) -> str:
        return "|".join(sorted((str(agent_a), str(agent_b))))

    @staticmethod
    def _chat_buffer_until(*, npc: NpcState, partner_name: str) -> int:
        raw = npc.memory.scratch.chatting_with_buffer.get(str(partner_name), -1)
        try:
            return int(raw)
        except Exception:
            return -1

    def _chat_buffer_active(self, *, npc: NpcState, partner_name: str, step: int) -> bool:
        return self._chat_buffer_until(npc=npc, partner_name=partner_name) > int(step)

    def _chat_buffer_pair_active(self, *, a: NpcState, b: NpcState, step: int) -> bool:
        return self._chat_buffer_active(npc=a, partner_name=b.name, step=step) or self._chat_buffer_active(
            npc=b,
            partner_name=a.name,
            step=step,
        )

    def _prune_chat_buffer(self, *, npc: NpcState, step: int) -> None:
        for key in list(npc.memory.scratch.chatting_with_buffer.keys()):
            until = self._chat_buffer_until(npc=npc, partner_name=key)
            if until <= int(step):
                npc.memory.scratch.chatting_with_buffer.pop(key, None)

    def _schedule_remaining_minutes(self, *, town: TownRuntime, npc: NpcState) -> int:
        schedule_item, _, schedule_offset = npc.memory.scratch.current_schedule_state(
            step=town.step,
            step_minutes=max(1, int(town.step_minutes)),
        )
        if schedule_item is None:
            return 999
        duration = max(1, int(schedule_item.duration_mins))
        elapsed = max(0, int(schedule_offset))
        return max(0, duration - elapsed)

    @staticmethod
    def _relationship_score(*, source: NpcState, target: NpcState) -> float:
        return float(source.memory.relationship_scores.get(target.agent_id, 0.0))

    def _clear_social_wait(self, *, town: TownRuntime, agent_id: str) -> None:
        town.social_wait_states.pop(str(agent_id), None)

    def _set_social_wait(
        self,
        *,
        town: TownRuntime,
        waiting_agent_id: str,
        target_agent_id: str,
        until_step: int,
        reason: str,
    ) -> None:
        waiting_id = str(waiting_agent_id).strip()
        target_id = str(target_agent_id).strip()
        if not waiting_id or not target_id:
            return
        town.social_wait_states[waiting_id] = {
            "target_agent_id": target_id,
            "until_step": max(int(town.step), int(until_step)),
            "reason": str(reason or "waiting_for_social_opening"),
        }

    def _active_social_wait(
        self,
        *,
        town: TownRuntime,
        agent_id: str,
        step: int,
    ) -> dict[str, Any] | None:
        payload = town.social_wait_states.get(str(agent_id))
        if not isinstance(payload, dict):
            return None
        target_agent_id = str(payload.get("target_agent_id") or "").strip()
        if target_agent_id and target_agent_id not in town.npcs:
            town.social_wait_states.pop(str(agent_id), None)
            return None
        try:
            until_step = int(payload.get("until_step") or 0)
        except Exception:
            until_step = 0
        if until_step < int(step):
            town.social_wait_states.pop(str(agent_id), None)
            return None
        return {
            "target_agent_id": target_agent_id,
            "until_step": until_step,
            "reason": str(payload.get("reason") or "waiting_for_social_opening"),
        }

    def _reaction_policy(
        self,
        *,
        town: TownRuntime,
        a: NpcState,
        b: NpcState,
        step: int,
    ) -> dict[str, Any]:
        a_remaining = self._schedule_remaining_minutes(town=town, npc=a)
        b_remaining = self._schedule_remaining_minutes(town=town, npc=b)
        relationship = (
            self._relationship_score(source=a, target=b) + self._relationship_score(source=b, target=a)
        ) / 2.0
        relationship = max(0.0, relationship)
        urgency_penalty = 0
        if a_remaining <= 20:
            urgency_penalty += 10
        if b_remaining <= 20:
            urgency_penalty += 10
        if a.agent_id in town.active_actions:
            urgency_penalty += 6
        if b.agent_id in town.active_actions:
            urgency_penalty += 6

        talk_threshold = 22 + min(22, int(relationship * 2.5)) - urgency_penalty
        talk_threshold = max(6, min(60, talk_threshold))
        wait_threshold = min(90, talk_threshold + 28)
        gate = self._hash_int(f"reaction:{a.agent_id}:{b.agent_id}:{step}") % 100

        if gate < talk_threshold:
            return {
                "decision": "talk",
                "reason": "social affinity and timing window favored conversation",
                "gate": gate,
                "talk_threshold": talk_threshold,
                "wait_threshold": wait_threshold,
                "relationship_score": round(relationship, 3),
            }

        if gate < wait_threshold:
            a_busy = a_remaining <= b_remaining
            waiter = b if a_busy else a
            target = a if waiter.agent_id == b.agent_id else b
            if waiter.agent_id in town.active_actions:
                return {
                    "decision": "ignore",
                    "reason": "candidate waiter has active external action",
                    "gate": gate,
                    "talk_threshold": talk_threshold,
                    "wait_threshold": wait_threshold,
                    "relationship_score": round(relationship, 3),
                }
            wait_steps = 2 if relationship >= 2.0 else 1
            return {
                "decision": "wait",
                "reason": "nearby social intent deferred for a short wait window",
                "gate": gate,
                "talk_threshold": talk_threshold,
                "wait_threshold": wait_threshold,
                "relationship_score": round(relationship, 3),
                "waiter_agent_id": waiter.agent_id,
                "target_agent_id": target.agent_id,
                "wait_steps": int(wait_steps),
            }

        return {
            "decision": "ignore",
            "reason": "reaction gate rejected this encounter",
            "gate": gate,
            "talk_threshold": talk_threshold,
            "wait_threshold": wait_threshold,
            "relationship_score": round(relationship, 3),
        }

    def _conversation_session_id(self, *, agent_a: str, agent_b: str) -> str:
        return self._social_pair_key(agent_a, agent_b)

    def _end_conversation_session(
        self,
        *,
        town: TownRuntime,
        session_id: str,
        end_step: int | None = None,
    ) -> None:
        session = town.conversation_sessions.pop(str(session_id), None)
        if session is None:
            return
        closed_step = max(0, int(town.step if end_step is None else end_step))
        for participant in session.participants():
            current = town.agent_conversations.get(participant)
            if current == session.session_id:
                town.agent_conversations.pop(participant, None)
        if int(session.turns) <= 0 and not session.last_message:
            return
        for participant in session.participants():
            npc = town.npcs.get(participant)
            if npc is None:
                continue
            partner_id = session.partner_for(participant)
            partner = town.npcs.get(partner_id) if partner_id else None
            partner_name = partner.name if partner is not None else str(partner_id or "someone")
            npc.memory.scratch.chatting_with_buffer[partner_name] = int(closed_step) + 4
            # Conversation summary is handled by memory._conversation_follow_up_thought()
            # which fires on the next tick when chatting_end_step is reached, producing
            # memo + planning thought nodes and clearing scratch chat state.
            # Setting chatting_end_step here ensures the follow-up triggers properly.
            if npc.memory.scratch.chatting_end_step is None or int(npc.memory.scratch.chatting_end_step) > closed_step:
                npc.memory.scratch.chatting_end_step = closed_step

    def _conversation_for_agent(self, *, town: TownRuntime, agent_id: str) -> ConversationSession | None:
        normalized_agent = str(agent_id)
        session_id = town.agent_conversations.get(normalized_agent)
        if not session_id:
            return None
        session = town.conversation_sessions.get(session_id)
        if session is None or not session.involves(normalized_agent):
            town.agent_conversations.pop(normalized_agent, None)
            return None
        return session

    @staticmethod
    def _conversation_ttl_steps(source: str) -> int:
        normalized = str(source or "").strip().lower()
        if normalized in {"external_action", "external_interrupt"}:
            return 3
        return 2

    def _start_or_refresh_conversation(
        self,
        *,
        town: TownRuntime,
        agent_a: str,
        agent_b: str,
        step: int,
        source: str,
        message: str | None = None,
    ) -> ConversationSession:
        a_id = str(agent_a)
        b_id = str(agent_b)
        session_id = self._conversation_session_id(agent_a=a_id, agent_b=b_id)

        for participant in (a_id, b_id):
            previous = town.agent_conversations.get(participant)
            if previous and previous != session_id:
                self._end_conversation_session(town=town, session_id=previous, end_step=step)
            self._clear_social_wait(town=town, agent_id=participant)

        session = town.conversation_sessions.get(session_id)
        ttl_steps = self._conversation_ttl_steps(source)
        if session is None:
            session = ConversationSession(
                session_id=session_id,
                agent_a=min(a_id, b_id),
                agent_b=max(a_id, b_id),
                source=str(source or "ambient_social"),
                started_step=int(step),
                last_step=int(step),
                expires_step=int(step) + ttl_steps,
                turns=0,
            )
            town.conversation_sessions[session_id] = session
        else:
            session.source = str(source or session.source or "ambient_social")
            session.last_step = int(step)
            session.expires_step = max(int(session.expires_step), int(step) + ttl_steps)
        if message:
            session.turns += 1
            session.last_message = str(message)
        town.agent_conversations[session.agent_a] = session.session_id
        town.agent_conversations[session.agent_b] = session.session_id
        return session

    def _prune_conversations(self, *, town: TownRuntime, step: int) -> None:
        for session_id in list(town.conversation_sessions.keys()):
            session = town.conversation_sessions.get(session_id)
            if session is None:
                continue
            if session.expires_step < int(step):
                self._end_conversation_session(town=town, session_id=session_id, end_step=step)
                continue
            a = town.npcs.get(session.agent_a)
            b = town.npcs.get(session.agent_b)
            if a is None or b is None:
                self._end_conversation_session(town=town, session_id=session_id, end_step=step)
                continue
            if not self._adjacent(a, b):
                self._end_conversation_session(town=town, session_id=session_id, end_step=step)
                continue

        for agent_id in list(town.agent_conversations.keys()):
            session_id = town.agent_conversations.get(agent_id)
            if not session_id:
                town.agent_conversations.pop(agent_id, None)
                continue
            session = town.conversation_sessions.get(session_id)
            if session is None or not session.involves(agent_id):
                town.agent_conversations.pop(agent_id, None)

    def _conversation_locks(self, *, town: TownRuntime) -> set[str]:
        locked: set[str] = set()
        for session in town.conversation_sessions.values():
            locked.add(session.agent_a)
            locked.add(session.agent_b)
        return locked

    def _conversation_event_payload(
        self,
        *,
        town: TownRuntime,
        step: int,
        session: ConversationSession,
        source: str = "conversation_session",
    ) -> dict[str, Any] | None:
        a = town.npcs.get(session.agent_a)
        b = town.npcs.get(session.agent_b)
        if a is None or b is None:
            return None
        message, transcript = self._generate_conversation_content(a=a, b=b, step=step, town=town)
        return {
            "step": int(step),
            "type": "social",
            "agents": [
                {"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y},
                {"agent_id": b.agent_id, "name": b.name, "x": b.x, "y": b.y},
            ],
            "message": message,
            "transcript": transcript,
            "source": source,
            "conversation_session_id": session.session_id,
        }

    def _build_conversation_continuation_events(self, *, town: TownRuntime, step: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        self._prune_conversations(town=town, step=step)
        for session_id in sorted(town.conversation_sessions.keys()):
            session = town.conversation_sessions.get(session_id)
            if session is None:
                continue
            if session.last_step >= int(step):
                continue
            gate = self._hash_int(f"conversation_gate:{session_id}:{step}") % 100
            if gate >= 45:
                continue
            payload = self._conversation_event_payload(town=town, step=step, session=session)
            if payload is None:
                continue
            out.append(payload)
            if len(out) >= self.max_social_events_per_step:
                break
        return out

    def _ingest_conversations_from_events(self, *, town: TownRuntime, social_events: list[dict[str, Any]], step: int) -> None:
        for event in social_events:
            if str(event.get("type") or "social") != "social":
                continue
            agents = event.get("agents") or []
            if len(agents) != 2:
                continue
            first = agents[0] if isinstance(agents[0], dict) else {}
            second = agents[1] if isinstance(agents[1], dict) else {}
            first_id = str(first.get("agent_id") or "").strip()
            second_id = str(second.get("agent_id") or "").strip()
            if not first_id or not second_id:
                continue
            if first_id not in town.npcs or second_id not in town.npcs:
                continue
            message = str(event.get("message") or "").strip() or None
            source = str(event.get("source") or "ambient_social")
            self._start_or_refresh_conversation(
                town=town,
                agent_a=first_id,
                agent_b=second_id,
                step=step,
                source=source,
                message=message,
            )

    def _nearby_interrupt_target(self, *, town: TownRuntime, npc: NpcState, step: int) -> dict[str, Any] | None:
        candidates: list[tuple[int, NpcState]] = []
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            if self._conversation_for_agent(town=town, agent_id=other.agent_id) is not None:
                continue
            if self._chat_buffer_pair_active(a=npc, b=other, step=step):
                continue
            dist = self._manhattan(npc.x, npc.y, other.x, other.y)
            if dist > 1:
                continue
            pair_key = self._social_pair_key(npc.agent_id, other.agent_id)
            try:
                cooldown_until = int(town.social_interrupt_cooldowns.get(pair_key, -1))
            except Exception:
                cooldown_until = -1
            if cooldown_until > int(step):
                continue
            candidates.append((dist, other))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1].agent_id))
        _, target = candidates[0]
        return {
            "agent_id": target.agent_id,
            "name": target.name,
            "x": target.x,
            "y": target.y,
        }

    @staticmethod
    def _active_action_done(
        *,
        active_action: dict[str, Any] | None,
        current_position: tuple[int, int],
    ) -> bool:
        if not active_action:
            return True
        try:
            remaining_steps = int(active_action.get("_remaining_steps", 0))
        except Exception:
            remaining_steps = 0
        if remaining_steps <= 1:
            return True
        target_x = active_action.get("target_x")
        target_y = active_action.get("target_y")
        if target_x is not None and target_y is not None:
            try:
                if int(target_x) == int(current_position[0]) and int(target_y) == int(current_position[1]):
                    return True
            except Exception:
                return False
        return False

    @staticmethod
    def _pending_social_items(*, action: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not action:
            return []
        raw_pending = action.get("_socials_pending")
        if not isinstance(raw_pending, list):
            raw_pending = []
            for item in action.get("socials") or []:
                if not isinstance(item, dict):
                    continue
                target_agent_id = str(item.get("target_agent_id") or "").strip()
                message = str(item.get("message") or "").strip()
                if not target_agent_id or not message:
                    continue
                transcript_raw = item.get("transcript") or []
                transcript: list[list[str]] = []
                for row in transcript_raw:
                    if not isinstance(row, (list, tuple)) or len(row) != 2:
                        continue
                    transcript.append([str(row[0]), str(row[1])])
                raw_pending.append(
                    {
                        "target_agent_id": target_agent_id,
                        "message": message,
                        "transcript": transcript,
                        "sent": False,
                    }
                )
            action["_socials_pending"] = raw_pending

        normalized_pending: list[dict[str, Any]] = []
        for item in raw_pending:
            if not isinstance(item, dict):
                continue
            target_agent_id = str(item.get("target_agent_id") or "").strip()
            message = str(item.get("message") or "").strip()
            if not target_agent_id or not message:
                continue
            transcript_raw = item.get("transcript") or []
            transcript: list[list[str]] = []
            for row in transcript_raw:
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    continue
                transcript.append([str(row[0]), str(row[1])])
            normalized_item: dict[str, Any] = {
                "target_agent_id": target_agent_id,
                "message": message,
                "transcript": transcript,
                "sent": bool(item.get("sent") or False),
            }
            if item.get("sent_step") is not None:
                try:
                    normalized_item["sent_step"] = int(item.get("sent_step"))
                except Exception:
                    pass
            normalized_pending.append(normalized_item)
        action["_socials_pending"] = normalized_pending
        return normalized_pending

    def _has_pending_socials(self, *, action: dict[str, Any] | None) -> bool:
        for item in self._pending_social_items(action=action):
            if not bool(item.get("sent") or False):
                return True
        return False

    def _next_pending_social_target(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        action: dict[str, Any] | None,
        step: int,
    ) -> NpcState | None:
        pending = self._pending_social_items(action=action)
        for item in pending:
            if bool(item.get("sent") or False):
                continue
            target_agent_id = str(item.get("target_agent_id") or "").strip()
            target = town.npcs.get(target_agent_id)
            if target is None or target.agent_id == npc.agent_id:
                item["sent"] = True
                item["sent_step"] = int(step)
                continue
            return target
        return None

    def _context_for_npc(self, *, town: TownRuntime, npc: NpcState, planning_scope: str) -> dict[str, Any]:
        self._update_location_memory(town=town, npc=npc, step=town.step)
        perception = self._perceive_nearby_agents_preview(town=town, npc=npc)
        focal_parts: list[str] = [planning_scope]
        if npc.current_location:
            focal_parts.append(npc.current_location)
        for other in perception["nearby_agents"]:
            focal_parts.append(str(other.get("status") or "idle"))
        focal_text = " ".join(i for i in focal_parts if i)
        retrieved = npc.memory.retrieve_ranked(
            focal_text=focal_text,
            step=town.step,
            limit=6,
        )

        goal_x: int | None = None
        goal_y: int | None = None
        goal_reason: str | None = None
        goal_affordance: str | None = None
        normalized_scope = self._normalize_scope(planning_scope)
        if normalized_scope == "long_term_plan":
            rel_goal = self._goal_from_relationship(town=town, npc=npc)
            if rel_goal is not None:
                goal_x, goal_y, goal_reason = rel_goal
        if goal_x is None or goal_y is None:
            location_goal = self._choose_location_goal(
                town=town,
                npc=npc,
                scope=normalized_scope,
            )
            if location_goal is not None:
                goal_x, goal_y, goal_reason, goal_affordance = location_goal

        schedule_item, schedule_index, schedule_offset = npc.memory.scratch.current_schedule_state(
            step=town.step,
            step_minutes=max(1, int(town.step_minutes)),
        )
        schedule: dict[str, Any] | None = None
        if schedule_item is not None:
            duration = max(1, int(schedule_item.duration_mins))
            elapsed = max(0, int(schedule_offset))
            schedule = {
                "index": int(schedule_index),
                "description": str(schedule_item.description),
                "affordance_hint": schedule_item.affordance_hint,
                "duration_mins": duration,
                "elapsed_mins": elapsed,
                "remaining_mins": max(0, duration - elapsed),
            }

        return {
            "town_id": town.town_id,
            "step": town.step,
            "clock": self._clock_for_runtime(town=town),
            "agent_id": npc.agent_id,
            "agent_name": npc.name,
            "persona": npc.persona,
            "planning_scope": planning_scope,
            "position": {"x": npc.x, "y": npc.y},
            "map": {"width": town.width, "height": town.height},
            "goal_hint": {
                "x": goal_x,
                "y": goal_y,
                "reason": goal_reason,
                "affordance": goal_affordance,
            },
            "perception": perception,
            "memory": {
                "summary": npc.memory.summary(),
                "retrieved": retrieved,
            },
            "schedule": schedule,
        }

    def _conversation_context_for_agent(self, *, town: TownRuntime, npc: NpcState) -> dict[str, Any] | None:
        session = self._conversation_for_agent(town=town, agent_id=npc.agent_id)
        if session is None:
            return None
        partner_id = session.partner_for(npc.agent_id)
        if not partner_id:
            return None
        partner = town.npcs.get(partner_id)
        if partner is None:
            return None
        return {
            "session_id": session.session_id,
            "partner_agent_id": partner.agent_id,
            "partner_name": partner.name,
            "partner_persona": partner.persona,
            "source": session.source,
            "started_step": int(session.started_step),
            "last_step": int(session.last_step),
            "expires_step": int(session.expires_step),
            "turns": int(session.turns),
            "last_message": session.last_message,
        }

    def _normalize_pending_action(self, *, action: dict[str, Any], step: int) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "_submitted_step": int(step),
        }
        if action.get("planning_scope"):
            normalized["planning_scope"] = str(action.get("planning_scope"))

        socials = action.get("socials") or []
        normalized_socials: list[dict[str, Any]] = []
        normalized_socials_pending: list[dict[str, Any]] = []
        for item in socials:
            if not isinstance(item, dict):
                continue
            target_agent_id = str(item.get("target_agent_id") or "").strip()
            message = str(item.get("message") or "").strip()
            if not target_agent_id or not message:
                continue
            transcript_raw = item.get("transcript") or []
            transcript: list[list[str]] = []
            for row in transcript_raw:
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    continue
                transcript.append([str(row[0]), str(row[1])])
            normalized_socials.append(
                {
                    "target_agent_id": target_agent_id,
                    "message": message,
                    "transcript": transcript,
                }
            )
            normalized_socials_pending.append(
                {
                    "target_agent_id": target_agent_id,
                    "message": message,
                    "transcript": transcript,
                    "sent": False,
                }
            )
        normalized["socials"] = normalized_socials
        normalized["_socials_pending"] = normalized_socials_pending

        default_ttl_steps = 200 if normalized_socials else 6
        try:
            action_ttl_steps = int(action.get("action_ttl_steps", default_ttl_steps))
        except Exception:
            action_ttl_steps = default_ttl_steps
        normalized["_action_ttl_steps"] = max(1, min(240, action_ttl_steps))
        normalized["_action_ttl_explicit"] = action.get("action_ttl_steps") is not None
        normalized["_remaining_steps"] = normalized["_action_ttl_steps"]
        normalized["_blocked_steps"] = 0
        normalized["_interrupt_on_social"] = self._as_bool(action.get("interrupt_on_social"), default=False)
        try:
            cooldown_steps = int(action.get("social_interrupt_cooldown_steps", 6))
        except Exception:
            cooldown_steps = 6
        normalized["_social_interrupt_cooldown_steps"] = max(1, min(240, cooldown_steps))

        if action.get("target_x") is not None and action.get("target_y") is not None:
            try:
                normalized["target_x"] = int(action.get("target_x"))
                normalized["target_y"] = int(action.get("target_y"))
            except Exception:
                pass
        if action.get("dx") is not None:
            normalized["dx"] = self._clamp_delta(action.get("dx"))
        if action.get("dy") is not None:
            normalized["dy"] = self._clamp_delta(action.get("dy"))
        if action.get("goal_reason"):
            normalized["goal_reason"] = str(action.get("goal_reason"))
        if action.get("affordance"):
            normalized["affordance"] = str(action.get("affordance"))
        if action.get("action_description"):
            normalized["action_description"] = str(action.get("action_description"))
        if action.get("action_ttl_steps") is not None:
            normalized["action_ttl_steps"] = normalized["_action_ttl_steps"]
        else:
            normalized["action_ttl_steps"] = normalized["_action_ttl_steps"]
        normalized["interrupt_on_social"] = normalized["_interrupt_on_social"]
        normalized["social_interrupt_cooldown_steps"] = normalized["_social_interrupt_cooldown_steps"]

        memory_nodes = action.get("memory_nodes") or []
        normalized_nodes: list[dict[str, Any]] = []
        for item in memory_nodes:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind not in {"event", "thought", "chat", "reflection"}:
                continue
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            node_payload = {
                "kind": kind,
                "description": description,
                "subject": str(item.get("subject") or "").strip(),
                "predicate": str(item.get("predicate") or "").strip(),
                "object": str(item.get("object") or "").strip(),
                "poignancy": max(1, min(10, int(item.get("poignancy") or 3))),
                "evidence_ids": [str(v) for v in (item.get("evidence_ids") or []) if str(v).strip()],
            }
            normalized_nodes.append(node_payload)
        normalized["memory_nodes"] = normalized_nodes
        return normalized

    def _build_social_events_from_action(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        action: dict[str, Any] | None,
        step: int,
    ) -> list[dict[str, Any]]:
        if not action:
            return []
        socials = self._pending_social_items(action=action)
        out: list[dict[str, Any]] = []
        for item in socials:
            if bool(item.get("sent") or False):
                continue
            target_agent_id = str(item.get("target_agent_id") or "")
            message = str(item.get("message") or "")
            if not target_agent_id or not message:
                continue
            target = town.npcs.get(target_agent_id)
            if not target:
                item["sent"] = True
                item["sent_step"] = int(step)
                continue
            if target.agent_id == npc.agent_id:
                item["sent"] = True
                item["sent_step"] = int(step)
                continue
            if not self._adjacent(npc, target):
                continue
            if self._conversation_for_agent(town=town, agent_id=npc.agent_id) is not None:
                continue
            if self._conversation_for_agent(town=town, agent_id=target.agent_id) is not None:
                continue
            if self._chat_buffer_pair_active(a=npc, b=target, step=step):
                continue
            pair_key = self._social_pair_key(npc.agent_id, target.agent_id)
            try:
                cooldown_until = int(town.social_interrupt_cooldowns.get(pair_key, -1))
            except Exception:
                cooldown_until = -1
            if cooldown_until > int(step):
                continue
            transcript = item.get("transcript") or []
            if not transcript:
                transcript = [[npc.name, message]]
            session = self._start_or_refresh_conversation(
                town=town,
                agent_a=npc.agent_id,
                agent_b=target.agent_id,
                step=step,
                source="external_action",
                message=None,
            )
            out.append(
                {
                    "step": int(step),
                    "type": "social",
                    "agents": [
                        {"agent_id": npc.agent_id, "name": npc.name, "x": npc.x, "y": npc.y},
                        {"agent_id": target.agent_id, "name": target.name, "x": target.x, "y": target.y},
                    ],
                    "message": message,
                    "transcript": transcript,
                    "source": "external_action",
                    "conversation_session_id": session.session_id,
                }
            )
            item["sent"] = True
            item["sent_step"] = int(step)
            try:
                cooldown_steps = int(action.get("_social_interrupt_cooldown_steps", 6))
            except Exception:
                cooldown_steps = 6
            cooldown_steps = max(1, cooldown_steps)
            town.social_interrupt_cooldowns[pair_key] = int(step) + cooldown_steps
            if len(out) >= self.max_social_events_per_step:
                break
        return out

    def _ingest_memory_nodes_from_action(
        self,
        *,
        npc: NpcState,
        action: dict[str, Any] | None,
        step: int,
    ) -> None:
        if not action:
            return
        nodes = action.get("memory_nodes") or []
        for node in nodes:
            kind = str(node.get("kind") or "").strip().lower()
            if kind not in {"event", "thought", "chat", "reflection"}:
                continue
            description = str(node.get("description") or "").strip()
            if not description:
                continue
            subject = str(node.get("subject") or npc.name).strip() or npc.name
            predicate = str(node.get("predicate") or "notes").strip() or "notes"
            obj = str(node.get("object") or "context").strip() or "context"
            evidence_ids = tuple(str(v) for v in node.get("evidence_ids") or [] if str(v).strip())
            npc.memory.add_node(
                kind=kind,
                step=int(step),
                subject=subject,
                predicate=predicate,
                object=obj,
                description=description,
                poignancy=max(1, min(10, int(node.get("poignancy") or 3))),
                evidence_ids=evidence_ids,
                decrement_trigger=False,
            )

    def _ingest_social_events(self, *, town: TownRuntime, social_events: list[dict[str, Any]], step: int) -> None:
        for event in social_events:
            if str(event.get("type") or "social") != "social":
                continue
            agents = event.get("agents") or []
            if len(agents) != 2:
                continue
            first = agents[0]
            second = agents[1]
            a_id = str(first.get("agent_id") or "")
            b_id = str(second.get("agent_id") or "")
            a = town.npcs.get(a_id)
            b = town.npcs.get(b_id)
            if not a or not b:
                continue
            message = str(event.get("message") or "")
            transcript_raw = event.get("transcript") or []
            transcript: list[tuple[str, str]] = []
            for row in transcript_raw:
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    continue
                transcript.append((str(row[0]), str(row[1])))

            # Estimate conversation duration from transcript length
            total_chars = sum(len(line) for _, line in transcript)
            convo_duration_mins = max(10, total_chars // 30)

            a.memory.add_social_interaction(
                step=step,
                target_agent_id=b.agent_id,
                target_name=b.name,
                message=message,
                transcript=transcript,
            )
            b.memory.add_social_interaction(
                step=step,
                target_agent_id=a.agent_id,
                target_name=a.name,
                message=message,
                transcript=transcript,
            )

            # Recompose schedules for both participants after chat interruption
            a.memory.recompose_schedule_after_interruption(
                step=step,
                inserted_description=f"chatting with {b.name}",
                inserted_duration_mins=convo_duration_mins,
            )
            b.memory.recompose_schedule_after_interruption(
                step=step,
                inserted_description=f"chatting with {a.name}",
                inserted_duration_mins=convo_duration_mins,
            )

    def _run_reflection_cycle(self, *, town: TownRuntime, step: int) -> list[dict[str, Any]]:
        reflection_events: list[dict[str, Any]] = []
        for npc in sorted(town.npcs.values(), key=lambda n: n.agent_id):
            reflections = npc.memory.maybe_reflect(step=step)
            for node in reflections:
                reflection_events.append(
                    {
                        "step": step,
                        "type": "reflection",
                        "agent_id": npc.agent_id,
                        "name": npc.name,
                        "node": node.as_dict(),
                    }
                )
        return reflection_events

    def sync_town(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
    ) -> TownRuntime:
        runtime = self._towns.get(town_id)
        active_map_version_id = str(active_version["id"])
        if runtime is None or runtime.map_version_id != active_map_version_id:
            runtime = self._load_persisted_runtime(
                town_id=town_id,
                active_version=active_version,
            )
            if runtime is None:
                runtime = self._create_town_runtime(
                    town_id=town_id,
                    active_version=active_version,
                    map_data=map_data,
                )
            self._towns[town_id] = runtime

        limited_members = self._normalize_members(members, self.max_agents_per_town)
        allowed_ids = {str(m["agent_id"]) for m in limited_members if m.get("agent_id")}

        for agent_id in list(runtime.npcs.keys()):
            if agent_id not in allowed_ids:
                runtime.npcs.pop(agent_id, None)
                runtime.pending_actions.pop(agent_id, None)
                runtime.active_actions.pop(agent_id, None)

        for agent_id in list(runtime.pending_actions.keys()):
            if agent_id not in allowed_ids:
                runtime.pending_actions.pop(agent_id, None)
        for agent_id in list(runtime.active_actions.keys()):
            if agent_id not in allowed_ids:
                runtime.active_actions.pop(agent_id, None)
        for agent_id in list(runtime.autonomy_tick_counts.keys()):
            if agent_id not in allowed_ids:
                runtime.autonomy_tick_counts.pop(agent_id, None)
        for pair_key in list(runtime.social_interrupt_cooldowns.keys()):
            parts = pair_key.split("|")
            if len(parts) != 2:
                runtime.social_interrupt_cooldowns.pop(pair_key, None)
                continue
            if parts[0] not in allowed_ids or parts[1] not in allowed_ids:
                runtime.social_interrupt_cooldowns.pop(pair_key, None)
        for waiting_agent_id in list(runtime.social_wait_states.keys()):
            if waiting_agent_id not in allowed_ids:
                runtime.social_wait_states.pop(waiting_agent_id, None)
                continue
            payload = runtime.social_wait_states.get(waiting_agent_id)
            target_agent_id = str((payload or {}).get("target_agent_id") or "").strip()
            if target_agent_id and target_agent_id not in allowed_ids:
                runtime.social_wait_states.pop(waiting_agent_id, None)

        for session_id in list(runtime.conversation_sessions.keys()):
            session = runtime.conversation_sessions.get(session_id)
            if session is None:
                runtime.conversation_sessions.pop(session_id, None)
                continue
            if session.agent_a not in allowed_ids or session.agent_b not in allowed_ids:
                self._end_conversation_session(town=runtime, session_id=session_id, end_step=runtime.step)

        for agent_id in list(runtime.agent_conversations.keys()):
            if agent_id not in allowed_ids:
                runtime.agent_conversations.pop(agent_id, None)

        self._prune_conversations(town=runtime, step=runtime.step)

        for idx, member in enumerate(limited_members):
            agent_id = str(member["agent_id"])
            npc = runtime.npcs.get(agent_id)
            if npc is None:
                spawn_x, spawn_y = self._spawn_for(runtime, idx)
                member_persona = member.get("personality") if isinstance(member.get("personality"), dict) else None
                runtime.npcs[agent_id] = NpcState(
                    agent_id=agent_id,
                    name=str(member.get("name") or f"Agent-{agent_id[:8]}"),
                    owner_handle=member.get("owner_handle"),
                    claim_status=str(member.get("claim_status") or "unknown"),
                    x=spawn_x,
                    y=spawn_y,
                    joined_at=member.get("joined_at"),
                    status="idle",
                    sprite_name=member.get("sprite_name"),
                    persona=member_persona,
                    memory=AgentMemory.bootstrap(
                        agent_id=agent_id,
                        agent_name=str(member.get("name") or f"Agent-{agent_id[:8]}"),
                        step=runtime.step,
                        persona=member_persona,
                    ),
                )
                runtime.npcs[agent_id].memory.set_position(x=spawn_x, y=spawn_y, step=runtime.step)
                continue

            npc.name = str(member.get("name") or npc.name)
            npc.owner_handle = member.get("owner_handle")
            npc.claim_status = str(member.get("claim_status") or npc.claim_status)
            npc.joined_at = member.get("joined_at")
            npc.memory.agent_name = npc.name
            npc.memory.set_position(x=npc.x, y=npc.y, step=runtime.step)

        runtime.updated_at = _utc_now()
        self._persist_runtime(runtime)
        return runtime

    def get_state(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
    ) -> dict[str, Any]:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )
        return runtime.to_dict(max_agents=self.max_agents_per_town)

    def get_agent_memory(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        agent_id: str,
        limit: int = 40,
    ) -> dict[str, Any] | None:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )
        npc = runtime.npcs.get(str(agent_id))
        if not npc:
            return None
        return {
            "town_id": runtime.town_id,
            "step": runtime.step,
            "agent": {
                "agent_id": npc.agent_id,
                "name": npc.name,
                "current_location": npc.current_location,
                "memory": npc.memory.snapshot(limit=limit),
            },
        }

    def get_agent_context(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        agent_id: str,
        planning_scope: str = "short_action",
    ) -> dict[str, Any] | None:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )
        npc = runtime.npcs.get(str(agent_id))
        if not npc:
            return None
        pending = runtime.pending_actions.get(npc.agent_id)
        active = runtime.active_actions.get(npc.agent_id)
        pending_clean = self._clean_action_for_output(pending)
        active_clean = self._clean_action_for_output(active)
        return {
            "town_id": runtime.town_id,
            "step": runtime.step,
            "clock": self._clock_for_runtime(town=runtime),
            "agent": {
                "agent_id": npc.agent_id,
                "name": npc.name,
                "current_location": npc.current_location,
                "status": npc.status,
            },
            "planning_scope": planning_scope,
            "context": self._context_for_npc(
                town=runtime,
                npc=npc,
                planning_scope=planning_scope,
            ),
            "active_conversation": self._conversation_context_for_agent(town=runtime, npc=npc),
            "pending_action": pending_clean,
            "active_action": active_clean,
        }

    def submit_agent_action(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        agent_id: str,
        action: dict[str, Any],
    ) -> dict[str, Any] | None:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )
        npc = runtime.npcs.get(str(agent_id))
        if not npc:
            return None
        normalized = self._normalize_pending_action(action=action, step=runtime.step)
        runtime.active_actions.pop(npc.agent_id, None)
        runtime.pending_actions[npc.agent_id] = normalized
        runtime.social_wait_states.pop(npc.agent_id, None)
        runtime.updated_at = _utc_now()
        self._persist_runtime(runtime)
        return {
            "ok": True,
            "town_id": runtime.town_id,
            "step": runtime.step,
            "agent_id": npc.agent_id,
            "queued_action": self._clean_action_for_output(normalized),
        }

    def tick(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        steps: int = 1,
        planning_scope: str = "short_action",
        control_mode: str = "external",
        planner: Optional[PlannerFn] = None,
        agent_autonomy: dict[str, dict[str, Any]] | None = None,
        cognition_planner: Any = None,
    ) -> dict[str, Any]:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )

        # Wire cognition planner into all NPC memory instances
        if cognition_planner is not None:
            for npc in runtime.npcs.values():
                npc.memory.cognition = cognition_planner
                npc.memory.town_id = town_id

        mode = self._normalize_control_mode(control_mode)
        events: list[dict[str, Any]] = []
        social_events: list[dict[str, Any]] = []
        reflection_events: list[dict[str, Any]] = []
        clamped_steps = max(1, int(steps))

        for _ in range(clamped_steps):
            runtime.step += 1
            self._prune_conversations(town=runtime, step=runtime.step)
            step_external_social_events: list[dict[str, Any]] = []
            step_conversation_locks = self._conversation_locks(town=runtime)

            for agent_id in sorted(runtime.npcs.keys()):
                npc = runtime.npcs[agent_id]
                self._prune_chat_buffer(npc=npc, step=runtime.step)
                self._update_location_memory(town=runtime, npc=npc, step=runtime.step)

                submitted_action: dict[str, Any] | None = None
                active_action: dict[str, Any] | None = runtime.active_actions.get(agent_id)
                if mode in {"external", "hybrid"}:
                    submitted_action = runtime.pending_actions.pop(agent_id, None)
                    if submitted_action is not None:
                        active_action = dict(submitted_action)
                        runtime.active_actions[agent_id] = active_action

                use_external_action = submitted_action is not None and mode in {"external", "hybrid"}
                continue_external_action = submitted_action is None and active_action is not None and mode in {"external", "hybrid"}
                has_external_action = use_external_action or continue_external_action
                autonomy_contract_raw = (
                    agent_autonomy.get(agent_id, {})
                    if isinstance(agent_autonomy, dict)
                    else {}
                )
                autonomy_mode = self._normalize_autonomy_mode(autonomy_contract_raw.get("mode"))
                autonomy_allowed_scopes = self._normalize_autonomy_scopes(autonomy_contract_raw.get("allowed_scopes"))
                try:
                    autonomy_max_ticks = max(0, int(autonomy_contract_raw.get("max_autonomous_ticks") or 0))
                except Exception:
                    autonomy_max_ticks = 0
                autonomy_scope = self._normalize_scope(planning_scope)
                autonomy_scope_allowed = autonomy_scope in autonomy_allowed_scopes
                autonomy_ticks_used = int(runtime.autonomy_tick_counts.get(agent_id, 0))
                autonomy_limit_reached = autonomy_max_ticks > 0 and autonomy_ticks_used >= autonomy_max_ticks
                delegated_autopilot_enabled = (
                    mode in {"external", "hybrid"}
                    and not has_external_action
                    and autonomy_mode in {"delegated", "autonomous"}
                    and autonomy_scope_allowed
                    and not autonomy_limit_reached
                )
                autopilot_enabled = (
                    (mode in {"autopilot", "hybrid"} and not has_external_action)
                    or delegated_autopilot_enabled
                )

                if has_external_action:
                    runtime.autonomy_tick_counts[agent_id] = 0

                if autopilot_enabled:
                    npc.memory.maybe_refresh_plans(
                        step=runtime.step,
                        scope=planning_scope,
                        step_minutes=max(1, int(runtime.step_minutes)),
                    )

                perception = self._perceive_nearby_agents(town=runtime, npc=npc, step=runtime.step)
                focal_parts: list[str] = [planning_scope]
                if npc.current_location:
                    focal_parts.append(npc.current_location)
                focal_parts.extend(perception["new_event_descriptions"])
                focal_text = " ".join(i for i in focal_parts if i)
                retrieved = npc.memory.retrieve_ranked(
                    focal_text=focal_text,
                    step=runtime.step,
                    limit=6,
                )

                from_x, from_y = npc.x, npc.y
                planner_meta: dict[str, Any]
                dx = 0
                dy = 0
                goal_x: int | None = None
                goal_y: int | None = None
                goal_reason: str | None = None
                goal_affordance: str | None = None
                external_action_description: str | None = None
                social_interrupted = False
                external_action: dict[str, Any] | None = (active_action or submitted_action or {}) if has_external_action else None
                conversation_session = self._conversation_for_agent(town=runtime, agent_id=agent_id)
                pause_for_conversation = (
                    conversation_session is not None and npc.agent_id in step_conversation_locks
                )
                wait_state = self._active_social_wait(
                    town=runtime,
                    agent_id=agent_id,
                    step=runtime.step,
                )
                pause_for_wait = (
                    wait_state is not None
                    and not has_external_action
                    and not pause_for_conversation
                )
                action_force_finish = False

                normalized_scope = self._normalize_scope(planning_scope)
                if pause_for_conversation and conversation_session is not None:
                    if external_action is not None:
                        if external_action.get("target_x") is not None and external_action.get("target_y") is not None:
                            try:
                                goal_x = int(external_action.get("target_x"))
                                goal_y = int(external_action.get("target_y"))
                            except Exception:
                                goal_x = None
                                goal_y = None
                        if external_action.get("goal_reason"):
                            goal_reason = str(external_action.get("goal_reason"))
                        if external_action.get("affordance"):
                            goal_affordance = str(external_action.get("affordance"))
                        if external_action.get("action_description"):
                            external_action_description = str(external_action.get("action_description"))
                    partner_id = conversation_session.partner_for(npc.agent_id)
                    partner = runtime.npcs.get(partner_id) if partner_id else None
                    route = "external_action_conversation_pause" if has_external_action else "conversation_session"
                    reason = "active conversation session keeps both agents stationary"
                    policy_task = str((external_action or {}).get("planning_scope") or planning_scope)
                    planner_meta = {
                        "route": route,
                        "used_tier": "external_agent" if has_external_action else None,
                        "context_trimmed": False,
                        "reason": reason,
                        "policy_task": policy_task,
                        "retrieved_count": len(retrieved),
                        "perceived_new_events": len(perception["new_event_descriptions"]),
                        "goal": {
                            "x": goal_x,
                            "y": goal_y,
                            "reason": goal_reason,
                            "affordance": goal_affordance,
                        },
                        "conversation_session_id": conversation_session.session_id,
                    }
                    if partner is not None:
                        planner_meta["conversation_partner"] = {
                            "agent_id": partner.agent_id,
                            "name": partner.name,
                        }
                elif pause_for_wait and wait_state is not None:
                    target_id = str(wait_state.get("target_agent_id") or "").strip()
                    target = runtime.npcs.get(target_id) if target_id else None
                    goal_x = int(target.x) if target is not None else None
                    goal_y = int(target.y) if target is not None else None
                    goal_reason = str(wait_state.get("reason") or "waiting_for_social_opening")
                    planner_meta = {
                        "route": "reaction_wait",
                        "used_tier": "reaction_policy",
                        "context_trimmed": False,
                        "reason": goal_reason,
                        "policy_task": "reaction_policy",
                        "retrieved_count": len(retrieved),
                        "perceived_new_events": len(perception["new_event_descriptions"]),
                        "goal": {
                            "x": goal_x,
                            "y": goal_y,
                            "reason": goal_reason,
                            "affordance": None,
                        },
                        "wait_until_step": int(wait_state.get("until_step") or runtime.step),
                        "wait_target_agent_id": target_id or None,
                    }
                    if target is not None:
                        planner_meta["wait_target_name"] = target.name
                elif autopilot_enabled:
                    if normalized_scope == "long_term_plan":
                        rel_goal = self._goal_from_relationship(town=runtime, npc=npc)
                        if rel_goal is not None:
                            goal_x, goal_y, goal_reason = rel_goal

                    if goal_x is None or goal_y is None:
                        location_goal = self._choose_location_goal(
                            town=runtime,
                            npc=npc,
                            scope=normalized_scope,
                        )
                        if location_goal is not None:
                            goal_x, goal_y, goal_reason, goal_affordance = location_goal

                    if planner:
                        context = {
                            "town_id": town_id,
                            "step": runtime.step,
                            "agent_id": npc.agent_id,
                            "agent_name": npc.name,
                            "planning_scope": planning_scope,
                            "position": {"x": from_x, "y": from_y},
                            "map": {"width": runtime.width, "height": runtime.height},
                            "goal_hint": {
                                "x": goal_x,
                                "y": goal_y,
                                "reason": goal_reason,
                                "affordance": goal_affordance,
                            },
                            "perception": perception,
                            "memory": {
                                "summary": npc.memory.summary(),
                                "retrieved": retrieved,
                            },
                        }
                        try:
                            decision = planner(context) or {}
                            dx = self._clamp_delta(decision.get("dx", 0))
                            dy = self._clamp_delta(decision.get("dy", 0))
                            if decision.get("target_x") is not None and decision.get("target_y") is not None:
                                try:
                                    goal_x = int(decision.get("target_x"))
                                    goal_y = int(decision.get("target_y"))
                                except Exception:
                                    pass
                            if decision.get("goal_reason"):
                                goal_reason = str(decision.get("goal_reason"))
                            if decision.get("affordance"):
                                goal_affordance = str(decision.get("affordance"))
                            planner_meta = {
                                "route": decision.get("route", "planner"),
                                "used_tier": decision.get("used_tier"),
                                "context_trimmed": bool(decision.get("context_trimmed", False)),
                                "reason": decision.get("reason"),
                                "policy_task": decision.get("policy_task"),
                                "retrieved_count": len(retrieved),
                                "perceived_new_events": len(perception["new_event_descriptions"]),
                                "goal": {
                                    "x": goal_x,
                                    "y": goal_y,
                                    "reason": goal_reason,
                                    "affordance": goal_affordance,
                                },
                            }
                        except Exception as exc:
                            dx, dy = 0, 0
                            planner_meta = {
                                "route": "planner_error",
                                "reason": str(exc),
                                "used_tier": "heuristic",
                                "context_trimmed": False,
                                "policy_task": "plan_move",
                                "retrieved_count": len(retrieved),
                                "perceived_new_events": len(perception["new_event_descriptions"]),
                                "goal": {
                                    "x": goal_x,
                                    "y": goal_y,
                                    "reason": goal_reason,
                                    "affordance": goal_affordance,
                                },
                            }
                    else:
                        dx, dy = self._step_direction(agent_id=agent_id, step=runtime.step)
                        planner_meta = {
                            "route": "deterministic_hash",
                            "used_tier": None,
                            "context_trimmed": False,
                            "reason": "legacy deterministic movement",
                            "policy_task": "plan_move",
                            "retrieved_count": len(retrieved),
                            "perceived_new_events": len(perception["new_event_descriptions"]),
                            "goal": {
                                "x": goal_x,
                                "y": goal_y,
                                "reason": goal_reason,
                                "affordance": goal_affordance,
                            },
                        }
                elif has_external_action and external_action is not None:
                    if external_action.get("target_x") is not None and external_action.get("target_y") is not None:
                        try:
                            goal_x = int(external_action.get("target_x"))
                            goal_y = int(external_action.get("target_y"))
                        except Exception:
                            goal_x = None
                            goal_y = None
                    dx = self._clamp_delta(external_action.get("dx", 0))
                    dy = self._clamp_delta(external_action.get("dy", 0))
                    if external_action.get("goal_reason"):
                        goal_reason = str(external_action.get("goal_reason"))
                    if external_action.get("affordance"):
                        goal_affordance = str(external_action.get("affordance"))
                    if external_action.get("action_description"):
                        external_action_description = str(external_action.get("action_description"))
                    pending_social_target = self._next_pending_social_target(
                        town=runtime,
                        npc=npc,
                        action=external_action,
                        step=runtime.step,
                    )
                    if pending_social_target is not None and (goal_x is None or goal_y is None):
                        goal_x = int(pending_social_target.x)
                        goal_y = int(pending_social_target.y)
                    if pending_social_target is not None and not goal_reason:
                        goal_reason = f"talk_with_{pending_social_target.name}"

                    policy_task = str(external_action.get("planning_scope") or planning_scope)
                    if use_external_action:
                        planner_meta = {
                            "route": "external_action",
                            "used_tier": "external_agent",
                            "context_trimmed": False,
                            "reason": "applied action submitted by user-controlled agent",
                            "policy_task": policy_task,
                            "retrieved_count": len(retrieved),
                            "perceived_new_events": len(perception["new_event_descriptions"]),
                            "goal": {
                                "x": goal_x,
                                "y": goal_y,
                                "reason": goal_reason,
                                "affordance": goal_affordance,
                            },
                        }
                    else:
                        planner_meta = {
                            "route": "external_action_continuation",
                            "used_tier": "external_agent",
                            "context_trimmed": False,
                            "reason": "continuing previously submitted external action",
                            "policy_task": policy_task,
                            "retrieved_count": len(retrieved),
                            "perceived_new_events": len(perception["new_event_descriptions"]),
                            "goal": {
                                "x": goal_x,
                                "y": goal_y,
                                "reason": goal_reason,
                                "affordance": goal_affordance,
                            },
                        }
                    if pending_social_target is not None:
                        planner_meta["pending_social_target"] = {
                            "agent_id": pending_social_target.agent_id,
                            "name": pending_social_target.name,
                        }

                    if self._as_bool(external_action.get("_interrupt_on_social"), default=True):
                        interrupt_target = self._nearby_interrupt_target(
                            town=runtime,
                            npc=npc,
                            step=runtime.step,
                        )
                        if interrupt_target is not None:
                            social_interrupted = True
                            target_agent_id = str(interrupt_target["agent_id"])
                            try:
                                cooldown_steps = int(external_action.get("_social_interrupt_cooldown_steps", 6))
                            except Exception:
                                cooldown_steps = 6
                            cooldown_steps = max(1, cooldown_steps)
                            pair_key = self._social_pair_key(npc.agent_id, target_agent_id)
                            runtime.social_interrupt_cooldowns[pair_key] = runtime.step + cooldown_steps
                            interrupt_partner = runtime.npcs.get(target_agent_id)
                            interrupt_message: str | None = None
                            interrupt_transcript: list[list[str]] = []
                            if interrupt_partner is not None:
                                interrupt_message = self._social_message(a=npc, b=interrupt_partner, step=runtime.step)
                                interrupt_transcript = self._social_transcript(a=npc, b=interrupt_partner, step=runtime.step)
                            conversation_session = self._start_or_refresh_conversation(
                                town=runtime,
                                agent_a=npc.agent_id,
                                agent_b=target_agent_id,
                                step=runtime.step,
                                source="external_interrupt",
                                message=None,
                            )
                            pause_for_conversation = True
                            step_conversation_locks.add(npc.agent_id)
                            step_conversation_locks.add(target_agent_id)
                            if interrupt_message and interrupt_partner is not None:
                                step_external_social_events.append(
                                    {
                                        "step": int(runtime.step),
                                        "type": "social",
                                        "agents": [
                                            {"agent_id": npc.agent_id, "name": npc.name, "x": npc.x, "y": npc.y},
                                            {
                                                "agent_id": interrupt_partner.agent_id,
                                                "name": interrupt_partner.name,
                                                "x": interrupt_partner.x,
                                                "y": interrupt_partner.y,
                                            },
                                        ],
                                        "message": interrupt_message,
                                        "transcript": interrupt_transcript,
                                        "source": "external_interrupt",
                                        "conversation_session_id": conversation_session.session_id,
                                    }
                                )
                            planner_meta = {
                                "route": "external_action_social_interrupt",
                                "used_tier": "external_agent",
                                "context_trimmed": False,
                                "reason": "nearby social opportunity interrupted the active action",
                                "policy_task": policy_task,
                                "retrieved_count": len(retrieved),
                                "perceived_new_events": len(perception["new_event_descriptions"]),
                                "goal": {
                                    "x": goal_x,
                                    "y": goal_y,
                                    "reason": goal_reason,
                                    "affordance": goal_affordance,
                                },
                                "interrupt_target": interrupt_target,
                                "conversation_session_id": conversation_session.session_id,
                            }
                else:
                    waiting_route = "awaiting_user_agent"
                    waiting_reason = "no queued external action for this step"
                    if (
                        mode in {"external", "hybrid"}
                        and not has_external_action
                        and autonomy_mode in {"delegated", "autonomous"}
                        and autonomy_scope_allowed
                        and autonomy_limit_reached
                    ):
                        waiting_route = "awaiting_user_agent_autonomy_limit"
                        waiting_reason = "autonomy contract max_autonomous_ticks limit reached"
                    planner_meta = {
                        "route": waiting_route,
                        "used_tier": None,
                        "context_trimmed": False,
                        "reason": waiting_reason,
                        "policy_task": planning_scope,
                        "retrieved_count": len(retrieved),
                        "perceived_new_events": len(perception["new_event_descriptions"]),
                        "goal": {
                            "x": None,
                            "y": None,
                            "reason": None,
                            "affordance": None,
                        },
                    }

                planner_meta["autonomy_mode"] = autonomy_mode
                planner_meta["autonomy_scope_allowed"] = bool(autonomy_scope_allowed)
                planner_meta["autonomy_ticks_used"] = int(autonomy_ticks_used)
                planner_meta["autonomy_max_ticks"] = int(autonomy_max_ticks)
                planner_meta["delegated_autopilot"] = bool(delegated_autopilot_enabled)
                if delegated_autopilot_enabled:
                    planner_meta["autonomy_source"] = "delegated_contract"

                if pause_for_conversation:
                    to_x, to_y = from_x, from_y
                    planner_meta["movement_mode"] = "paused_conversation"
                elif pause_for_wait:
                    to_x, to_y = from_x, from_y
                    planner_meta["movement_mode"] = "paused_wait"
                elif social_interrupted:
                    to_x, to_y = from_x, from_y
                    planner_meta["movement_mode"] = "paused_social_interrupt"
                else:
                    if goal_x is not None and goal_y is not None:
                        path_target_x = goal_x
                        path_target_y = goal_y
                        using_reroute = False
                        if has_external_action and external_action is not None:
                            reroute_x_raw = external_action.get("_reroute_target_x")
                            reroute_y_raw = external_action.get("_reroute_target_y")
                            if reroute_x_raw is not None and reroute_y_raw is not None:
                                try:
                                    reroute_x = int(reroute_x_raw)
                                    reroute_y = int(reroute_y_raw)
                                except Exception:
                                    reroute_x = goal_x
                                    reroute_y = goal_y
                                if from_x != reroute_x or from_y != reroute_y:
                                    path_target_x = reroute_x
                                    path_target_y = reroute_y
                                    using_reroute = True
                                    planner_meta["reroute_target"] = {"x": reroute_x, "y": reroute_y}
                        next_x, next_y = self._next_step_towards(
                            town=runtime,
                            start=(from_x, from_y),
                            target=(path_target_x, path_target_y),
                        )
                        to_x, to_y = next_x, next_y
                        planner_meta["movement_mode"] = "path_to_reroute_goal" if using_reroute else "path_to_goal"
                        if has_external_action and external_action is not None:
                            try:
                                blocked_before = int(external_action.get("_blocked_steps", 0))
                            except Exception:
                                blocked_before = 0
                            blocked_before = max(0, blocked_before)
                            goal_reached = to_x == goal_x and to_y == goal_y
                            moved = (to_x != from_x) or (to_y != from_y)
                            blocked_after = 0 if moved or goal_reached else (blocked_before + 1)
                            external_action["_blocked_steps"] = blocked_after
                            planner_meta["blocked_steps"] = blocked_after

                            if using_reroute and to_x == path_target_x and to_y == path_target_y:
                                external_action.pop("_reroute_target_x", None)
                                external_action.pop("_reroute_target_y", None)
                                planner_meta["reroute_reached"] = True

                            if not moved and not goal_reached:
                                reroute_target = self._reachable_tile_near_target(
                                    town=runtime,
                                    start=(from_x, from_y),
                                    target=(goal_x, goal_y),
                                )
                                if reroute_target is not None:
                                    external_action["_reroute_target_x"] = int(reroute_target[0])
                                    external_action["_reroute_target_y"] = int(reroute_target[1])
                                    planner_meta["reroute_target"] = {
                                        "x": int(reroute_target[0]),
                                        "y": int(reroute_target[1]),
                                    }
                                    reroute_x, reroute_y = self._next_step_towards(
                                        town=runtime,
                                        start=(from_x, from_y),
                                        target=reroute_target,
                                    )
                                    if reroute_x != from_x or reroute_y != from_y:
                                        to_x, to_y = reroute_x, reroute_y
                                        external_action["_blocked_steps"] = 0
                                        planner_meta["blocked_steps"] = 0
                                        planner_meta["movement_mode"] = "reroute_near_goal"
                                if int(external_action.get("_blocked_steps", 0)) >= 3:
                                    action_force_finish = True
                                    planner_meta["route"] = "external_action_stuck"
                                    planner_meta["reason"] = "external action target is unreachable from current position"
                                    planner_meta["stuck"] = True
                                    planner_meta["movement_mode"] = "stuck_unreachable_goal"
                    else:
                        to_x, to_y = self._attempt_move(
                            town=runtime,
                            x=from_x,
                            y=from_y,
                            dx=dx,
                            dy=dy,
                        )
                        if autopilot_enabled and to_x == from_x and to_y == from_y and (dx != 0 or dy != 0):
                            fallback_dx, fallback_dy = self._step_direction(agent_id=agent_id, step=runtime.step)
                            to_x, to_y = self._attempt_move(
                                town=runtime,
                                x=from_x,
                                y=from_y,
                                dx=fallback_dx,
                                dy=fallback_dy,
                            )
                            planner_meta["movement_mode"] = "planner_delta_with_fallback"
                        elif has_external_action:
                            planner_meta["movement_mode"] = "external_delta"
                            if external_action is not None:
                                if to_x == from_x and to_y == from_y and (dx != 0 or dy != 0):
                                    try:
                                        blocked_before = int(external_action.get("_blocked_steps", 0))
                                    except Exception:
                                        blocked_before = 0
                                    blocked_after = max(0, blocked_before) + 1
                                    external_action["_blocked_steps"] = blocked_after
                                    planner_meta["blocked_steps"] = blocked_after
                                    if blocked_after >= 3:
                                        action_force_finish = True
                                        planner_meta["route"] = "external_action_stuck"
                                        planner_meta["reason"] = "external delta movement remained blocked"
                                        planner_meta["stuck"] = True
                                        planner_meta["movement_mode"] = "stuck_blocked_delta"
                                    else:
                                        planner_meta["movement_mode"] = "external_delta_blocked"
                                else:
                                    external_action["_blocked_steps"] = 0
                                    external_action.pop("_reroute_target_x", None)
                                    external_action.pop("_reroute_target_y", None)
                        else:
                            planner_meta["movement_mode"] = "idle_wait"

                npc.x = to_x
                npc.y = to_y
                npc.goal_x = goal_x
                npc.goal_y = goal_y
                npc.goal_reason = goal_reason
                npc.last_step = runtime.step
                npc.status = "moving" if (from_x != to_x or from_y != to_y) else "idle"
                self._update_location_memory(town=runtime, npc=npc, step=runtime.step)

                if external_action_description:
                    action_description = external_action_description
                else:
                    schedule_item = npc.memory.scratch.current_schedule_item(
                        step=runtime.step,
                        step_minutes=max(1, int(runtime.step_minutes)),
                    )
                    if autopilot_enabled and schedule_item is not None:
                        action_description = npc.memory.planned_action_description(
                            step=runtime.step,
                            scope=planning_scope,
                            fallback=goal_reason,
                        )
                    elif pause_for_conversation and conversation_session is not None:
                        partner_id = conversation_session.partner_for(npc.agent_id)
                        partner = runtime.npcs.get(partner_id) if partner_id else None
                        if partner is not None:
                            action_description = f"talking_with_{partner.name}"
                        else:
                            action_description = "in_conversation"
                    elif pause_for_wait:
                        action_description = goal_reason or "waiting_for_social_opening"
                    elif social_interrupted:
                        action_description = "waiting_for_social_decision"
                    elif has_external_action:
                        action_description = goal_reason or "continuing_external_action"
                    elif goal_reason:
                        action_description = str(goal_reason)
                    elif mode == "external":
                        action_description = "awaiting_user_agent"
                    else:
                        action_description = "roaming"
                action_address = npc.current_location
                npc.memory.set_action(
                    step=runtime.step,
                    description=action_description,
                    address=action_address,
                    duration=10,
                    planned_path=[(goal_x, goal_y)] if goal_x is not None and goal_y is not None else [],
                )

                if from_x != to_x or from_y != to_y:
                    npc.memory.add_node(
                        kind="event",
                        step=runtime.step,
                        subject=npc.name,
                        predicate="moves_to",
                        object=f"{to_x},{to_y}",
                        description=f"{npc.name} moves from ({from_x},{from_y}) to ({to_x},{to_y}).",
                        poignancy=1,
                        evidence_ids=(),
                        decrement_trigger=True,
                    )

                if use_external_action:
                    self._ingest_memory_nodes_from_action(
                        npc=npc,
                        action=submitted_action,
                        step=runtime.step,
                    )

                if has_external_action and external_action is not None:
                    step_external_social_events.extend(
                        self._build_social_events_from_action(
                            town=runtime,
                            npc=npc,
                            action=external_action,
                            step=runtime.step,
                        )
                    )

                if has_external_action:
                    try:
                        remaining_before = int((active_action or {}).get("_remaining_steps", 1))
                    except Exception:
                        remaining_before = 1
                    remaining_before = max(1, remaining_before)
                    action_payload = active_action or external_action or {}
                    pending_social_items = self._pending_social_items(action=action_payload)
                    pending_social_count = sum(1 for item in pending_social_items if not bool(item.get("sent") or False))
                    pending_socials = pending_social_count > 0
                    has_social_payload = bool(action_payload.get("socials"))
                    has_explicit_target = (
                        action_payload.get("target_x") is not None and action_payload.get("target_y") is not None
                    )
                    ttl_explicit = bool(action_payload.get("_action_ttl_explicit"))
                    try:
                        action_dx = int(action_payload.get("dx", 0))
                    except Exception:
                        action_dx = 0
                    try:
                        action_dy = int(action_payload.get("dy", 0))
                    except Exception:
                        action_dy = 0
                    has_movement_delta = action_dx != 0 or action_dy != 0
                    countdown_paused = pause_for_conversation or social_interrupted
                    if action_force_finish:
                        done = True
                    elif countdown_paused:
                        done = False
                    else:
                        done_without_social = self._active_action_done(
                            active_action=active_action,
                            current_position=(to_x, to_y),
                        )
                        if pending_socials:
                            if remaining_before <= 1:
                                done = True
                                planner_meta["action_social_timeout"] = True
                                if not planner_meta.get("reason"):
                                    planner_meta["reason"] = (
                                        "external action expired before pending social interactions completed"
                                    )
                            else:
                                done = False
                                planner_meta["action_waiting_for_social"] = True
                        else:
                            if (
                                not done_without_social
                                and has_social_payload
                                and not has_explicit_target
                                and not has_movement_delta
                                and not ttl_explicit
                            ):
                                done_without_social = True
                                planner_meta["action_socials_completed"] = True
                            done = done_without_social
                    if done:
                        remaining_after = 0
                    elif countdown_paused:
                        remaining_after = remaining_before
                    else:
                        remaining_after = max(0, remaining_before - 1)
                    if done:
                        runtime.active_actions.pop(agent_id, None)
                    else:
                        if active_action is None:
                            active_action = dict(external_action or {})
                        active_action["_remaining_steps"] = remaining_after
                        runtime.active_actions[agent_id] = active_action
                    planner_meta["action_remaining_steps_before"] = remaining_before
                    planner_meta["action_remaining_steps_after"] = remaining_after
                    planner_meta["action_finished"] = bool(done)
                    planner_meta["action_paused"] = bool(countdown_paused and not done)
                    planner_meta["action_stuck"] = bool(action_force_finish)
                    planner_meta["pending_social_count"] = pending_social_count

                if delegated_autopilot_enabled and autopilot_enabled and not has_external_action:
                    runtime.autonomy_tick_counts[agent_id] = max(0, int(autonomy_ticks_used)) + 1
                elif has_external_action:
                    runtime.autonomy_tick_counts[agent_id] = 0
                planner_meta["autonomy_ticks_after"] = int(runtime.autonomy_tick_counts.get(agent_id, 0))

                events.append(
                    {
                        "step": runtime.step,
                        "agent_id": npc.agent_id,
                        "name": npc.name,
                        "from": {"x": from_x, "y": from_y},
                        "to": {"x": to_x, "y": to_y},
                        "action": npc.status,
                        "decision": planner_meta,
                        "location": npc.current_location,
                        "goal": {"x": npc.goal_x, "y": npc.goal_y, "reason": npc.goal_reason},
                    }
                )

            step_social_events: list[dict[str, Any]] = []
            if mode in {"autopilot", "hybrid"}:
                step_social_events.extend(self._build_social_events(town=runtime, step=runtime.step))
            if step_external_social_events:
                step_social_events.extend(step_external_social_events)
            if mode in {"autopilot", "hybrid"}:
                step_social_events.extend(self._build_conversation_continuation_events(town=runtime, step=runtime.step))
            if len(step_social_events) > self.max_social_events_per_step:
                step_social_events = step_social_events[: self.max_social_events_per_step]

            if step_social_events:
                social_events.extend(step_social_events)
                self._ingest_social_events(town=runtime, social_events=step_social_events, step=runtime.step)
                self._ingest_conversations_from_events(town=runtime, social_events=step_social_events, step=runtime.step)

            if mode in {"autopilot", "hybrid"}:
                step_reflections = self._run_reflection_cycle(town=runtime, step=runtime.step)
                reflection_events.extend(step_reflections)
            else:
                # Conversation follow-ups must fire in all modes (including external)
                for npc in sorted(runtime.npcs.values(), key=lambda n: n.agent_id):
                    npc.memory._conversation_follow_up_thought(step=runtime.step)

        runtime.recent_events.extend(events)
        runtime.recent_events.extend(social_events)
        runtime.recent_events.extend(reflection_events)
        if len(runtime.recent_events) > self.max_recent_events:
            runtime.recent_events = runtime.recent_events[-self.max_recent_events :]
        runtime.updated_at = _utc_now()
        self._persist_runtime(runtime)

        return {
            "town_id": town_id,
            "planning_scope": planning_scope,
            "control_mode": mode,
            "steps_run": clamped_steps,
            "step": runtime.step,
            "clock": self._clock_for_runtime(town=runtime),
            "event_count": len(events),
            "events": events,
            "social_event_count": len(social_events),
            "social_events": social_events,
            "reflection_event_count": len(reflection_events),
            "reflection_events": reflection_events,
            "state": runtime.to_dict(max_agents=self.max_agents_per_town),
        }


_RUNNER = SimulationRunner()
_RUNNER_LOCK = threading.RLock()


def reset_simulation_states_for_tests(*, clear_persisted: bool = True) -> None:
    with _RUNNER_LOCK:
        _RUNNER.reset(clear_persisted=clear_persisted)


def get_simulation_state(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    with _RUNNER_LOCK:
        return _RUNNER.get_state(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )


def get_agent_memory_snapshot(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    limit: int = 40,
) -> dict[str, Any] | None:
    with _RUNNER_LOCK:
        return _RUNNER.get_agent_memory(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            limit=limit,
        )


def get_agent_context_snapshot(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    planning_scope: str = "short_action",
) -> dict[str, Any] | None:
    with _RUNNER_LOCK:
        return _RUNNER.get_agent_context(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            planning_scope=planning_scope,
        )


def submit_agent_action(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    with _RUNNER_LOCK:
        return _RUNNER.submit_agent_action(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            action=action,
        )


def tick_simulation(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    steps: int = 1,
    planning_scope: str = "short_action",
    control_mode: str = "external",
    planner: Optional[PlannerFn] = None,
    agent_autonomy: dict[str, dict[str, Any]] | None = None,
    cognition_planner: Any = None,
) -> dict[str, Any]:
    with _RUNNER_LOCK:
        return _RUNNER.tick(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            steps=steps,
            planning_scope=planning_scope,
            control_mode=control_mode,
            planner=planner,
            agent_autonomy=agent_autonomy,
            cognition_planner=cognition_planner,
        )
