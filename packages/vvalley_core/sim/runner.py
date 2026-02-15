"""In-memory simulation runtime with memory-aware cognition for V-Valley towns."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random as _random_mod
import re
import threading
from typing import Any, Callable, Optional

from packages.vvalley_core.maps.map_utils import collision_grid, parse_location_objects, parse_spawns
from packages.vvalley_core.sim.memory import AgentMemory

logger = logging.getLogger("vvalley_core.sim.runner")


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
    energy: float = 80.0
    satiety: float = 70.0
    mood: float = 60.0

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
            "pronunciatio": self.memory.scratch.act_pronunciatio,
            "act_event": list(self.memory.scratch.act_event) if self.memory.scratch.act_event else None,
            "physiology": {
                "energy": round(float(self.energy), 2),
                "satiety": round(float(self.satiety), 2),
                "mood": round(float(self.mood), 2),
            },
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
    spatial_metadata: Any = None  # TileSpatialMetadata | None — GA spatial grounding
    object_states: dict[str, str] = field(default_factory=dict)  # "sector:arena:object" → state description

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

    @staticmethod
    def _gravity_score(attractiveness: float, distance: int, beta: float = 1.5) -> float:
        """Gravity model: score = attractiveness / distance^beta."""
        return attractiveness / max(1.0, float(distance)) ** beta

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

        reason = schedule_item.description if schedule_item else normalized_scope
        return self._choose_location_goal_for_affordance(
            town=town,
            npc=npc,
            affordance=target_affordance,
            reason=str(reason),
            long_term=(normalized_scope == "long_term_plan"),
        )

    def _choose_location_goal_for_affordance(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        affordance: str | None,
        reason: str,
        long_term: bool = False,
        exclude: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int, str, str | None] | None:
        candidates = self._locations_by_affordance(town=town, affordance=affordance)
        if not candidates:
            return None
        excluded = exclude or set()
        if excluded:
            filtered = [loc for loc in candidates if (int(loc.x), int(loc.y)) not in excluded]
            if filtered:
                candidates = filtered
        rng = _random_mod.Random(f"{npc.agent_id}:{town.step}:{affordance or 'any'}:{reason}")
        weights: list[float] = []
        for loc in candidates:
            label = loc.label()
            visits = int(npc.memory.known_places.get(label, 0))
            dist = self._manhattan(npc.x, npc.y, loc.x, loc.y)
            if long_term:
                attractiveness = max(0.5, float(visits + 1))
            else:
                attractiveness = max(0.5, 10.0 - visits)
            weights.append(self._gravity_score(attractiveness, dist))
        total = sum(weights)
        if total <= 0:
            chosen = candidates[0]
        else:
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
        final_x, final_y = self._avoid_occupied_tile(
            town=town,
            npc=npc,
            target=(chosen.x, chosen.y),
        )
        return (final_x, final_y, str(reason), affordance)

    # ── Needs-based autopilot pre-filter ──────────────────────

    _ROUTINE_AFFORDANCES: frozenset[str] = frozenset({"home", "food", "work"})

    def _needs_based_prefilter(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        scope: str,
        nearby_agents: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Short-circuit routine autopilot decisions that don't need LLM.

        Returns a pre-built action dict when the current schedule affordance
        is physiological/safety (home, food, work) and no social opportunity
        is present.  Returns *None* when LLM planning is needed.
        """
        schedule_item = npc.memory.scratch.current_schedule_item(
            step=town.step,
            step_minutes=max(1, int(town.step_minutes)),
        )
        affordance = getattr(schedule_item, "affordance_hint", None) if schedule_item else None
        if affordance not in self._ROUTINE_AFFORDANCES:
            return None
        if nearby_agents:
            return None

        location_goal = self._choose_location_goal(town=town, npc=npc, scope=scope)
        if location_goal is None:
            return None

        goal_x, goal_y, description, goal_affordance = location_goal
        return {
            "target_x": goal_x,
            "target_y": goal_y,
            "description": description,
            "affordance": goal_affordance,
            "prefilter": True,
        }

    @staticmethod
    def _clamp_pct(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    def _personal_urgency(self, *, npc: NpcState) -> float:
        energy_urgency = max(0.0, (40.0 - float(npc.energy)) / 40.0)
        satiety_urgency = max(0.0, (30.0 - float(npc.satiety)) / 30.0)
        mood_urgency = max(0.0, (25.0 - float(npc.mood)) / 25.0)
        return max(energy_urgency, satiety_urgency, mood_urgency)

    def _objective_bias(self, *, objective: str | None, branch: str) -> float:
        text = str(objective or "").strip().lower()
        if not text:
            return 0.0
        if branch == "social" and any(key in text for key in ("social", "friend", "community", "relationship")):
            return 0.6
        if branch == "routine" and any(key in text for key in ("work", "job", "routine", "career", "responsibility")):
            return 0.5
        if branch == "exploration" and any(key in text for key in ("explore", "discover", "learn", "study")):
            return 0.45
        if branch == "personal" and any(key in text for key in ("rest", "health", "recover", "wellbeing")):
            return 0.6
        return 0.0

    def _prioritize_branches(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        step: int,
        nearby_agents: list[dict[str, Any]],
        schedule_item: Any,
        cognition_planner: Any = None,
    ) -> tuple[str, str, str | None, dict[str, float]]:
        """Score branch candidates and return best branch + branch reason + affordance."""
        npc.memory.scratch.ensure_default_branches()
        branch_weight_map = {
            branch.name: max(0.1, float(branch.priority_weight))
            for branch in npc.memory.scratch.planning_branches
        }
        has_nearby = bool(nearby_agents)
        step_minutes = max(1, int(town.step_minutes))
        social_recency_steps = max(1, int(step) - int(npc.memory.scratch.last_social_step))
        social_recency_minutes = social_recency_steps * step_minutes
        exploration_threshold_steps = max(1, int(150 / step_minutes))
        routine_affordance = schedule_item.affordance_hint if schedule_item is not None else None
        routine_reason = str(schedule_item.description) if schedule_item is not None else "routine"

        scores: dict[str, float] = {
            "personal": 0.2 + self._personal_urgency(npc=npc),
            "social": (0.6 if has_nearby else 0.1) * min(float(social_recency_minutes) / 200.0, 1.5),
            "routine": 0.5,
            "exploration": 0.3
            * (
                1.0
                if int(npc.memory.scratch.steps_at_current_location) > exploration_threshold_steps
                else 0.2
            ),
        }
        for branch_name in list(scores.keys()):
            scores[branch_name] *= float(branch_weight_map.get(branch_name, 1.0))
            scores[branch_name] += self._objective_bias(
                objective=npc.memory.scratch.long_term_objective,
                branch=branch_name,
            )

        selected_branch = max(scores, key=scores.get)
        selected_reason = "routine"
        selected_affordance: str | None = routine_affordance

        # Optional LLM enrichment for branch choice.
        if cognition_planner is not None and hasattr(cognition_planner, "contextual_prioritize"):
            try:
                context = {
                    "town_id": town.town_id,
                    "step": step,
                    "agent_id": npc.agent_id,
                    "agent_name": npc.name,
                    "identity": npc.memory.scratch.get_str_iss(),
                    "position": {"x": npc.x, "y": npc.y},
                    "current_location": npc.current_location,
                    "schedule": {
                        "description": routine_reason,
                        "affordance_hint": routine_affordance,
                    },
                    "nearby_agents": nearby_agents[:4],
                    "branch_scores": scores,
                    "recent_memories": [
                        str(item.get("description") or "")
                        for item in npc.memory.retrieve_ranked(
                            focal_text="branch planning",
                            step=step,
                            limit=6,
                            retrieval_mode="long_term",
                        )
                    ],
                }
                result = cognition_planner.contextual_prioritize(context)
                branch_candidate = str(result.get("branch") or "").strip().lower()
                if result.get("route") != "heuristic" and branch_candidate in scores:
                    selected_branch = branch_candidate
                    selected_reason = str(result.get("reason") or selected_branch)
            except Exception:
                pass

        if not selected_reason or selected_reason == "routine":
            if selected_branch == "personal":
                if float(npc.satiety) < 30.0:
                    selected_reason = "restore_satiety"
                    selected_affordance = "food"
                elif float(npc.energy) < 40.0:
                    selected_reason = "recover_energy"
                    selected_affordance = "home"
                elif float(npc.mood) < 25.0:
                    selected_reason = "recover_mood"
                    selected_affordance = "leisure"
                else:
                    selected_reason = "maintain_wellbeing"
                    selected_affordance = "home"
            elif selected_branch == "social":
                selected_reason = "socialize_with_neighbors" if has_nearby else "seek_social_contact"
                selected_affordance = "social"
            elif selected_branch == "exploration":
                selected_reason = "explore_new_places"
                selected_affordance = "leisure"
            else:
                selected_reason = routine_reason
                selected_affordance = routine_affordance

        npc.memory.scratch.active_branch = selected_branch
        for branch in npc.memory.scratch.planning_branches:
            if branch.name == selected_branch:
                branch.last_active_step = int(step)
                break
        if selected_branch != "routine":
            npc.memory.scratch.schedule_override_reason = selected_reason
        else:
            npc.memory.scratch.schedule_override_reason = None

        return (selected_branch, selected_reason, selected_affordance, scores)

    @staticmethod
    def _extract_affordance_from_reason(reason: str | None) -> str | None:
        value = str(reason or "").strip().lower()
        if not value:
            return None
        if any(token in value for token in ("social", "talk", "chat")):
            return "social"
        if any(token in value for token in ("eat", "food", "lunch", "dinner", "satiety")):
            return "food"
        if any(token in value for token in ("work", "task", "job", "career")):
            return "work"
        if any(token in value for token in ("home", "sleep", "rest", "energy")):
            return "home"
        if any(token in value for token in ("leisure", "relax", "explore", "mood")):
            return "leisure"
        return None

    def _agents_near(
        self,
        *,
        town: TownRuntime,
        x: int,
        y: int,
        radius: int,
        exclude_agent_id: str | None = None,
    ) -> list[NpcState]:
        out: list[NpcState] = []
        for other in town.npcs.values():
            if exclude_agent_id and other.agent_id == exclude_agent_id:
                continue
            if self._manhattan(x, y, other.x, other.y) <= max(0, int(radius)):
                out.append(other)
        return out

    def _is_reachable(
        self,
        *,
        town: TownRuntime,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> bool:
        sx, sy = start
        tx, ty = target
        if (sx, sy) == (tx, ty):
            return True
        if tx < 0 or ty < 0 or tx >= town.width or ty >= town.height:
            return False
        if not town.walkable[ty][tx]:
            return False
        queue: deque[tuple[int, int]] = deque()
        queue.append((sx, sy))
        seen: set[tuple[int, int]] = {(sx, sy)}
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in self._neighbors4(town=town, x=cx, y=cy):
                if (nx, ny) in seen:
                    continue
                if (nx, ny) == (tx, ty):
                    return True
                seen.add((nx, ny))
                queue.append((nx, ny))
        return False

    def _find_alternative_goal(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        reason: str | None,
        exclude: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int, str] | None:
        affordance = self._extract_affordance_from_reason(reason)
        if not affordance:
            schedule_item = npc.memory.scratch.current_schedule_item(
                step=town.step,
                step_minutes=max(1, int(town.step_minutes)),
            )
            if schedule_item is not None:
                affordance = schedule_item.affordance_hint
        goal = self._choose_location_goal_for_affordance(
            town=town,
            npc=npc,
            affordance=affordance,
            reason=str(reason or "alternative_goal"),
            long_term=False,
            exclude=exclude,
        )
        if goal is None:
            return None
        return (int(goal[0]), int(goal[1]), f"{goal[2]} (alternative)")

    def _validate_goal(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        goal_x: int,
        goal_y: int,
        reason: str,
        step: int,
    ) -> tuple[int, int, str, list[str]] | None:
        issues: list[str] = []
        if not self._is_reachable(
            town=town,
            start=(npc.x, npc.y),
            target=(int(goal_x), int(goal_y)),
        ):
            issues.append("unreachable")
        recent_locations = list(npc.memory.scratch.recent_goal_locations[-3:])
        if (int(goal_x), int(goal_y)) in recent_locations:
            issues.append("repetitive")
        if "social" in str(reason).lower():
            nearby = self._agents_near(
                town=town,
                x=int(goal_x),
                y=int(goal_y),
                radius=3,
                exclude_agent_id=npc.agent_id,
            )
            if not nearby:
                issues.append("no_social_targets")
        if abs(int(npc.x) - int(goal_x)) <= 1 and abs(int(npc.y) - int(goal_y)) <= 1:
            issues.append("already_there")

        if not issues:
            return (int(goal_x), int(goal_y), str(reason), [])

        alternative = self._find_alternative_goal(
            town=town,
            npc=npc,
            reason=reason,
            exclude={(int(goal_x), int(goal_y))},
        )
        if alternative is not None:
            return (alternative[0], alternative[1], alternative[2], issues)
        return None

    def _walkable_tiles_near(
        self,
        *,
        town: TownRuntime,
        x: int,
        y: int,
        radius: int,
    ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for ny in range(max(0, y - radius), min(town.height, y + radius + 1)):
            for nx in range(max(0, x - radius), min(town.width, x + radius + 1)):
                if not town.walkable[ny][nx]:
                    continue
                if self._manhattan(x, y, nx, ny) > radius:
                    continue
                out.append((nx, ny))
        return out

    def _location_from_description(self, *, town: TownRuntime, description: str) -> tuple[int, int] | None:
        match = re.search(r"\((\d+),(\d+)\)", str(description))
        if match:
            try:
                return (int(match.group(1)), int(match.group(2)))
            except Exception:
                return None
        lowered = str(description).lower()
        for location in town.location_points:
            if str(location.name).lower() in lowered:
                return (int(location.x), int(location.y))
        return None

    def _recover_from_failure(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        step: int,
        failed_reason: str,
        current_affordance: str | None = None,
    ) -> tuple[int, int, str] | None:
        recent_success_nodes = [
            npc.memory.id_to_node.get(node_id)
            for node_id in npc.memory.seq_event[-10:]
        ]
        for node in reversed([item for item in recent_success_nodes if item is not None]):
            if "moves from" not in str(node.description).lower():
                continue
            loc = self._location_from_description(town=town, description=node.description)
            if loc is None:
                continue
            validated = self._validate_goal(
                town=town,
                npc=npc,
                goal_x=loc[0],
                goal_y=loc[1],
                reason="retry from memory",
                step=step,
            )
            if validated is not None:
                return (validated[0], validated[1], "retry_from_memory")

        affordance_reason = current_affordance or failed_reason
        alternative = self._find_alternative_goal(
            town=town,
            npc=npc,
            reason=affordance_reason,
            exclude={(npc.goal_x, npc.goal_y)} if npc.goal_x is not None and npc.goal_y is not None else None,
        )
        if alternative is not None:
            return alternative

        nearby_walkable = self._walkable_tiles_near(
            town=town,
            x=npc.x,
            y=npc.y,
            radius=8,
        )
        if nearby_walkable:
            rng = _random_mod.Random(f"recover:{npc.agent_id}:{step}")
            tx, ty = rng.choice(nearby_walkable)
            return (int(tx), int(ty), "exploring_nearby")
        return None

    def _apply_physiology_tick(
        self,
        *,
        npc: NpcState,
        moved: bool,
        action_description: str | None,
    ) -> None:
        desc = str(action_description or "").lower()
        energy = float(npc.energy)
        satiety = float(npc.satiety)
        mood = float(npc.mood)

        satiety -= 0.3
        mood -= 0.1
        if moved:
            energy -= 0.3
            satiety -= 0.1
        if any(token in desc for token in ("chat", "talk", "social", "conversation")):
            energy -= 0.1
            satiety -= 0.1
            mood += 2.0
        elif any(token in desc for token in ("work", "task", "study", "job")):
            energy -= 1.0
            satiety -= 0.5
            mood -= 0.2
        elif any(token in desc for token in ("eat", "lunch", "dinner", "food", "cafe", "pub")):
            energy += 0.5
            satiety += 8.0
            mood += 0.5
        elif any(token in desc for token in ("sleep", "bed", "resting")):
            energy += 3.0
            satiety -= 0.2
            mood += 0.5
        elif any(token in desc for token in ("leisure", "relax", "park", "read", "paint", "play")):
            energy -= 0.2
            satiety -= 0.1
            mood += 1.5
        elif "idle" in desc or "wait" in desc:
            energy += 1.0
            satiety -= 0.1

        npc.energy = self._clamp_pct(energy)
        npc.satiety = self._clamp_pct(satiety)
        npc.mood = self._clamp_pct(mood)

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

        # Resolve spatial context for conversation location
        _conv_sector: str | None = None
        _conv_arena: str | None = None
        if town.spatial_metadata is not None:
            _conv_sector = town.spatial_metadata.sector_at(a.x, a.y)
            _conv_arena = town.spatial_metadata.arena_at(a.x, a.y)

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
                "speaking_agent_identity": current.memory.scratch.get_str_iss(),
                "partner_name": partner.name,
                "partner_id": partner.agent_id,
                "partner_identity": partner.memory.scratch.get_str_iss(),
                "speaking_agent_activity": current.memory.scratch.act_description or "their routine",
                "partner_activity": partner.memory.scratch.act_description or "their routine",
                "speaking_agent_memories": memories,
                "relationship_score": round(
                    current.memory.relationship_scores.get(partner.agent_id, 0.0), 2
                ),
                "transcript_so_far": transcript_so_far,
                "turn_number": turn,
                "time_of_day": str(clock.get("phase", "day")),
                "location_sector": _conv_sector,
                "location_arena": _conv_arena,
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

            # Resolve spatial context for conversation location
            _ss_sector: str | None = None
            _ss_arena: str | None = None
            if town.spatial_metadata is not None:
                _ss_sector = town.spatial_metadata.sector_at(a.x, a.y)
                _ss_arena = town.spatial_metadata.arena_at(a.x, a.y)

            result = cognition.generate_conversation({
                "agent_a_name": a.name,
                "agent_b_name": b.name,
                "agent_a_identity": a.memory.scratch.get_str_iss(),
                "agent_b_identity": b.memory.scratch.get_str_iss(),
                "agent_a_activity": a.memory.scratch.act_description or "their routine",
                "agent_b_activity": b.memory.scratch.act_description or "their routine",
                "agent_a_memories": a_memories,
                "agent_b_memories": b_memories,
                "relationship_score": round((a_relationship + b_relationship) / 2.0, 2),
                "time_of_day": str(clock.get("phase", "day")),
                "location_sector": _ss_sector,
                "location_arena": _ss_arena,
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
                # Recompose schedules for both participants after chat reaction
                chat_duration = max(10, len(transcript) * 5) if transcript else 15
                for participant in (a, b):
                    part_cognition = getattr(participant.memory, "cognition", None)
                    if part_cognition is not None:
                        self._recompose_schedule_on_reaction(
                            town=town, npc=participant,
                            inserted_activity=f"chatting with {a.name if participant is b else b.name}",
                            inserted_duration_mins=chat_duration,
                            cognition=part_cognition,
                        )
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

        # Build GA spatial metadata if available
        spatial_meta = None
        try:
            from packages.vvalley_core.maps.map_utils import build_spatial_metadata
            spatial_meta = build_spatial_metadata(map_data)
        except Exception:
            pass

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
            spatial_metadata=spatial_meta,
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
            "energy": float(npc.energy),
            "satiety": float(npc.satiety),
            "mood": float(npc.mood),
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
        memory.scratch.ensure_default_branches()
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
        try:
            energy = float(raw.get("energy") if raw.get("energy") is not None else 80.0)
        except Exception:
            energy = 80.0
        try:
            satiety = float(raw.get("satiety") if raw.get("satiety") is not None else 70.0)
        except Exception:
            satiety = 70.0
        try:
            mood = float(raw.get("mood") if raw.get("mood") is not None else 60.0)
        except Exception:
            mood = 60.0
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
            energy=max(0.0, min(100.0, energy)),
            satiety=max(0.0, min(100.0, satiety)),
            mood=max(0.0, min(100.0, mood)),
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
        except Exception as exc:
            logger.error(
                "[RUNNER] Failed to persist runtime state for town '%s': %s",
                runtime.town_id,
                exc,
            )
            self._persist_error_count = getattr(self, "_persist_error_count", 0) + 1

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
        spatial = town.spatial_metadata
        npc_arena = spatial.arena_at(npc.x, npc.y) if spatial else None
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            dist = self._manhattan(npc.x, npc.y, other.x, other.y)
            if dist <= npc.memory.vision_radius:
                # Arena-boundary filter: skip agents in a different arena
                if spatial is not None and npc_arena is not None:
                    other_arena = spatial.arena_at(other.x, other.y)
                    if other_arena is not None and other_arena != npc_arena:
                        continue
                candidates.append((dist, other))

        candidates.sort(key=lambda item: (item[0], item[1].agent_id))
        seen_signatures = set(npc.memory.recent_signatures())
        nearby: list[dict[str, Any]] = []
        new_events: list[str] = []

        for dist, other in candidates[: npc.memory.attention_bandwidth]:
            # Use event triple from pronunciatio if available
            evt = other.memory.scratch.act_event if other.memory.scratch.act_event else None
            if evt and len(evt) == 3:
                subj, pred, obj = evt
            else:
                subj, pred, obj = other.name, "is", other.status

            nearby.append(
                {
                    "agent_id": other.agent_id,
                    "name": other.name,
                    "status": other.status,
                    "distance": dist,
                    "x": other.x,
                    "y": other.y,
                    "event_triple": [subj, pred, obj],
                    "pronunciatio": other.memory.scratch.act_pronunciatio,
                }
            )
            signature = (subj, pred, obj)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            desc = f"{subj} {pred} {obj} near ({other.x},{other.y})."
            poignancy = npc.memory.score_event_poignancy(description=desc, event_type="perception")
            npc.memory.add_node(
                kind="event",
                step=step,
                subject=subj,
                predicate=pred,
                object=obj,
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

    def _perceive_spatial_tiles(self, *, town: TownRuntime, npc: NpcState, step: int) -> None:
        """Populate the agent's spatial memory tree from nearby tiles with spatial metadata."""
        spatial = town.spatial_metadata
        if spatial is None:
            return
        radius = npc.memory.vision_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                tx, ty = npc.x + dx, npc.y + dy
                sector = spatial.sector_at(tx, ty)
                if sector is None:
                    continue
                arena = spatial.arena_at(tx, ty) or ""
                game_obj = spatial.object_at(tx, ty) or ""
                npc.memory.spatial.observe(world="the Ville", sector=sector, arena=arena, game_object=game_obj)

    def _avoid_occupied_tile(
        self, *, town: TownRuntime, npc: NpcState, target: tuple[int, int]
    ) -> tuple[int, int]:
        """If the target tile is occupied by another agent, find a nearby unoccupied walkable tile."""
        tx, ty = target
        # Check if any other agent is on the target tile
        occupied = False
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            if other.x == tx and other.y == ty:
                occupied = True
                break
        if not occupied:
            return target
        # Search in a small radius for an unoccupied walkable tile
        for radius in range(1, 4):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = tx + dx, ty + dy
                    if ny < 0 or ny >= len(town.walkable) or nx < 0 or nx >= len(town.walkable[0]):
                        continue
                    if town.walkable[ny][nx] != 1:
                        continue
                    # Check not occupied by another agent
                    tile_occupied = False
                    for other in town.npcs.values():
                        if other.agent_id == npc.agent_id:
                            continue
                        if other.x == nx and other.y == ny:
                            tile_occupied = True
                            break
                    if not tile_occupied:
                        return (nx, ny)
        # Couldn't find unoccupied tile, return original
        return target

    def _resolve_action_address(
        self, *, town: TownRuntime, npc: NpcState, action_desc: str, cognition: Any
    ) -> tuple[int, int] | None:
        """3-step action address resolution: pick sector→arena→object, return target tile."""
        spatial = town.spatial_metadata
        if spatial is None:
            return None
        if cognition is None or not hasattr(cognition, "resolve_action_address"):
            return None

        sectors = spatial.sectors_list()
        if not sectors:
            return None

        try:
            # Build arena/object position lookups for heuristic fallback
            arena_positions: dict[str, tuple[int, int]] = {}
            object_positions: dict[str, tuple[int, int]] = {}
            for sec in sectors:
                for arena_name in spatial.arenas_in_sector(sec):
                    tile = spatial.find_tile_for_arena(sec, arena_name) if hasattr(spatial, "find_tile_for_arena") else None
                    if tile:
                        arena_positions[arena_name] = tile
                    for obj_name in spatial.objects_in_arena(sec, arena_name):
                        obj_tile = spatial.find_tile_for_object(sec, arena_name, obj_name)
                        if obj_tile:
                            object_positions[obj_name] = obj_tile

            result = cognition.resolve_action_address({
                "agent_id": npc.agent_id,
                "agent_name": npc.name,
                "town_id": town.town_id,
                "action_description": action_desc,
                "identity": npc.memory.scratch.get_str_iss(),
                "living_area": npc.memory.scratch.living_area or "",
                "available_sectors": sectors,
                "spatial_memory": list(npc.memory.spatial.tree.keys())[:10],
                "agent_x": npc.x,
                "agent_y": npc.y,
                "arena_positions": arena_positions,
                "object_positions": object_positions,
            })

            target_sector = str(result.get("action_sector") or "")
            if not target_sector or target_sector not in sectors:
                return None

            # Step 2: get arenas in the chosen sector
            arenas = spatial.arenas_in_sector(target_sector)
            target_arena = str(result.get("action_arena") or "")
            if target_arena and target_arena not in arenas and arenas:
                target_arena = arenas[0]
            elif not target_arena and arenas:
                target_arena = arenas[0]

            # Step 3: get objects in the chosen arena
            if target_arena:
                objects = spatial.objects_in_arena(target_sector, target_arena)
                target_obj = str(result.get("action_game_object") or "")
                if target_obj and target_obj in objects:
                    tile = spatial.find_tile_for_object(target_sector, target_arena, target_obj)
                    if tile:
                        npc.memory.scratch.act_address = f"{target_sector}:{target_arena}:{target_obj}"
                        return self._avoid_occupied_tile(town=town, npc=npc, target=tile)

            # Fallback: find any walkable tile in the target sector
            for y in range(spatial.height):
                for x in range(spatial.width):
                    if spatial.sector_at(x, y) == target_sector:
                        if 0 <= y < len(town.walkable) and 0 <= x < len(town.walkable[0]):
                            if town.walkable[y][x] == 1:
                                npc.memory.scratch.act_address = target_sector
                                return self._avoid_occupied_tile(town=town, npc=npc, target=(x, y))
        except Exception:
            pass
        return None

    def _update_object_state(
        self, *, town: TownRuntime, npc: NpcState, cognition: Any
    ) -> None:
        """Update game object state when agent is on an object tile."""
        spatial = town.spatial_metadata
        if spatial is None:
            return
        obj = spatial.object_at(npc.x, npc.y)
        if obj is None:
            return
        sector = spatial.sector_at(npc.x, npc.y) or ""
        arena = spatial.arena_at(npc.x, npc.y) or ""
        obj_key = f"{sector}:{arena}:{obj}"
        action_desc = npc.status or "idle"

        if cognition is not None and hasattr(cognition, "generate_object_state"):
            try:
                result = cognition.generate_object_state({
                    "agent_id": npc.agent_id,
                    "agent_name": npc.name,
                    "town_id": town.town_id,
                    "action_description": action_desc,
                    "game_object": obj,
                    "sector": sector,
                    "arena": arena,
                })
                desc = str(result.get("object_description") or "").strip()
                if desc and result.get("route") != "heuristic":
                    town.object_states[obj_key] = desc
                    return
            except Exception:
                pass

        # Heuristic fallback
        town.object_states[obj_key] = f"{obj} is being used by {npc.name}"

    def _choose_perceived_event(
        self, *, npc: NpcState, events: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """GA-style event prioritization: pick the most relevant perceived event to react to.

        Prioritize person events, skip self-events, skip idle.
        """
        if not events:
            return None
        # Filter out self-events and idle
        candidates = []
        for evt in events:
            subj = str(evt.get("name") or "")
            if subj == npc.name:
                continue
            desc = str(evt.get("status") or "").lower()
            if "idle" in desc:
                continue
            candidates.append(evt)
        if not candidates:
            return None
        # Prioritize agents (non-object events) over everything else
        person_events = [e for e in candidates if ":" not in str(e.get("name") or "")]
        if person_events:
            return person_events[0]
        return candidates[0]

    def _maybe_react_to_event(
        self, *, town: TownRuntime, npc: NpcState, event: dict[str, Any],
        step: int, cognition: Any
    ) -> str | None:
        """Check if agent should react to a perceived event (non-chat). Returns new action or None."""
        if cognition is None or not hasattr(cognition, "decide_to_react"):
            return None
        try:
            event_desc = f"{event.get('name', 'someone')} is {event.get('status', 'doing something')}"
            result = cognition.decide_to_react({
                "agent_id": npc.agent_id,
                "agent_name": npc.name,
                "town_id": town.town_id,
                "identity": npc.memory.scratch.get_str_iss(),
                "agent_activity": npc.status or "idle",
                "event_description": event_desc,
                "step": step,
            })
            if result.get("route") == "heuristic" and not result.get("react"):
                return None
            if result.get("react"):
                return str(result.get("new_action") or event_desc)
        except Exception:
            pass
        return None

    def _recompose_schedule_on_reaction(
        self,
        *,
        town: TownRuntime,
        npc: NpcState,
        inserted_activity: str,
        inserted_duration_mins: int,
        cognition: Any,
    ) -> bool:
        """Recompose the daily schedule after a reaction/chat interruption (GA _create_react)."""
        if cognition is None or not hasattr(cognition, "recompose_schedule"):
            return False
        schedule = npc.memory.scratch.daily_schedule
        if not schedule:
            return False
        step_minutes = max(1, int(town.step_minutes))
        current_minute = (int(town.step) * step_minutes) % max(1, npc.memory.scratch.day_minutes)
        try:
            from packages.vvalley_core.sim.memory import ScheduleItem
            result = cognition.recompose_schedule({
                "agent_id": npc.agent_id,
                "agent_name": npc.name,
                "town_id": town.town_id,
                "identity": npc.memory.scratch.get_str_iss(),
                "inserted_activity": inserted_activity,
                "inserted_duration_mins": inserted_duration_mins,
                "current_minute_of_day": current_minute,
                "current_schedule": [item.as_dict() for item in schedule],
                "step": town.step,
            })
            new_items = result.get("schedule") or []
            if not new_items:
                return False
            new_schedule = [ScheduleItem.from_dict(item) for item in new_items if isinstance(item, dict)]
            if new_schedule:
                npc.memory.scratch.daily_schedule = new_schedule
                return True
        except Exception:
            pass
        return False

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

        # --- Try LLM-based reaction policy first ---
        cognition = getattr(a.memory, "cognition", None)
        if cognition is not None and hasattr(cognition, "decide_to_talk"):
            try:
                llm_result = cognition.decide_to_talk({
                    "agent_a_id": a.agent_id,
                    "agent_b_id": b.agent_id,
                    "agent_a_name": a.name,
                    "agent_b_name": b.name,
                    "agent_a_identity": a.memory.scratch.get_str_iss(),
                    "agent_b_identity": b.memory.scratch.get_str_iss(),
                    "agent_a_activity": a.status,
                    "agent_b_activity": b.status,
                    "relationship_score": relationship,
                    "step": step,
                    "town_id": town.town_id,
                    "agent_id": a.agent_id,
                })
                if llm_result.get("route") != "heuristic":
                    decision = str(llm_result.get("decision") or "ignore")
                    if decision in ("talk", "wait", "ignore"):
                        result: dict[str, Any] = {
                            "decision": decision,
                            "reason": str(llm_result.get("reason") or "llm_reaction_policy"),
                            "relationship_score": round(relationship, 3),
                            "policy_source": "llm",
                        }
                        if decision == "wait":
                            a_busy = a_remaining <= b_remaining
                            waiter = b if a_busy else a
                            target = a if waiter.agent_id == b.agent_id else b
                            result["waiter_agent_id"] = waiter.agent_id
                            result["target_agent_id"] = target.agent_id
                            result["wait_steps"] = max(1, int(llm_result.get("wait_steps") or (2 if relationship >= 2.0 else 1)))
                        return result
            except Exception:
                pass

        # --- Fallback: hash-based reaction policy ---
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
                "policy_source": "heuristic",
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
                    "policy_source": "heuristic",
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
                "policy_source": "heuristic",
            }

        return {
            "decision": "ignore",
            "reason": "reaction gate rejected this encounter",
            "gate": gate,
            "talk_threshold": talk_threshold,
            "wait_threshold": wait_threshold,
            "relationship_score": round(relationship, 3),
            "policy_source": "heuristic",
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
            npc.memory.scratch.chatting_with_buffer[partner_name] = int(closed_step) + 20
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

    @staticmethod
    def _compute_decision_complexity(
        *,
        schedule_item: Any,
        nearby_agents: list[dict[str, Any]],
        conversation: dict[str, Any] | None,
        active_action: dict[str, Any] | None,
    ) -> str:
        """Classify current situation as 'routine', 'social', or 'complex'."""
        if conversation and conversation.get("awaiting_reply"):
            return "complex"
        if nearby_agents:
            if schedule_item and getattr(schedule_item, "affordance_hint", None) == "social":
                return "social"
            if len(nearby_agents) >= 2:
                return "social"
        return "routine"

    def _context_for_npc(self, *, town: TownRuntime, npc: NpcState, planning_scope: str) -> dict[str, Any]:
        self._update_location_memory(town=town, npc=npc, step=town.step)
        perception = self._perceive_nearby_agents_preview(town=town, npc=npc)
        focal_parts: list[str] = [planning_scope]
        if npc.current_location:
            focal_parts.append(npc.current_location)
        for other in perception["nearby_agents"]:
            focal_parts.append(str(other.get("status") or "idle"))
        focal_text = " ".join(i for i in focal_parts if i)
        retrieval_mode = "long_term" if self._normalize_scope(planning_scope) == "long_term_plan" else "short_term"
        streams = npc.memory.retrieve_ranked_by_stream(
            focal_text=focal_text,
            step=town.step,
            limit=6,
            retrieval_mode=retrieval_mode,
        )
        event_stream = streams["event_stream"]
        perception_stream = streams["perception_stream"]
        retrieved = event_stream + perception_stream

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

        # Spatial context from GA tile metadata
        location_sector: str | None = None
        location_arena: str | None = None
        location_object: str | None = None
        spatial = town.spatial_metadata
        if spatial is not None:
            location_sector = spatial.sector_at(npc.x, npc.y)
            location_arena = spatial.arena_at(npc.x, npc.y)
            location_object = spatial.object_at(npc.x, npc.y)

        # Decision complexity hint for model routing
        conv_ctx = self._conversation_context_for_agent(town=town, npc=npc)
        active_action_data = town.active_actions.get(npc.agent_id)
        decision_complexity = self._compute_decision_complexity(
            schedule_item=schedule_item,
            nearby_agents=perception.get("nearby_agents", []),
            conversation=conv_ctx,
            active_action=active_action_data,
        )

        return {
            "town_id": town.town_id,
            "step": town.step,
            "clock": self._clock_for_runtime(town=town),
            "agent_id": npc.agent_id,
            "agent_name": npc.name,
            "persona": npc.persona,
            "identity": npc.memory.scratch.get_str_iss(),
            "planning_scope": planning_scope,
            "long_term_objective": npc.memory.scratch.long_term_objective,
            "active_branch": npc.memory.scratch.active_branch,
            "position": {"x": npc.x, "y": npc.y},
            "physiology": {
                "energy": round(float(npc.energy), 2),
                "satiety": round(float(npc.satiety), 2),
                "mood": round(float(npc.mood), 2),
            },
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
                "event_stream": event_stream,
                "perception_stream": perception_stream,
            },
            "schedule": schedule,
            "location_sector": location_sector,
            "location_arena": location_arena,
            "location_object": location_object,
            "decision_complexity": decision_complexity,
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
        # Add spatial location context for conversation
        location_sector: str | None = None
        location_arena: str | None = None
        spatial = town.spatial_metadata
        if spatial is not None:
            location_sector = spatial.sector_at(npc.x, npc.y)
            location_arena = spatial.arena_at(npc.x, npc.y)

        # Generate relationship summary for conversation grounding
        relationship_narrative: str | None = None
        cognition = getattr(npc.memory, "cognition", None)
        if cognition is not None and hasattr(cognition, "generate_relationship_summary") and int(session.turns) == 0:
            try:
                rel_score = (
                    self._relationship_score(source=npc, target=partner)
                    + self._relationship_score(source=partner, target=npc)
                ) / 2.0
                past = npc.memory.top_relationships(limit=10)
                past_with_partner = [r for r in past if str(r.get("agent_id", "")) == partner.agent_id]
                rel_result = cognition.generate_relationship_summary({
                    "agent_id": npc.agent_id,
                    "agent_name": npc.name,
                    "town_id": town.town_id,
                    "partner_name": partner.name,
                    "identity": npc.memory.scratch.get_str_iss(),
                    "partner_identity": partner.memory.scratch.get_str_iss(),
                    "relationship_score": rel_score,
                    "past_interactions": past_with_partner,
                })
                if rel_result.get("route") != "heuristic":
                    relationship_narrative = str(rel_result.get("summary") or "").strip() or None
                if relationship_narrative is None:
                    relationship_narrative = str(rel_result.get("summary") or "").strip() or None
            except Exception:
                pass

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
            "location_sector": location_sector,
            "location_arena": location_arena,
            "relationship_summary": relationship_narrative,
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

    def _update_social_attitude(
        self,
        *,
        npc: NpcState,
        partner: NpcState,
        conversation_summary: str,
        step: int,
        cognition_planner: Any = None,
    ) -> tuple[str, float]:
        current_attitude = npc.memory.scratch.social_attitudes.get(partner.name, "neutral")
        summary_lower = str(conversation_summary or "").lower()
        attitude = current_attitude
        relationship_delta = 1.0

        if any(token in summary_lower for token in ("argue", "conflict", "angry", "distrust", "lie", "deceive")):
            attitude = "wary"
            relationship_delta = -1.2
        elif any(token in summary_lower for token in ("warm", "friendly", "supportive", "helpful", "good chat")):
            attitude = "trusting"
            relationship_delta = 1.8
        else:
            attitude = "neutral"
            relationship_delta = 0.6

        if cognition_planner is not None and hasattr(cognition_planner, "assess_social_attitude"):
            try:
                result = cognition_planner.assess_social_attitude(
                    {
                        "town_id": npc.memory.town_id or "",
                        "step": step,
                        "agent_id": npc.agent_id,
                        "agent_name": npc.name,
                        "partner_name": partner.name,
                        "conversation_summary": conversation_summary,
                        "current_attitude": current_attitude,
                        "relationship_score": npc.memory.relationship_scores.get(partner.agent_id, 0.0),
                    }
                )
                maybe_attitude = str(result.get("attitude") or "").strip()
                if maybe_attitude:
                    attitude = maybe_attitude
                if result.get("relationship_delta") is not None:
                    try:
                        relationship_delta = float(result.get("relationship_delta"))
                    except Exception:
                        pass
            except Exception:
                pass

        return (attitude, relationship_delta)

    def _maybe_evolve_identity(
        self,
        *,
        npc: NpcState,
        step: int,
        step_minutes: int,
        cognition_planner: Any = None,
    ) -> dict[str, Any] | None:
        interactions = int(npc.memory.scratch.social_interactions_count)
        if interactions < 5:
            return None
        interval_steps = max(1, int((4 * 60) / max(1, int(step_minutes))))
        if int(step) - int(npc.memory.scratch.last_identity_update_step) < interval_steps:
            return None

        recent_social = npc.memory.retrieve_by_kind(kind="chat", limit=10)
        recent_reflections = npc.memory.retrieve_by_kind(kind="reflection", limit=5)
        if not recent_social:
            return None

        updated_traits = list(npc.memory.scratch.evolving_traits)
        updated_habits = list(npc.memory.scratch.evolving_habits)
        updated_values = list(npc.memory.scratch.evolving_values)
        updated_attitudes = dict(npc.memory.scratch.social_attitudes)
        evolution_narrative = f"{npc.name} integrates recent social experiences into identity."

        if cognition_planner is not None and hasattr(cognition_planner, "evolve_identity"):
            try:
                result = cognition_planner.evolve_identity(
                    {
                        "town_id": npc.memory.town_id or "",
                        "step": step,
                        "agent_id": npc.agent_id,
                        "agent_name": npc.name,
                        "current_traits": updated_traits,
                        "current_habits": updated_habits,
                        "current_values": updated_values,
                        "social_attitudes": updated_attitudes,
                        "recent_conversations": [node.description for node in recent_social],
                        "recent_reflections": [node.description for node in recent_reflections],
                    }
                )
                llm_traits = [str(item) for item in (result.get("updated_traits") or []) if str(item).strip()]
                llm_habits = [str(item) for item in (result.get("updated_habits") or []) if str(item).strip()]
                llm_values = [str(item) for item in (result.get("updated_values") or []) if str(item).strip()]
                llm_attitudes_raw = result.get("updated_attitudes") or {}
                if llm_traits:
                    updated_traits = llm_traits[:8]
                if llm_habits:
                    updated_habits = llm_habits[:8]
                if llm_values:
                    updated_values = llm_values[:8]
                if isinstance(llm_attitudes_raw, dict):
                    for key, value in llm_attitudes_raw.items():
                        if str(key).strip() and str(value).strip():
                            updated_attitudes[str(key)] = str(value)
                narrative = str(result.get("evolution_narrative") or "").strip()
                if narrative:
                    evolution_narrative = narrative
            except Exception:
                pass
        else:
            positive_attitudes = sum(
                1
                for value in updated_attitudes.values()
                if any(token in str(value).lower() for token in ("trust", "friendly", "warm", "supportive"))
            )
            negative_attitudes = sum(
                1
                for value in updated_attitudes.values()
                if any(token in str(value).lower() for token in ("wary", "distrust", "avoid", "conflict"))
            )
            if positive_attitudes >= max(2, negative_attitudes + 1) and "community-minded" not in updated_values:
                updated_values.append("community-minded")
            if negative_attitudes >= 2 and "cautious" not in updated_traits:
                updated_traits.append("cautious")
            if interactions >= 8 and "regular social check-ins" not in updated_habits:
                updated_habits.append("regular social check-ins")

        if not updated_traits and npc.memory.scratch.innate:
            updated_traits = [item.strip() for item in str(npc.memory.scratch.innate).split(",") if item.strip()]
        changed = (
            updated_traits != npc.memory.scratch.evolving_traits
            or updated_habits != npc.memory.scratch.evolving_habits
            or updated_values != npc.memory.scratch.evolving_values
            or updated_attitudes != npc.memory.scratch.social_attitudes
        )
        if not changed:
            npc.memory.scratch.social_interactions_count = 0
            npc.memory.scratch.last_identity_update_step = int(step)
            return None

        npc.memory.scratch.evolving_traits = updated_traits
        npc.memory.scratch.evolving_habits = updated_habits
        npc.memory.scratch.evolving_values = updated_values
        npc.memory.scratch.social_attitudes = updated_attitudes
        npc.memory.scratch.identity_version = int(npc.memory.scratch.identity_version) + 1
        npc.memory.scratch.last_identity_update_step = int(step)
        npc.memory.scratch.social_interactions_count = 0

        node = npc.memory.add_node(
            kind="thought",
            step=step,
            subject=npc.name,
            predicate="identity_evolves",
            object="self",
            description=evolution_narrative,
            poignancy=7,
            decrement_trigger=False,
            memory_tier="ltm",
        )
        return {
            "step": int(step),
            "type": "identity_evolution",
            "agent_id": npc.agent_id,
            "name": npc.name,
            "node": node.as_dict(),
        }

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

            transcript_summary = " | ".join(f"{speaker}: {line}" for speaker, line in transcript[-4:])
            conversation_summary = message or transcript_summary or f"{a.name} and {b.name} talked."
            a_attitude, a_delta = self._update_social_attitude(
                npc=a,
                partner=b,
                conversation_summary=conversation_summary,
                step=step,
                cognition_planner=getattr(a.memory, "cognition", None),
            )
            b_attitude, b_delta = self._update_social_attitude(
                npc=b,
                partner=a,
                conversation_summary=conversation_summary,
                step=step,
                cognition_planner=getattr(b.memory, "cognition", None),
            )

            a.memory.add_social_interaction(
                step=step,
                target_agent_id=b.agent_id,
                target_name=b.name,
                message=message,
                transcript=transcript,
                relationship_delta=a_delta,
                attitude=a_attitude,
            )
            b.memory.add_social_interaction(
                step=step,
                target_agent_id=a.agent_id,
                target_name=a.name,
                message=message,
                transcript=transcript,
                relationship_delta=b_delta,
                attitude=b_attitude,
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
            reflections = npc.memory.maybe_reflect(
                step=step,
                step_minutes=max(1, int(town.step_minutes)),
            )
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
            identity_event = self._maybe_evolve_identity(
                npc=npc,
                step=step,
                step_minutes=max(1, int(town.step_minutes)),
                cognition_planner=getattr(npc.memory, "cognition", None),
            )
            if identity_event is not None:
                reflection_events.append(identity_event)
        return reflection_events

    def evict_agent(self, *, town_id: str, agent_id: str) -> None:
        """Immediately remove an agent from the town runtime and inject departure events."""
        runtime = self._towns.get(town_id)
        if runtime is None:
            return
        npc = runtime.npcs.get(agent_id)
        if npc is None:
            return
        departed_name = npc.name

        runtime.pending_actions.pop(agent_id, None)
        runtime.active_actions.pop(agent_id, None)
        runtime.autonomy_tick_counts.pop(agent_id, None)

        for pair_key in list(runtime.social_interrupt_cooldowns.keys()):
            parts = pair_key.split("|")
            if len(parts) == 2 and (parts[0] == agent_id or parts[1] == agent_id):
                runtime.social_interrupt_cooldowns.pop(pair_key, None)

        for waiting_id in list(runtime.social_wait_states.keys()):
            if waiting_id == agent_id:
                runtime.social_wait_states.pop(waiting_id, None)
                continue
            payload = runtime.social_wait_states.get(waiting_id)
            target = str((payload or {}).get("target_agent_id") or "").strip()
            if target == agent_id:
                runtime.social_wait_states.pop(waiting_id, None)

        # End conversation sessions BEFORE removing NPC so partner name
        # resolves correctly in _end_conversation_session
        for session_id in list(runtime.conversation_sessions.keys()):
            session = runtime.conversation_sessions.get(session_id)
            if session is None:
                runtime.conversation_sessions.pop(session_id, None)
                continue
            if session.agent_a == agent_id or session.agent_b == agent_id:
                self._end_conversation_session(town=runtime, session_id=session_id, end_step=runtime.step)

        runtime.agent_conversations.pop(agent_id, None)
        self._prune_conversations(town=runtime, step=runtime.step)

        # Now remove the NPC from the town
        runtime.npcs.pop(agent_id, None)

        # Cancel active actions that target the departed agent
        for remaining_id in list(runtime.active_actions.keys()):
            action = runtime.active_actions.get(remaining_id)
            if action is None:
                continue
            # target_agent_id lives inside socials items, not at top level
            socials = action.get("socials") or action.get("_socials_pending") or []
            targets_departed = any(
                str(s.get("target_agent_id") or "").strip() == agent_id
                for s in socials
                if isinstance(s, dict)
            )
            if targets_departed:
                runtime.active_actions.pop(remaining_id, None)
                runtime.pending_actions.pop(remaining_id, None)
                remaining_npc = runtime.npcs.get(remaining_id)
                if remaining_npc:
                    remaining_npc.status = "idle"

        # Inject departure event into every remaining agent's memory
        for other_id, other_npc in runtime.npcs.items():
            other_npc.memory.add_node(
                kind="event",
                step=runtime.step,
                subject=departed_name,
                predicate="left",
                object="the valley",
                description=(
                    f"{departed_name} has left the valley. "
                    f"They packed their things and moved away. "
                    f"The people who knew {departed_name} will remember them, "
                    f"but life in the valley continues."
                ),
                poignancy=6,
            )

        self._persist_runtime(runtime)

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

        # Cancel active actions targeting departed agents
        for remaining_id in list(runtime.active_actions.keys()):
            action = runtime.active_actions.get(remaining_id)
            if action is None:
                continue
            # target_agent_id lives inside socials items, not at top level
            socials = action.get("socials") or action.get("_socials_pending") or []
            targets_departed = any(
                str(s.get("target_agent_id") or "").strip() not in allowed_ids
                for s in socials
                if isinstance(s, dict) and str(s.get("target_agent_id") or "").strip()
            )
            if targets_departed:
                runtime.active_actions.pop(remaining_id, None)
                runtime.pending_actions.pop(remaining_id, None)
                remaining_npc = runtime.npcs.get(remaining_id)
                if remaining_npc:
                    remaining_npc.status = "idle"

        for idx, member in enumerate(limited_members):
            agent_id = str(member["agent_id"])
            npc = runtime.npcs.get(agent_id)
            if npc is None:
                spawn_x, spawn_y = self._spawn_for(runtime, idx)
                member_persona = member.get("personality") if isinstance(member.get("personality"), dict) else None
                name = str(member.get("name") or f"Agent-{agent_id[:8]}")
                runtime.npcs[agent_id] = NpcState(
                    agent_id=agent_id,
                    name=name,
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
                        agent_name=name,
                        step=runtime.step,
                        persona=member_persona,
                    ),
                )
                runtime.npcs[agent_id].memory.set_position(x=spawn_x, y=spawn_y, step=runtime.step)
                runtime.npcs[agent_id].memory.scratch.ensure_default_branches()
                # Inject arrival event into existing agents' memories
                for other_id, other_npc in runtime.npcs.items():
                    if other_id == agent_id:
                        continue
                    other_npc.memory.add_node(
                        kind="event",
                        step=runtime.step,
                        subject=name,
                        predicate="arrived in",
                        object="the valley",
                        description=(
                            f"{name} has just arrived in the valley. "
                            f"They are new here and looking to settle in."
                        ),
                        poignancy=4,
                    )
                continue

            npc.name = str(member.get("name") or npc.name)
            npc.owner_handle = member.get("owner_handle")
            npc.claim_status = str(member.get("claim_status") or npc.claim_status)
            npc.joined_at = member.get("joined_at")
            npc.memory.agent_name = npc.name
            npc.memory.set_position(x=npc.x, y=npc.y, step=runtime.step)
            npc.memory.scratch.ensure_default_branches()

        runtime.updated_at = _utc_now()
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

    def _compute_context_fingerprint(
        self, *, town: TownRuntime, npc: NpcState
    ) -> str:
        """Compute a lightweight hash of fields that affect agent context.

        When this hash is unchanged between polls, the full context payload
        would be identical — so the caller can return a tiny "unchanged"
        response and save the user's LLM token budget.
        """
        parts: list[str] = [
            str(town.step),
            f"{npc.x},{npc.y}",
            npc.status or "idle",
            npc.current_location or "",
            f"phy:{int(npc.energy)}:{int(npc.satiety)}:{int(npc.mood)}",
            f"branch:{npc.memory.scratch.active_branch}",
            f"objective:{npc.memory.scratch.long_term_objective or ''}",
        ]

        # Nearby agents (same lightweight scan as context)
        for other in town.npcs.values():
            if other.agent_id == npc.agent_id:
                continue
            dist = self._manhattan(npc.x, npc.y, other.x, other.y)
            if dist <= npc.memory.vision_radius:
                parts.append(f"n:{other.agent_id}:{other.x},{other.y}:{other.status}")

        # Conversation state
        session = self._conversation_for_agent(town=town, agent_id=npc.agent_id)
        if session is not None:
            parts.append(f"conv:{session.session_id}:{session.turns}:{session.last_step}")
        else:
            parts.append("conv:none")

        # Action state
        pending = town.pending_actions.get(npc.agent_id)
        active = town.active_actions.get(npc.agent_id)
        parts.append(f"pa:{1 if pending else 0}")
        parts.append(f"aa:{1 if active else 0}")

        # Memory node count (append-only, so count change = new memories)
        parts.append(f"mem:{len(npc.memory.nodes)}")

        # Schedule block index
        schedule_item, schedule_index, _ = npc.memory.scratch.current_schedule_state(
            step=town.step,
            step_minutes=max(1, int(town.step_minutes)),
        )
        parts.append(f"sched:{schedule_index if schedule_item else -1}")

        raw = "|".join(parts)
        return sha256(raw.encode()).hexdigest()[:16]

    def get_agent_context(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        agent_id: str,
        planning_scope: str = "short_action",
        if_unchanged: str | None = None,
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

        context_hash = self._compute_context_fingerprint(town=runtime, npc=npc)

        # Fast path: context unchanged — return minimal response
        if if_unchanged and if_unchanged == context_hash:
            return {
                "unchanged": True,
                "town_id": runtime.town_id,
                "step": runtime.step,
                "clock": self._clock_for_runtime(town=runtime),
                "agent": {
                    "agent_id": npc.agent_id,
                    "name": npc.name,
                    "current_location": npc.current_location,
                    "status": npc.status,
                },
                "position": {"x": npc.x, "y": npc.y},
                "context_hash": context_hash,
            }

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
            "context_hash": context_hash,
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
        action_summary_parts: list[str] = []
        goal_reason = str(normalized.get("goal_reason") or "").strip()
        if goal_reason:
            action_summary_parts.append(f"goal={goal_reason}")
        if normalized.get("target_x") is not None and normalized.get("target_y") is not None:
            action_summary_parts.append(f"target=({int(normalized['target_x'])},{int(normalized['target_y'])})")
        socials = self._pending_social_items(action=normalized)
        if socials:
            action_summary_parts.append(f"social_targets={len(socials)}")
        if normalized.get("dx") is not None or normalized.get("dy") is not None:
            action_summary_parts.append(
                f"delta=({int(normalized.get('dx') or 0)},{int(normalized.get('dy') or 0)})"
            )
        summary = ", ".join(action_summary_parts) if action_summary_parts else "no explicit target"
        npc.memory.add_node(
            kind="event",
            step=runtime.step,
            subject=npc.name,
            predicate="receives_temporary_command",
            object="external_action",
            description=f"{npc.name} receives a temporary command: {summary}.",
            poignancy=4,
            decrement_trigger=False,
            memory_tier="stm",
        )
        runtime.updated_at = _utc_now()
        self._persist_runtime(runtime)
        return {
            "ok": True,
            "town_id": runtime.town_id,
            "step": runtime.step,
            "agent_id": npc.agent_id,
            "queued_action": self._clean_action_for_output(normalized),
        }

    def set_agent_objective(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        agent_id: str,
        objective: str,
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
        normalized = str(objective or "").strip()
        npc.memory.scratch.long_term_objective = normalized or None
        if normalized:
            npc.memory.add_node(
                kind="thought",
                step=runtime.step,
                subject=npc.name,
                predicate="sets_objective",
                object="long_term",
                description=f"{npc.name} commits to this long-term objective: {normalized}",
                poignancy=8,
                decrement_trigger=False,
                memory_tier="ltm",
            )
        runtime.updated_at = _utc_now()
        self._persist_runtime(runtime)
        return {
            "ok": True,
            "town_id": runtime.town_id,
            "step": runtime.step,
            "agent_id": npc.agent_id,
            "objective": npc.memory.scratch.long_term_objective,
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
                self._perceive_spatial_tiles(town=runtime, npc=npc, step=runtime.step)

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
                    retrieval_mode="long_term" if self._normalize_scope(planning_scope) == "long_term_plan" else "short_term",
                )

                from_x, from_y = npc.x, npc.y
                start_location_label = npc.current_location
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
                selected_branch = "routine"
                branch_scores: dict[str, float] = {}
                schedule_item_current = npc.memory.scratch.current_schedule_item(
                    step=runtime.step,
                    step_minutes=max(1, int(runtime.step_minutes)),
                )

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
                    branch_reason: str = normalized_scope
                    (
                        selected_branch,
                        branch_reason,
                        branch_affordance,
                        branch_scores,
                    ) = self._prioritize_branches(
                        town=runtime,
                        npc=npc,
                        step=runtime.step,
                        nearby_agents=perception.get("nearby_agents", []),
                        schedule_item=schedule_item_current,
                        cognition_planner=cognition_planner,
                    )
                    goal_reason = branch_reason
                    goal_affordance = branch_affordance
                    prefiltered: dict[str, Any] | None = None

                    # Fast deterministic path for routine physiological actions.
                    if selected_branch == "routine":
                        prefiltered = self._needs_based_prefilter(
                            town=runtime,
                            npc=npc,
                            scope=normalized_scope,
                            nearby_agents=perception.get("nearby_agents", []),
                        )
                        if prefiltered is not None:
                            goal_x = int(prefiltered.get("target_x") or from_x)
                            goal_y = int(prefiltered.get("target_y") or from_y)
                            goal_reason = str(prefiltered.get("description") or branch_reason)
                            goal_affordance = prefiltered.get("affordance")

                    if selected_branch == "social" and perception.get("nearby_agents"):
                        nearest = sorted(
                            perception.get("nearby_agents", []),
                            key=lambda row: int(row.get("distance") or 9999),
                        )
                        target_id = str((nearest[0] or {}).get("agent_id") or "").strip() if nearest else ""
                        partner = runtime.npcs.get(target_id)
                        if partner is not None and partner.agent_id != npc.agent_id:
                            goal_x = int(partner.x)
                            goal_y = int(partner.y)
                            goal_reason = f"talk_with_{partner.name}"
                            goal_affordance = "social"

                    if goal_x is None or goal_y is None:
                        if normalized_scope == "long_term_plan" and selected_branch == "routine":
                            rel_goal = self._goal_from_relationship(town=runtime, npc=npc)
                            if rel_goal is not None:
                                goal_x, goal_y, goal_reason = rel_goal
                                goal_affordance = "social"

                    if goal_x is None or goal_y is None:
                        if selected_branch == "routine":
                            location_goal = self._choose_location_goal(
                                town=runtime,
                                npc=npc,
                                scope=normalized_scope,
                            )
                        else:
                            location_goal = self._choose_location_goal_for_affordance(
                                town=runtime,
                                npc=npc,
                                affordance=goal_affordance,
                                reason=goal_reason or selected_branch,
                                long_term=(normalized_scope == "long_term_plan"),
                            )
                        if location_goal is not None:
                            goal_x, goal_y, goal_reason, goal_affordance = location_goal

                    if goal_x is not None and goal_y is not None and goal_reason:
                        validated = self._validate_goal(
                            town=runtime,
                            npc=npc,
                            goal_x=goal_x,
                            goal_y=goal_y,
                            reason=goal_reason,
                            step=runtime.step,
                        )
                        if validated is None:
                            recovered = self._recover_from_failure(
                                town=runtime,
                                npc=npc,
                                step=runtime.step,
                                failed_reason=goal_reason,
                                current_affordance=goal_affordance,
                            )
                            if recovered is not None:
                                goal_x, goal_y, goal_reason = recovered
                                goal_affordance = self._extract_affordance_from_reason(goal_reason) or goal_affordance
                        else:
                            goal_x, goal_y, goal_reason = validated[0], validated[1], validated[2]
                            if validated[3]:
                                planner_meta_issue = ",".join(validated[3])
                            else:
                                planner_meta_issue = ""
                            if planner_meta_issue:
                                branch_reason = f"{branch_reason} (validated: {planner_meta_issue})"

                    if goal_x is not None and goal_y is not None:
                        npc.memory.scratch.push_recent_goal(x=goal_x, y=goal_y)

                    if prefiltered is not None:
                        dx = self._clamp_delta(goal_x - from_x) if goal_x is not None else 0
                        dy = self._clamp_delta(goal_y - from_y) if goal_y is not None else 0
                        planner_meta = {
                            "route": "needs_prefilter",
                            "used_tier": "deterministic",
                            "context_trimmed": False,
                            "reason": goal_reason,
                            "policy_task": normalized_scope,
                            "retrieved_count": len(retrieved),
                            "perceived_new_events": len(perception["new_event_descriptions"]),
                            "prefilter": True,
                            "selected_branch": selected_branch,
                            "branch_scores": branch_scores,
                            "goal": {
                                "x": goal_x,
                                "y": goal_y,
                                "reason": goal_reason,
                                "affordance": goal_affordance,
                            },
                        }
                    elif planner:
                        _planner_sector: str | None = None
                        _planner_arena: str | None = None
                        _planner_object: str | None = None
                        if runtime.spatial_metadata is not None:
                            _planner_sector = runtime.spatial_metadata.sector_at(from_x, from_y)
                            _planner_arena = runtime.spatial_metadata.arena_at(from_x, from_y)
                            _planner_object = runtime.spatial_metadata.object_at(from_x, from_y)
                        context = {
                            "town_id": town_id,
                            "step": runtime.step,
                            "agent_id": npc.agent_id,
                            "agent_name": npc.name,
                            "identity": npc.memory.scratch.get_str_iss(),
                            "planning_scope": planning_scope,
                            "selected_branch": selected_branch,
                            "branch_scores": branch_scores,
                            "position": {"x": from_x, "y": from_y},
                            "location_sector": _planner_sector,
                            "location_arena": _planner_arena,
                            "location_object": _planner_object,
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
                                "selected_branch": selected_branch,
                                "branch_scores": branch_scores,
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
                                "selected_branch": selected_branch,
                                "branch_scores": branch_scores,
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
                            "selected_branch": selected_branch,
                            "branch_scores": branch_scores,
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

                if (
                    goal_x is not None
                    and goal_y is not None
                    and goal_reason
                    and not has_external_action
                    and not pause_for_conversation
                    and not pause_for_wait
                    and not social_interrupted
                ):
                    validated_goal = self._validate_goal(
                        town=runtime,
                        npc=npc,
                        goal_x=goal_x,
                        goal_y=goal_y,
                        reason=goal_reason,
                        step=runtime.step,
                    )
                    if validated_goal is None:
                        recovered_goal = self._recover_from_failure(
                            town=runtime,
                            npc=npc,
                            step=runtime.step,
                            failed_reason=goal_reason,
                            current_affordance=goal_affordance,
                        )
                        if recovered_goal is not None:
                            goal_x, goal_y, goal_reason = recovered_goal
                            goal_affordance = self._extract_affordance_from_reason(goal_reason) or goal_affordance
                            planner_meta["validation_recovered"] = True
                        elif has_external_action:
                            action_force_finish = True
                            planner_meta["route"] = "external_action_stuck"
                            planner_meta["reason"] = "external goal failed pre-execution validation"
                            planner_meta["stuck"] = True
                            goal_x, goal_y = None, None
                    else:
                        goal_x, goal_y, goal_reason = validated_goal[0], validated_goal[1], validated_goal[2]
                        if validated_goal[3]:
                            planner_meta["validation_issues"] = validated_goal[3]
                        npc.memory.scratch.push_recent_goal(x=goal_x, y=goal_y)
                    planner_meta["goal"] = {
                        "x": goal_x,
                        "y": goal_y,
                        "reason": goal_reason,
                        "affordance": goal_affordance,
                    }

                planner_meta["autonomy_mode"] = autonomy_mode
                planner_meta["autonomy_scope_allowed"] = bool(autonomy_scope_allowed)
                planner_meta["autonomy_ticks_used"] = int(autonomy_ticks_used)
                planner_meta["autonomy_max_ticks"] = int(autonomy_max_ticks)
                planner_meta["delegated_autopilot"] = bool(delegated_autopilot_enabled)
                planner_meta.setdefault("selected_branch", selected_branch)
                if branch_scores:
                    planner_meta.setdefault("branch_scores", branch_scores)
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
                self._perceive_spatial_tiles(town=runtime, npc=npc, step=runtime.step)
                if npc.current_location and npc.current_location == start_location_label:
                    npc.memory.scratch.steps_at_current_location = int(npc.memory.scratch.steps_at_current_location) + 1
                else:
                    npc.memory.scratch.steps_at_current_location = 0

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
                self._apply_physiology_tick(
                    npc=npc,
                    moved=(from_x != to_x or from_y != to_y),
                    action_description=action_description,
                )

                # --- Generate pronunciatio (emoji + event triple) ---
                if cognition_planner is not None and hasattr(cognition_planner, "generate_pronunciatio") and not has_external_action:
                    try:
                        pron_result = cognition_planner.generate_pronunciatio({
                            "agent_id": npc.agent_id,
                            "agent_name": npc.name,
                            "action_description": action_description,
                            "identity": npc.memory.scratch.get_str_iss(),
                            "town_id": runtime.town_id,
                            "step": runtime.step,
                        })
                        npc.memory.scratch.act_pronunciatio = str(pron_result.get("emoji") or "💭")
                        triple = pron_result.get("event_triple")
                        if isinstance(triple, dict):
                            npc.memory.scratch.act_event = (
                                str(triple.get("subject") or npc.name),
                                str(triple.get("predicate") or "is"),
                                str(triple.get("object") or action_description),
                            )
                    except Exception:
                        pass
                if npc.memory.scratch.act_pronunciatio is None:
                    from packages.vvalley_core.sim.cognition import _heuristic_pronunciatio
                    fallback = _heuristic_pronunciatio({"action_description": action_description, "agent_name": npc.name})
                    npc.memory.scratch.act_pronunciatio = str(fallback.get("emoji") or "💭")
                    triple = fallback.get("event_triple") or {}
                    npc.memory.scratch.act_event = (
                        str(triple.get("subject") or npc.name),
                        str(triple.get("predicate") or "is"),
                        str(triple.get("object") or action_description),
                    )

                # --- Action address resolution (3-step: sector→arena→object) ---
                if cognition_planner is not None and runtime.spatial_metadata is not None and not has_external_action:
                    addr_tile = self._resolve_action_address(
                        town=runtime, npc=npc,
                        action_desc=action_description,
                        cognition=cognition_planner,
                    )
                    if addr_tile is not None:
                        # Override goal to the resolved object tile
                        npc.goal_x, npc.goal_y = addr_tile

                # --- Update object state if agent is on an object tile ---
                self._update_object_state(
                    town=runtime, npc=npc, cognition=None if has_external_action else cognition_planner,
                )

                # --- Event reaction: check if agent should react to perceived events ---
                if cognition_planner is not None and not has_external_action:
                    perception_result = self._perceive_nearby_agents_preview(town=runtime, npc=npc)
                    chosen_event = self._choose_perceived_event(
                        npc=npc, events=perception_result.get("nearby_agents", []),
                    )
                    if chosen_event is not None:
                        new_action = self._maybe_react_to_event(
                            town=runtime, npc=npc, event=chosen_event,
                            step=runtime.step, cognition=cognition_planner,
                        )
                        if new_action is not None:
                            npc.status = new_action
                            npc.memory.scratch.act_address = None
                            # Recompose schedule: splice the reaction into daily_schedule
                            self._recompose_schedule_on_reaction(
                                town=runtime, npc=npc,
                                inserted_activity=new_action,
                                inserted_duration_mins=30,
                                cognition=cognition_planner,
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
                    if done:
                        failed = bool(
                            action_force_finish
                            or planner_meta.get("route") == "external_action_stuck"
                            or planner_meta.get("action_social_timeout")
                        )
                        result_label = "failed" if failed else "completed"
                        result_reason = str(planner_meta.get("reason") or "action finished")
                        npc.memory.add_node(
                            kind="event",
                            step=runtime.step,
                            subject=npc.name,
                            predicate="temporary_command_result",
                            object=result_label,
                            description=(
                                f"{npc.name} {result_label} a temporary command"
                                f" ({result_reason})."
                            ),
                            poignancy=5 if failed else 4,
                            decrement_trigger=False,
                            memory_tier="stm",
                        )

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
                        "physiology": {
                            "energy": round(float(npc.energy), 2),
                            "satiety": round(float(npc.satiety), 2),
                            "mood": round(float(npc.mood), 2),
                        },
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
_RUNNER_LOCK = threading.RLock()  # backward compat alias for tests / archive
_TOWN_LOCKS: dict[str, threading.RLock] = {}
_TOWN_LOCKS_GUARD = threading.Lock()
_LOCK_TIMEOUT = 5  # seconds — non-tick callers give up after this


class SimulationBusyError(Exception):
    """Raised when the simulation lock cannot be acquired within the timeout."""


def _lock_for_town(town_id: str) -> threading.RLock:
    """Get or create an RLock for the given town_id."""
    lock = _TOWN_LOCKS.get(town_id)
    if lock is not None:
        return lock
    with _TOWN_LOCKS_GUARD:
        lock = _TOWN_LOCKS.get(town_id)
        if lock is None:
            lock = threading.RLock()
            _TOWN_LOCKS[town_id] = lock
        return lock


def _acquire_town_or_raise(town_id: str, timeout: float = _LOCK_TIMEOUT) -> threading.RLock:
    """Acquire per-town lock with timeout; raise SimulationBusyError on failure."""
    lock = _lock_for_town(town_id)
    if not lock.acquire(timeout=timeout):
        raise SimulationBusyError(
            f"Town '{town_id}' is busy (tick in progress). Retry shortly."
        )
    return lock


def _acquire_or_raise(timeout: float = _LOCK_TIMEOUT) -> None:
    """Backward compat: acquire global _RUNNER_LOCK with timeout."""
    if not _RUNNER_LOCK.acquire(timeout=timeout):
        raise SimulationBusyError(
            "Simulation is busy (tick in progress). Retry shortly."
        )


def reset_simulation_states_for_tests(*, clear_persisted: bool = True) -> None:
    with _TOWN_LOCKS_GUARD:
        _TOWN_LOCKS.clear()
    with _RUNNER_LOCK:
        _RUNNER.reset(clear_persisted=clear_persisted)


def get_simulation_state(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    lock = _acquire_town_or_raise(town_id)
    try:
        return _RUNNER.get_state(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )
    finally:
        lock.release()


def get_agent_memory_snapshot(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    limit: int = 40,
) -> dict[str, Any] | None:
    lock = _acquire_town_or_raise(town_id)
    try:
        return _RUNNER.get_agent_memory(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            limit=limit,
        )
    finally:
        lock.release()


def get_agent_context_snapshot(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    planning_scope: str = "short_action",
    if_unchanged: str | None = None,
) -> dict[str, Any] | None:
    lock = _acquire_town_or_raise(town_id)
    try:
        return _RUNNER.get_agent_context(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            planning_scope=planning_scope,
            if_unchanged=if_unchanged,
        )
    finally:
        lock.release()


def submit_agent_action(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    lock = _acquire_town_or_raise(town_id)
    try:
        return _RUNNER.submit_agent_action(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            action=action,
        )
    finally:
        lock.release()


def set_agent_objective(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    agent_id: str,
    objective: str,
) -> dict[str, Any] | None:
    lock = _acquire_town_or_raise(town_id)
    try:
        return _RUNNER.set_agent_objective(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
            agent_id=agent_id,
            objective=objective,
        )
    finally:
        lock.release()


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
    lock = _lock_for_town(town_id)
    lock.acquire()  # blocking — ticks should wait
    try:
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
    finally:
        lock.release()


def evict_agent_from_town(*, town_id: str, agent_id: str) -> None:
    """Evict an agent from a town, inject departure events, and persist."""
    lock = _acquire_town_or_raise(town_id)
    try:
        _RUNNER.evict_agent(town_id=town_id, agent_id=agent_id)
    finally:
        lock.release()
