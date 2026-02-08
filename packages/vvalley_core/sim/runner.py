"""In-memory simulation runtime with memory-aware cognition for V-Valley towns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Optional

from packages.vvalley_core.maps.map_utils import collision_grid, parse_location_objects, parse_spawns
from packages.vvalley_core.sim.memory import AgentMemory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


PlannerFn = Callable[[dict[str, Any]], dict[str, Any]]


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
            "memory_summary": self.memory.summary(),
        }


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
    npcs: dict[str, NpcState] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, max_agents: int) -> dict[str, Any]:
        ordered_npcs = sorted(self.npcs.values(), key=lambda npc: (npc.name.lower(), npc.agent_id))
        return {
            "town_id": self.town_id,
            "map_version_id": self.map_version_id,
            "map_version": self.map_version,
            "map_name": self.map_name,
            "map_width": self.width,
            "map_height": self.height,
            "location_count": len(self.location_points),
            "step": self.step,
            "npc_count": len(ordered_npcs),
            "max_agents": max_agents,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "npcs": [npc.to_dict() for npc in ordered_npcs],
            "recent_events": list(self.recent_events),
        }


class SimulationRunner:
    """Deterministic town simulation engine with memory-aware state."""

    def __init__(
        self,
        *,
        max_agents_per_town: int = 25,
        max_recent_events: int = 400,
        max_social_events_per_step: int = 4,
    ) -> None:
        self.max_agents_per_town = max_agents_per_town
        self.max_recent_events = max_recent_events
        self.max_social_events_per_step = max_social_events_per_step
        self._towns: dict[str, TownRuntime] = {}

    def reset(self) -> None:
        self._towns.clear()

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

    @staticmethod
    def _adjacent(a: NpcState, b: NpcState) -> bool:
        return abs(a.x - b.x) + abs(a.y - b.y) <= 1

    def _social_message(self, *, a: NpcState, b: NpcState, step: int) -> str:
        templates = (
            "{a} shares a town rumor with {b}.",
            "{a} and {b} trade quick survival tips.",
            "{a} asks {b} about their day plan.",
            "{a} and {b} sync on where to gather later.",
            "{a} gives {b} a friendly wave and joke.",
        )
        idx = self._hash_int(f"social:{a.agent_id}:{b.agent_id}:{step}") % len(templates)
        return templates[idx].format(a=a.name, b=b.name)

    def _build_social_events(self, *, town: TownRuntime, step: int) -> list[dict[str, Any]]:
        npcs = sorted(town.npcs.values(), key=lambda n: n.agent_id)
        out: list[dict[str, Any]] = []
        for i in range(len(npcs)):
            for j in range(i + 1, len(npcs)):
                a = npcs[i]
                b = npcs[j]
                if not self._adjacent(a, b):
                    continue
                out.append(
                    {
                        "step": step,
                        "type": "social",
                        "agents": [
                            {"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y},
                            {"agent_id": b.agent_id, "name": b.name, "x": b.x, "y": b.y},
                        ],
                        "message": self._social_message(a=a, b=b, step=step),
                    }
                )
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
            npc.memory.add_node(
                kind="event",
                step=step,
                subject=other.name,
                predicate="is",
                object=other.status,
                description=desc,
                poignancy=3,
                evidence_ids=(),
                decrement_trigger=True,
            )
            new_events.append(desc)

        return {
            "nearby_agents": nearby,
            "new_event_descriptions": new_events,
        }

    def _ingest_social_events(self, *, town: TownRuntime, social_events: list[dict[str, Any]], step: int) -> None:
        for event in social_events:
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
            a.memory.add_social_interaction(
                step=step,
                target_agent_id=b.agent_id,
                target_name=b.name,
                message=message,
            )
            b.memory.add_social_interaction(
                step=step,
                target_agent_id=a.agent_id,
                target_name=a.name,
                message=message,
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
        if runtime is None or runtime.map_version_id != str(active_version["id"]):
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

        for idx, member in enumerate(limited_members):
            agent_id = str(member["agent_id"])
            npc = runtime.npcs.get(agent_id)
            if npc is None:
                spawn_x, spawn_y = self._spawn_for(runtime, idx)
                runtime.npcs[agent_id] = NpcState(
                    agent_id=agent_id,
                    name=str(member.get("name") or f"Agent-{agent_id[:8]}"),
                    owner_handle=member.get("owner_handle"),
                    claim_status=str(member.get("claim_status") or "unknown"),
                    x=spawn_x,
                    y=spawn_y,
                    joined_at=member.get("joined_at"),
                    status="idle",
                    memory=AgentMemory.bootstrap(
                        agent_id=agent_id,
                        agent_name=str(member.get("name") or f"Agent-{agent_id[:8]}"),
                        step=runtime.step,
                    ),
                )
                continue

            npc.name = str(member.get("name") or npc.name)
            npc.owner_handle = member.get("owner_handle")
            npc.claim_status = str(member.get("claim_status") or npc.claim_status)
            npc.joined_at = member.get("joined_at")
            npc.memory.agent_name = npc.name

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

    def tick(
        self,
        *,
        town_id: str,
        active_version: dict[str, Any],
        map_data: dict[str, Any],
        members: list[dict[str, Any]],
        steps: int = 1,
        planning_scope: str = "short_action",
        planner: Optional[PlannerFn] = None,
    ) -> dict[str, Any]:
        runtime = self.sync_town(
            town_id=town_id,
            active_version=active_version,
            map_data=map_data,
            members=members,
        )

        events: list[dict[str, Any]] = []
        social_events: list[dict[str, Any]] = []
        reflection_events: list[dict[str, Any]] = []
        clamped_steps = max(1, int(steps))

        for _ in range(clamped_steps):
            runtime.step += 1

            for agent_id in sorted(runtime.npcs.keys()):
                npc = runtime.npcs[agent_id]
                self._update_location_memory(town=runtime, npc=npc, step=runtime.step)
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
                dx: int
                dy: int
                if planner:
                    context = {
                        "town_id": town_id,
                        "step": runtime.step,
                        "agent_id": npc.agent_id,
                        "agent_name": npc.name,
                        "planning_scope": planning_scope,
                        "position": {"x": from_x, "y": from_y},
                        "map": {"width": runtime.width, "height": runtime.height},
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
                        planner_meta = {
                            "route": decision.get("route", "planner"),
                            "used_tier": decision.get("used_tier"),
                            "context_trimmed": bool(decision.get("context_trimmed", False)),
                            "reason": decision.get("reason"),
                            "policy_task": decision.get("policy_task"),
                            "retrieved_count": len(retrieved),
                            "perceived_new_events": len(perception["new_event_descriptions"]),
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
                    }

                to_x, to_y = self._attempt_move(
                    town=runtime,
                    x=from_x,
                    y=from_y,
                    dx=dx,
                    dy=dy,
                )
                npc.x = to_x
                npc.y = to_y
                npc.last_step = runtime.step
                npc.status = "moving" if (from_x != to_x or from_y != to_y) else "idle"
                self._update_location_memory(town=runtime, npc=npc, step=runtime.step)

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
                    }
                )

            step_social_events = self._build_social_events(town=runtime, step=runtime.step)
            social_events.extend(step_social_events)
            self._ingest_social_events(town=runtime, social_events=step_social_events, step=runtime.step)
            step_reflections = self._run_reflection_cycle(town=runtime, step=runtime.step)
            reflection_events.extend(step_reflections)

        runtime.recent_events.extend(events)
        runtime.recent_events.extend(social_events)
        runtime.recent_events.extend(reflection_events)
        if len(runtime.recent_events) > self.max_recent_events:
            runtime.recent_events = runtime.recent_events[-self.max_recent_events :]
        runtime.updated_at = _utc_now()

        return {
            "town_id": town_id,
            "planning_scope": planning_scope,
            "steps_run": clamped_steps,
            "step": runtime.step,
            "event_count": len(events),
            "events": events,
            "social_event_count": len(social_events),
            "social_events": social_events,
            "reflection_event_count": len(reflection_events),
            "reflection_events": reflection_events,
            "state": runtime.to_dict(max_agents=self.max_agents_per_town),
        }


_RUNNER = SimulationRunner()


def reset_simulation_states_for_tests() -> None:
    _RUNNER.reset()


def get_simulation_state(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
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
    return _RUNNER.get_agent_memory(
        town_id=town_id,
        active_version=active_version,
        map_data=map_data,
        members=members,
        agent_id=agent_id,
        limit=limit,
    )


def tick_simulation(
    *,
    town_id: str,
    active_version: dict[str, Any],
    map_data: dict[str, Any],
    members: list[dict[str, Any]],
    steps: int = 1,
    planning_scope: str = "short_action",
    planner: Optional[PlannerFn] = None,
) -> dict[str, Any]:
    return _RUNNER.tick(
        town_id=town_id,
        active_version=active_version,
        map_data=map_data,
        members=members,
        steps=steps,
        planning_scope=planning_scope,
        planner=planner,
    )
