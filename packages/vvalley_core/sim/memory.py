"""Memory and cognition state inspired by Generative Agents, adapted for V-Valley."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import sqrt
from typing import Any, Iterable
import re
import uuid


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tokenize(value: str) -> tuple[str, ...]:
    lowered = str(value or "").lower()
    return tuple(tok for tok in _TOKEN_RE.findall(lowered) if tok not in _STOPWORDS)


def _normalize_dict_floats(scores: dict[str, float], *, lo: float = 0.0, hi: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    mn = min(float(v) for v in scores.values())
    mx = max(float(v) for v in scores.values())
    if mx - mn <= 1e-12:
        midpoint = (hi + lo) / 2.0
        return {k: midpoint for k in scores}
    return {k: ((float(v) - mn) * (hi - lo) / (mx - mn)) + lo for k, v in scores.items()}


def _embedding_for_tokens(tokens: Iterable[str], *, dims: int = 24) -> tuple[float, ...]:
    vec = [0.0] * dims
    used = False
    for raw in tokens:
        tok = str(raw).strip().lower()
        if not tok:
            continue
        used = True
        digest = sha256(tok.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        weight = 1.0 + (int(digest[10:12], 16) / 255.0)
        vec[bucket] += sign * weight
    if not used:
        return tuple(0.0 for _ in range(dims))
    norm = sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return tuple(0.0 for _ in range(dims))
    return tuple(v / norm for v in vec)


def _embedding_for_text(text: str, *, dims: int = 24) -> tuple[float, ...]:
    return _embedding_for_tokens(_tokenize(text), dims=dims)


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    an = 0.0
    bn = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        an += av * av
        bn += bv * bv
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return dot / (sqrt(an) * sqrt(bn))


@dataclass
class MemoryNode:
    node_id: str
    kind: str
    step: int
    created_at: str
    subject: str
    predicate: str
    object: str
    description: str
    poignancy: int
    keywords: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    last_accessed_step: int = 0
    depth: int = 0
    embedding_key: str = ""
    embedding: tuple[float, ...] = ()
    address: str | None = None
    expiration_step: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "step": self.step,
            "created_at": self.created_at,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "poignancy": self.poignancy,
            "keywords": list(self.keywords),
            "evidence_ids": list(self.evidence_ids),
            "last_accessed_step": self.last_accessed_step,
            "depth": self.depth,
            "embedding_key": self.embedding_key,
            "address": self.address,
            "expiration_step": self.expiration_step,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MemoryNode":
        if not isinstance(raw, dict):
            raise ValueError("MemoryNode payload must be an object")
        node_id = str(raw.get("node_id") or f"mem_{uuid.uuid4().hex[:12]}")
        description = str(raw.get("description") or "")
        embedding_key = str(raw.get("embedding_key") or description)
        keywords_raw = raw.get("keywords") or []
        keywords = tuple(str(value) for value in keywords_raw if str(value).strip())
        if not keywords:
            keywords = _tokenize(description)
        evidence_raw = raw.get("evidence_ids") or []
        expiration_value = raw.get("expiration_step")
        expiration_step: int | None = None
        if expiration_value is not None:
            try:
                expiration_step = int(expiration_value)
            except Exception:
                expiration_step = None
        return cls(
            node_id=node_id,
            kind=str(raw.get("kind") or "event"),
            step=int(raw.get("step") or 0),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            subject=str(raw.get("subject") or ""),
            predicate=str(raw.get("predicate") or ""),
            object=str(raw.get("object") or ""),
            description=description,
            poignancy=max(1, int(raw.get("poignancy") or 1)),
            keywords=keywords,
            evidence_ids=tuple(str(value) for value in evidence_raw if str(value).strip()),
            last_accessed_step=int(raw.get("last_accessed_step") or int(raw.get("step") or 0)),
            depth=max(0, int(raw.get("depth") or 0)),
            embedding_key=embedding_key,
            embedding=_embedding_for_text(embedding_key),
            address=(str(raw.get("address")) if raw.get("address") is not None else None),
            expiration_step=expiration_step,
        )


@dataclass
class ScheduleItem:
    description: str
    duration_mins: int
    affordance_hint: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "duration_mins": int(self.duration_mins),
            "affordance_hint": self.affordance_hint,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScheduleItem":
        if not isinstance(raw, dict):
            return cls(description="idle", duration_mins=60, affordance_hint=None)
        try:
            duration_mins = int(raw.get("duration_mins") or 60)
        except Exception:
            duration_mins = 60
        return cls(
            description=str(raw.get("description") or "idle"),
            duration_mins=max(1, duration_mins),
            affordance_hint=(str(raw.get("affordance_hint")) if raw.get("affordance_hint") is not None else None),
        )


@dataclass
class AgentScratch:
    """Short-term control state (GA scratch memory equivalent)."""

    vision_radius: int = 4
    attention_bandwidth: int = 3
    retention: int = 6

    recency_decay: float = 0.93
    recency_w: float = 1.0
    relevance_w: float = 1.0
    importance_w: float = 1.0
    importance_trigger_max: int = 28
    importance_trigger_curr: int = 28
    importance_ele_n: int = 0
    thought_count: int = 5

    day_minutes: int = 24 * 60
    daily_req: list[str] = field(default_factory=list)
    long_term_goals: list[str] = field(default_factory=list)
    daily_schedule: list[ScheduleItem] = field(default_factory=list)
    daily_schedule_hourly_original: list[ScheduleItem] = field(default_factory=list)

    curr_step: int = 0
    curr_tile: tuple[int, int] | None = None
    act_address: str | None = None
    act_start_step: int | None = None
    act_duration: int | None = None
    act_description: str | None = None
    act_path_set: bool = False
    planned_path: list[tuple[int, int]] = field(default_factory=list)

    chatting_with: str | None = None
    chat: list[tuple[str, str]] = field(default_factory=list)
    chatting_with_buffer: dict[str, int] = field(default_factory=dict)
    chatting_end_step: int | None = None

    next_daily_plan_step: int = 0
    next_long_term_plan_step: int = 0

    def ensure_default_schedule(self, *, agent_name: str) -> None:
        if self.daily_schedule:
            return
        self.daily_req = [
            f"{agent_name} maintains a stable routine",
            f"{agent_name} checks on relationships and town status",
            f"{agent_name} balances work, social, and rest",
        ]
        self.daily_schedule = [
            ScheduleItem("sleep", 360, "home"),
            ScheduleItem("morning routine", 60, "home"),
            ScheduleItem("work or task", 300, "work"),
            ScheduleItem("lunch", 60, "food"),
            ScheduleItem("socialize", 120, "social"),
            ScheduleItem("work follow-up", 240, "work"),
            ScheduleItem("leisure", 180, "leisure"),
            ScheduleItem("dinner", 60, "food"),
            ScheduleItem("night wind-down", 60, "home"),
        ]
        total = sum(int(item.duration_mins) for item in self.daily_schedule)
        if total < self.day_minutes:
            self.daily_schedule.append(ScheduleItem("sleep", self.day_minutes - total, "home"))
        self.daily_schedule_hourly_original = list(self.daily_schedule)
        if not self.long_term_goals:
            self.long_term_goals = [
                f"{agent_name} builds strong social ties in town",
                f"{agent_name} keeps a stable routine and useful role",
                f"{agent_name} adapts plans based on recent events",
            ]

    def current_schedule_state(self, *, step: int, step_minutes: int = 10) -> tuple[ScheduleItem | None, int, int]:
        self.curr_step = int(step)
        if not self.daily_schedule:
            return (None, -1, 0)
        minute_of_day = (int(step) * int(step_minutes)) % max(1, self.day_minutes)
        elapsed = 0
        for idx, item in enumerate(self.daily_schedule):
            duration = max(1, int(item.duration_mins))
            if minute_of_day < elapsed + duration:
                return (item, idx, minute_of_day - elapsed)
            elapsed += duration
        last = self.daily_schedule[-1]
        return (last, len(self.daily_schedule) - 1, max(0, minute_of_day - elapsed))

    def current_schedule_item(self, *, step: int, step_minutes: int = 10) -> ScheduleItem | None:
        item, _, _ = self.current_schedule_state(step=step, step_minutes=step_minutes)
        return item

    def as_dict(self) -> dict[str, object]:
        return {
            "vision_radius": int(self.vision_radius),
            "attention_bandwidth": int(self.attention_bandwidth),
            "retention": int(self.retention),
            "recency_decay": float(self.recency_decay),
            "recency_w": float(self.recency_w),
            "relevance_w": float(self.relevance_w),
            "importance_w": float(self.importance_w),
            "importance_trigger_max": int(self.importance_trigger_max),
            "importance_trigger_curr": int(self.importance_trigger_curr),
            "importance_ele_n": int(self.importance_ele_n),
            "thought_count": int(self.thought_count),
            "daily_req": list(self.daily_req),
            "long_term_goals": list(self.long_term_goals),
            "daily_schedule": [item.as_dict() for item in self.daily_schedule],
            "daily_schedule_hourly_original": [item.as_dict() for item in self.daily_schedule_hourly_original],
            "curr_step": int(self.curr_step),
            "curr_tile": list(self.curr_tile) if self.curr_tile is not None else None,
            "act_address": self.act_address,
            "act_start_step": self.act_start_step,
            "act_duration": self.act_duration,
            "act_description": self.act_description,
            "act_path_set": bool(self.act_path_set),
            "planned_path": [list(i) for i in self.planned_path],
            "chatting_with": self.chatting_with,
            "chat": [[speaker, utterance] for speaker, utterance in self.chat],
            "chatting_with_buffer": dict(self.chatting_with_buffer),
            "chatting_end_step": self.chatting_end_step,
            "next_daily_plan_step": int(self.next_daily_plan_step),
            "next_long_term_plan_step": int(self.next_long_term_plan_step),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentScratch":
        if not isinstance(raw, dict):
            return cls()
        scratch = cls(
            vision_radius=max(1, int(raw.get("vision_radius") or 4)),
            attention_bandwidth=max(1, int(raw.get("attention_bandwidth") or 3)),
            retention=max(1, int(raw.get("retention") or 6)),
            recency_decay=float(raw.get("recency_decay") or 0.93),
            recency_w=float(raw.get("recency_w") or 1.0),
            relevance_w=float(raw.get("relevance_w") or 1.0),
            importance_w=float(raw.get("importance_w") or 1.0),
            importance_trigger_max=max(1, int(raw.get("importance_trigger_max") or 28)),
            importance_trigger_curr=int(raw.get("importance_trigger_curr") or 28),
            importance_ele_n=max(0, int(raw.get("importance_ele_n") or 0)),
            thought_count=max(1, int(raw.get("thought_count") or 5)),
            day_minutes=max(60, int(raw.get("day_minutes") or (24 * 60))),
            daily_req=[str(value) for value in (raw.get("daily_req") or []) if str(value).strip()],
            long_term_goals=[str(value) for value in (raw.get("long_term_goals") or []) if str(value).strip()],
            curr_step=int(raw.get("curr_step") or 0),
            act_address=(str(raw.get("act_address")) if raw.get("act_address") is not None else None),
            act_start_step=(int(raw["act_start_step"]) if raw.get("act_start_step") is not None else None),
            act_duration=(int(raw["act_duration"]) if raw.get("act_duration") is not None else None),
            act_description=(str(raw.get("act_description")) if raw.get("act_description") is not None else None),
            act_path_set=bool(raw.get("act_path_set") or False),
            chatting_with=(str(raw.get("chatting_with")) if raw.get("chatting_with") is not None else None),
            chatting_end_step=(int(raw["chatting_end_step"]) if raw.get("chatting_end_step") is not None else None),
            next_daily_plan_step=max(0, int(raw.get("next_daily_plan_step") or 0)),
            next_long_term_plan_step=max(0, int(raw.get("next_long_term_plan_step") or 0)),
        )
        curr_tile = raw.get("curr_tile")
        if isinstance(curr_tile, (list, tuple)) and len(curr_tile) == 2:
            try:
                scratch.curr_tile = (int(curr_tile[0]), int(curr_tile[1]))
            except Exception:
                scratch.curr_tile = None
        daily_schedule_raw = raw.get("daily_schedule") or []
        scratch.daily_schedule = [ScheduleItem.from_dict(item) for item in daily_schedule_raw if isinstance(item, dict)]
        hourly_raw = raw.get("daily_schedule_hourly_original") or []
        scratch.daily_schedule_hourly_original = [
            ScheduleItem.from_dict(item) for item in hourly_raw if isinstance(item, dict)
        ]
        if not scratch.daily_schedule_hourly_original and scratch.daily_schedule:
            scratch.daily_schedule_hourly_original = list(scratch.daily_schedule)
        planned_path_raw = raw.get("planned_path") or []
        planned_path: list[tuple[int, int]] = []
        for point in planned_path_raw:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                planned_path.append((int(point[0]), int(point[1])))
            except Exception:
                continue
        scratch.planned_path = planned_path
        chat_raw = raw.get("chat") or []
        chat_lines: list[tuple[str, str]] = []
        for row in chat_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                continue
            chat_lines.append((str(row[0]), str(row[1])))
        scratch.chat = chat_lines
        buffer_raw = raw.get("chatting_with_buffer") or {}
        buffer_out: dict[str, int] = {}
        if isinstance(buffer_raw, dict):
            for key, value in buffer_raw.items():
                try:
                    buffer_out[str(key)] = int(value)
                except Exception:
                    continue
        scratch.chatting_with_buffer = buffer_out
        return scratch


@dataclass
class SpatialMemoryTree:
    """Spatial memory tree (world -> sector -> arena -> objects)."""

    tree: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)

    def observe(self, *, world: str, sector: str, arena: str, game_object: str) -> None:
        w = str(world or "town").strip().lower() or "town"
        s = str(sector or "common").strip().lower() or "common"
        a = str(arena or "open").strip().lower() or "open"
        obj = str(game_object or "spot").strip().lower() or "spot"
        if w not in self.tree:
            self.tree[w] = {}
        if s not in self.tree[w]:
            self.tree[w][s] = {}
        if a not in self.tree[w][s]:
            self.tree[w][s][a] = []
        if obj not in self.tree[w][s][a]:
            self.tree[w][s][a].append(obj)

    def observe_place_label(self, place_label: str) -> None:
        normalized = str(place_label or "").strip()
        if not normalized:
            return
        name = normalized
        affordances: list[str] = []
        if "(" in normalized and normalized.endswith(")"):
            name, raw = normalized.rsplit("(", 1)
            name = name.strip()
            affordances = [piece.strip().lower() for piece in raw[:-1].split(",") if piece.strip()]
        sector = affordances[0] if affordances else "common"
        obj = affordances[-1] if affordances else name.lower().replace(" ", "_")
        arena = name.lower().replace(" ", "_")
        self.observe(world="town", sector=sector, arena=arena, game_object=obj)

    def as_dict(self) -> dict[str, object]:
        return {
            world: {
                sector: {arena: list(objects) for arena, objects in arenas.items()}
                for sector, arenas in sectors.items()
            }
            for world, sectors in self.tree.items()
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SpatialMemoryTree":
        if not isinstance(raw, dict):
            return cls()
        tree: dict[str, dict[str, dict[str, list[str]]]] = {}
        for world, sectors_raw in raw.items():
            if not isinstance(sectors_raw, dict):
                continue
            world_key = str(world)
            tree[world_key] = {}
            for sector, arenas_raw in sectors_raw.items():
                if not isinstance(arenas_raw, dict):
                    continue
                sector_key = str(sector)
                tree[world_key][sector_key] = {}
                for arena, objects_raw in arenas_raw.items():
                    if not isinstance(objects_raw, list):
                        continue
                    tree[world_key][sector_key][str(arena)] = [str(item) for item in objects_raw if str(item).strip()]
        return cls(tree=tree)


@dataclass
class AgentMemory:
    """Per-agent memory state for simulation ticks."""

    agent_id: str
    agent_name: str
    scratch: AgentScratch = field(default_factory=AgentScratch)
    spatial: SpatialMemoryTree = field(default_factory=SpatialMemoryTree)
    nodes: list[MemoryNode] = field(default_factory=list)
    id_to_node: dict[str, MemoryNode] = field(default_factory=dict)
    seq_event: list[str] = field(default_factory=list)
    seq_thought: list[str] = field(default_factory=list)
    seq_chat: list[str] = field(default_factory=list)
    seq_reflection: list[str] = field(default_factory=list)
    kw_to_event: dict[str, list[str]] = field(default_factory=dict)
    kw_to_thought: dict[str, list[str]] = field(default_factory=dict)
    kw_to_chat: dict[str, list[str]] = field(default_factory=dict)
    kw_strength_event: dict[str, int] = field(default_factory=dict)
    kw_strength_thought: dict[str, int] = field(default_factory=dict)
    known_places: dict[str, int] = field(default_factory=dict)
    relationship_scores: dict[str, float] = field(default_factory=dict)

    @property
    def retention(self) -> int:
        return int(self.scratch.retention)

    @property
    def attention_bandwidth(self) -> int:
        return int(self.scratch.attention_bandwidth)

    @property
    def vision_radius(self) -> int:
        return int(self.scratch.vision_radius)

    @property
    def recency_decay(self) -> float:
        return float(self.scratch.recency_decay)

    @property
    def recency_weight(self) -> float:
        return float(self.scratch.recency_w)

    @property
    def relevance_weight(self) -> float:
        return float(self.scratch.relevance_w)

    @property
    def importance_weight(self) -> float:
        return float(self.scratch.importance_w)

    @property
    def reflection_trigger_max(self) -> int:
        return int(self.scratch.importance_trigger_max)

    @reflection_trigger_max.setter
    def reflection_trigger_max(self, value: int) -> None:
        self.scratch.importance_trigger_max = max(1, int(value))

    @property
    def reflection_trigger_curr(self) -> int:
        return int(self.scratch.importance_trigger_curr)

    @reflection_trigger_curr.setter
    def reflection_trigger_curr(self, value: int) -> None:
        self.scratch.importance_trigger_curr = int(value)

    @property
    def importance_events_since_reflection(self) -> int:
        return int(self.scratch.importance_ele_n)

    @importance_events_since_reflection.setter
    def importance_events_since_reflection(self, value: int) -> None:
        self.scratch.importance_ele_n = max(0, int(value))

    @classmethod
    def bootstrap(cls, *, agent_id: str, agent_name: str, step: int) -> "AgentMemory":
        memory = cls(agent_id=agent_id, agent_name=agent_name)
        memory.scratch.ensure_default_schedule(agent_name=agent_name)
        memory.add_node(
            kind="thought",
            step=step,
            subject=agent_name,
            predicate="joins",
            object="town",
            description=f"{agent_name} has joined the town and is starting their routine.",
            poignancy=2,
            evidence_ids=(),
            decrement_trigger=False,
        )
        return memory

    def _node_keywords(self, *, subject: str, predicate: str, object_value: str, description: str) -> tuple[str, ...]:
        tokens: list[str] = []
        tokens.extend(_tokenize(subject))
        tokens.extend(_tokenize(predicate))
        tokens.extend(_tokenize(object_value))
        tokens.extend(_tokenize(description))
        unique: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            unique.append(tok)
        return tuple(unique)

    def _index_node_keywords(self, *, node: MemoryNode) -> None:
        lowered = [kw.lower() for kw in node.keywords]
        if node.kind == "event":
            for kw in lowered:
                self.kw_to_event.setdefault(kw, []).insert(0, node.node_id)
                if f"{node.predicate} {node.object}" != "is idle":
                    self.kw_strength_event[kw] = int(self.kw_strength_event.get(kw, 0)) + 1
        elif node.kind in {"thought", "reflection"}:
            for kw in lowered:
                self.kw_to_thought.setdefault(kw, []).insert(0, node.node_id)
                if f"{node.predicate} {node.object}" != "is idle":
                    self.kw_strength_thought[kw] = int(self.kw_strength_thought.get(kw, 0)) + 1
        elif node.kind == "chat":
            for kw in lowered:
                self.kw_to_chat.setdefault(kw, []).insert(0, node.node_id)

    def add_node(
        self,
        *,
        kind: str,
        step: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        poignancy: int,
        evidence_ids: Iterable[str] = (),
        decrement_trigger: bool = True,
        address: str | None = None,
        expiration_step: int | None = None,
    ) -> MemoryNode:
        kind_normalized = str(kind or "event").strip().lower() or "event"
        evidence = tuple(str(i) for i in evidence_ids if str(i))
        depth = 0
        if kind_normalized in {"thought", "reflection"}:
            depth = 1
            if evidence:
                try:
                    depth += max(int(self.id_to_node[e].depth) for e in evidence if e in self.id_to_node)
                except Exception:
                    pass

        node = MemoryNode(
            node_id=f"mem_{uuid.uuid4().hex[:12]}",
            kind=kind_normalized,
            step=int(step),
            created_at=utc_now_iso(),
            subject=str(subject),
            predicate=str(predicate),
            object=str(object),
            description=str(description),
            poignancy=max(1, int(poignancy)),
            keywords=self._node_keywords(
                subject=str(subject),
                predicate=str(predicate),
                object_value=str(object),
                description=str(description),
            ),
            evidence_ids=evidence,
            last_accessed_step=int(step),
            depth=depth,
            embedding_key=str(description),
            embedding=_embedding_for_text(str(description)),
            address=address,
            expiration_step=expiration_step,
        )
        self.nodes.insert(0, node)
        self.id_to_node[node.node_id] = node
        if node.kind == "event":
            self.seq_event.insert(0, node.node_id)
        elif node.kind == "chat":
            self.seq_chat.insert(0, node.node_id)
        elif node.kind == "reflection":
            self.seq_reflection.insert(0, node.node_id)
            self.seq_thought.insert(0, node.node_id)
        else:
            self.seq_thought.insert(0, node.node_id)
        self._index_node_keywords(node=node)

        if decrement_trigger:
            self.scratch.importance_trigger_curr -= max(1, int(node.poignancy))
            self.scratch.importance_ele_n += 1
        return node

    def recent_signatures(self) -> set[tuple[str, str, str]]:
        signatures: set[tuple[str, str, str]] = set()
        ordered = list(self.seq_event) + list(self.seq_chat)
        for node_id in ordered:
            node = self.id_to_node.get(node_id)
            if not node:
                continue
            signatures.add((node.subject, node.predicate, node.object))
            if len(signatures) >= self.retention:
                break
        return signatures

    def set_position(self, *, x: int, y: int, step: int) -> None:
        self.scratch.curr_tile = (int(x), int(y))
        self.scratch.curr_step = int(step)

    def observe_place(self, place_label: str) -> None:
        normalized = str(place_label or "").strip()
        if not normalized:
            return
        self.known_places[normalized] = int(self.known_places.get(normalized, 0)) + 1
        self.spatial.observe_place_label(normalized)

    def set_action(
        self,
        *,
        step: int,
        description: str,
        address: str | None,
        duration: int | None = None,
        planned_path: list[tuple[int, int]] | None = None,
    ) -> None:
        self.scratch.act_start_step = int(step)
        self.scratch.act_description = str(description or "")
        self.scratch.act_address = str(address) if address else None
        self.scratch.act_duration = int(duration) if duration is not None else None
        if planned_path is not None:
            self.scratch.planned_path = list(planned_path)
            self.scratch.act_path_set = len(self.scratch.planned_path) > 0

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        value = str(scope or "").strip().lower()
        if value in {"long_term_plan", "long_term", "long"}:
            return "long_term_plan"
        if value in {"daily_plan", "daily"}:
            return "daily_plan"
        return "short_action"

    def _decomposed_schedule_action(self, *, step: int, scope: str) -> tuple[str, str | None]:
        schedule_item, _, offset = self.scratch.current_schedule_state(step=step, step_minutes=10)
        if schedule_item is None:
            return ("roaming", None)

        desc = str(schedule_item.description)
        duration = max(1, int(schedule_item.duration_mins))
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope in {"short_action", "daily_plan"} and duration >= 90:
            progress = float(offset) / float(duration)
            if progress < 0.33:
                return (f"prepare for {desc}", schedule_item.affordance_hint)
            if progress < 0.66:
                return (f"doing {desc}", schedule_item.affordance_hint)
            return (f"wrap up {desc}", schedule_item.affordance_hint)
        return (desc, schedule_item.affordance_hint)

    def maybe_refresh_plans(self, *, step: int, scope: str) -> list[MemoryNode]:
        created: list[MemoryNode] = []
        normalized_scope = self._normalize_scope(scope)

        if step >= int(self.scratch.next_daily_plan_step):
            action_desc, affordance = self._decomposed_schedule_action(step=step, scope=normalized_scope)
            created.append(
                self.add_node(
                    kind="thought",
                    step=step,
                    subject=self.agent_name,
                    predicate="plans_day",
                    object=affordance or "routine",
                    description=f"{self.agent_name} updates daily plan: {action_desc}.",
                    poignancy=3,
                    evidence_ids=tuple(self.seq_event[:2]),
                    decrement_trigger=False,
                    expiration_step=int(step) + (24 * 2),
                )
            )
            # ~2h cadence when each step approximates 10 mins.
            self.scratch.next_daily_plan_step = int(step) + 12

        if normalized_scope == "long_term_plan" or step >= int(self.scratch.next_long_term_plan_step):
            rel = self.top_relationships(limit=1)
            rel_target = str(rel[0]["agent_id"]) if rel else "town"
            top_places = sorted(self.known_places.items(), key=lambda item: (-item[1], item[0]))
            place_target = top_places[0][0] if top_places else "community"
            created.append(
                self.add_node(
                    kind="thought",
                    step=step,
                    subject=self.agent_name,
                    predicate="plans_long_term",
                    object=rel_target,
                    description=f"{self.agent_name} sets long-term focus around {rel_target} and {place_target}.",
                    poignancy=4,
                    evidence_ids=tuple(self.seq_thought[:2]),
                    decrement_trigger=False,
                    expiration_step=int(step) + (24 * 7),
                )
            )
            # ~12h cadence.
            self.scratch.next_long_term_plan_step = int(step) + 72

        return created

    def planned_action_description(self, *, step: int, scope: str, fallback: str | None = None) -> str:
        action_desc, _ = self._decomposed_schedule_action(step=step, scope=scope)
        if action_desc and action_desc != "roaming":
            return action_desc
        if fallback:
            return str(fallback)
        return "roaming"

    def add_social_interaction(
        self,
        *,
        step: int,
        target_agent_id: str,
        target_name: str,
        message: str,
        transcript: Iterable[tuple[str, str]] | None = None,
    ) -> MemoryNode:
        target_id = str(target_agent_id)
        target = str(target_name)
        self.relationship_scores[target_id] = float(self.relationship_scores.get(target_id, 0.0)) + 2.0
        self.scratch.chatting_with = target
        self.scratch.chatting_end_step = int(step) + 2
        self.scratch.chatting_with_buffer[target] = int(step) + self.vision_radius
        if transcript:
            for speaker, utterance in transcript:
                self.scratch.chat.append((str(speaker), str(utterance)))
        else:
            self.scratch.chat.append((self.agent_name, str(message)))
        return self.add_node(
            kind="chat",
            step=step,
            subject=self.agent_name,
            predicate="talks_with",
            object=target,
            description=f"{self.agent_name} talks with {target}: {message}",
            poignancy=4,
            evidence_ids=(),
            decrement_trigger=True,
        )

    def top_relationships(self, *, limit: int = 3) -> list[dict[str, object]]:
        ranked = sorted(
            self.relationship_scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        out: list[dict[str, object]] = []
        for agent_id, score in ranked[: max(1, int(limit))]:
            out.append({"agent_id": agent_id, "score": round(float(score), 3)})
        return out

    def _combined_nodes_for_retrieval(self) -> list[MemoryNode]:
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for node_id in list(self.seq_event) + list(self.seq_thought):
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered_ids.append(node_id)
        out: list[MemoryNode] = []
        for node_id in ordered_ids:
            node = self.id_to_node.get(node_id)
            if not node:
                continue
            if "idle" in str(node.embedding_key).lower():
                continue
            out.append(node)
        return out

    def retrieve_ranked(self, *, focal_text: str, step: int, limit: int = 8) -> list[dict[str, object]]:
        nodes = self._combined_nodes_for_retrieval()
        if not nodes:
            return []

        focal_embedding = _embedding_for_text(str(focal_text))
        recency_out: dict[str, float] = {}
        importance_out: dict[str, float] = {}
        relevance_out: dict[str, float] = {}

        for idx, node in enumerate(nodes, start=1):
            recency_out[node.node_id] = float(self.recency_decay) ** float(idx)
            importance_out[node.node_id] = float(node.poignancy)
            relevance_out[node.node_id] = _cosine_similarity(node.embedding, focal_embedding)

        recency_out = _normalize_dict_floats(recency_out, lo=0.0, hi=1.0)
        importance_out = _normalize_dict_floats(importance_out, lo=0.0, hi=1.0)
        relevance_out = _normalize_dict_floats(relevance_out, lo=0.0, hi=1.0)

        gw_recency, gw_relevance, gw_importance = (0.5, 3.0, 2.0)
        scored: list[tuple[float, MemoryNode]] = []
        for node in nodes:
            nid = node.node_id
            score = (
                self.scratch.recency_w * recency_out.get(nid, 0.0) * gw_recency
                + self.scratch.relevance_w * relevance_out.get(nid, 0.0) * gw_relevance
                + self.scratch.importance_w * importance_out.get(nid, 0.0) * gw_importance
            )
            scored.append((float(score), node))

        scored.sort(key=lambda pair: (-pair[0], -int(pair[1].step)))
        out: list[dict[str, object]] = []
        for score, node in scored[: max(1, int(limit))]:
            node.last_accessed_step = int(step)
            out.append(
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "description": node.description,
                    "score": round(float(score), 4),
                    "poignancy": int(node.poignancy),
                    "step": int(node.step),
                    "keywords": list(node.keywords),
                    "evidence_ids": list(node.evidence_ids),
                }
            )
        return out

    def _reflection_focal_points(self) -> list[str]:
        candidates = self.nodes[: max(8, min(32, len(self.nodes)))]
        if not candidates:
            return []
        keyword_counter: Counter[str] = Counter()
        for node in candidates:
            keyword_counter.update(node.keywords[:4])
        top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]
        if top_keywords:
            return top_keywords[:3]
        return [node.description for node in candidates[:3]]

    def _conversation_follow_up_thought(self, *, step: int) -> MemoryNode | None:
        end_step = self.scratch.chatting_end_step
        if end_step is None or int(step) < int(end_step):
            return None
        if not self.scratch.chat:
            return None
        convo_excerpt = " | ".join(f"{speaker}: {line}" for speaker, line in self.scratch.chat[-4:])
        node = self.add_node(
            kind="thought",
            step=step,
            subject=self.agent_name,
            predicate="memo_on_chat",
            object=self.scratch.chatting_with or "chat",
            description=f"{self.agent_name} notes from recent chat: {convo_excerpt}",
            poignancy=4,
            evidence_ids=tuple(self.seq_chat[:2]),
            decrement_trigger=False,
            expiration_step=int(step) + (24 * 6),
        )
        self.scratch.chat = []
        self.scratch.chatting_with = None
        self.scratch.chatting_end_step = None
        return node

    def maybe_reflect(self, *, step: int) -> list[MemoryNode]:
        out: list[MemoryNode] = []
        if self.scratch.importance_trigger_curr <= 0 and self.nodes:
            focal_points = self._reflection_focal_points()
            for focal in focal_points[:3]:
                retrieved = self.retrieve_ranked(focal_text=focal, step=step, limit=5)
                evidence_ids = tuple(str(item["node_id"]) for item in retrieved[:3] if item.get("node_id"))
                out.append(
                    self.add_node(
                        kind="reflection",
                        step=step,
                        subject=self.agent_name,
                        predicate="reflects_on",
                        object=focal,
                        description=f"{self.agent_name} reflects on {focal} and updates priorities.",
                        poignancy=6,
                        evidence_ids=evidence_ids,
                        decrement_trigger=False,
                        expiration_step=int(step) + (24 * 7),
                    )
                )
            top_rel = self.top_relationships(limit=1)
            if top_rel:
                rel_id = str(top_rel[0]["agent_id"])
                out.append(
                    self.add_node(
                        kind="thought",
                        step=step,
                        subject=self.agent_name,
                        predicate="plans_with",
                        object=rel_id,
                        description=f"{self.agent_name} plans to coordinate with {rel_id} soon.",
                        poignancy=5,
                        evidence_ids=tuple(node.node_id for node in out[:1]),
                        decrement_trigger=False,
                        expiration_step=int(step) + (24 * 7),
                    )
                )
            self.scratch.importance_trigger_curr = int(self.scratch.importance_trigger_max)
            self.scratch.importance_ele_n = 0

        follow_up = self._conversation_follow_up_thought(step=step)
        if follow_up is not None:
            out.append(follow_up)
        return out

    def summary(self) -> dict[str, object]:
        event_count = len(self.seq_event)
        thought_count = len(self.seq_thought)
        chat_count = len(self.seq_chat)
        reflection_count = len(self.seq_reflection)
        return {
            "total_nodes": len(self.nodes),
            "event_count": int(event_count),
            "thought_count": int(thought_count),
            "chat_count": int(chat_count),
            "reflection_count": int(reflection_count),
            "known_places": len(self.known_places),
            "top_relationships": self.top_relationships(limit=3),
            "reflection_trigger_curr": int(self.scratch.importance_trigger_curr),
            "reflection_trigger_max": int(self.scratch.importance_trigger_max),
            "daily_schedule_items": len(self.scratch.daily_schedule),
            "active_action": self.scratch.act_description,
        }

    def snapshot(self, *, limit: int = 40) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "known_places": dict(sorted(self.known_places.items(), key=lambda item: (-item[1], item[0]))),
            "relationship_scores": {
                key: round(float(value), 3)
                for key, value in sorted(self.relationship_scores.items(), key=lambda item: (-item[1], item[0]))
            },
            "spatial_memory": self.spatial.as_dict(),
            "scratch": self.scratch.as_dict(),
            "nodes": [node.as_dict() for node in self.nodes[: max(1, int(limit))]],
        }

    def export_state(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "known_places": dict(self.known_places),
            "relationship_scores": dict(self.relationship_scores),
            "spatial_memory": self.spatial.as_dict(),
            "scratch": self.scratch.as_dict(),
            "nodes": [node.as_dict() for node in self.nodes],
        }

    @classmethod
    def from_state(cls, raw: dict[str, Any]) -> "AgentMemory":
        if not isinstance(raw, dict):
            raise ValueError("AgentMemory state payload must be an object")
        agent_id = str(raw.get("agent_id") or "")
        agent_name = str(raw.get("agent_name") or agent_id or "Agent")
        memory = cls(agent_id=agent_id, agent_name=agent_name)
        memory.scratch = AgentScratch.from_dict(raw.get("scratch") or {})
        memory.spatial = SpatialMemoryTree.from_dict(raw.get("spatial_memory") or {})
        known_places_raw = raw.get("known_places") or {}
        if isinstance(known_places_raw, dict):
            memory.known_places = {
                str(key): int(value)
                for key, value in known_places_raw.items()
                if str(key).strip()
            }
        relationship_raw = raw.get("relationship_scores") or {}
        if isinstance(relationship_raw, dict):
            memory.relationship_scores = {
                str(key): float(value)
                for key, value in relationship_raw.items()
                if str(key).strip()
            }
        nodes_raw = raw.get("nodes") or []
        nodes: list[MemoryNode] = []
        for item in nodes_raw:
            if not isinstance(item, dict):
                continue
            try:
                nodes.append(MemoryNode.from_dict(item))
            except Exception:
                continue
        memory.nodes = nodes
        memory.id_to_node = {node.node_id: node for node in memory.nodes}
        memory.seq_event = [node.node_id for node in memory.nodes if node.kind == "event"]
        memory.seq_chat = [node.node_id for node in memory.nodes if node.kind == "chat"]
        memory.seq_reflection = [node.node_id for node in memory.nodes if node.kind == "reflection"]
        memory.seq_thought = [node.node_id for node in memory.nodes if node.kind in {"thought", "reflection"}]
        memory.kw_to_event = {}
        memory.kw_to_thought = {}
        memory.kw_to_chat = {}
        memory.kw_strength_event = {}
        memory.kw_strength_thought = {}
        for node in reversed(memory.nodes):
            memory._index_node_keywords(node=node)
        memory.scratch.ensure_default_schedule(agent_name=memory.agent_name)
        if memory.scratch.importance_trigger_curr > memory.scratch.importance_trigger_max:
            memory.scratch.importance_trigger_curr = int(memory.scratch.importance_trigger_max)
        return memory
