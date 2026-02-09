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
    decomposed: bool = False

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "description": self.description,
            "duration_mins": int(self.duration_mins),
            "affordance_hint": self.affordance_hint,
        }
        if self.decomposed:
            d["decomposed"] = True
        return d

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
            decomposed=bool(raw.get("decomposed", False)),
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
    act_pronunciatio: str | None = None
    act_event: tuple[str, str, str] | None = None
    planned_path: list[tuple[int, int]] = field(default_factory=list)

    chatting_with: str | None = None
    chat: list[tuple[str, str]] = field(default_factory=list)
    chatting_with_buffer: dict[str, int] = field(default_factory=dict)
    chatting_end_step: int | None = None

    currently: str | None = None
    last_day_index: int = -1

    # --- ISS (Identity Stable Set) fields ---
    iss_name: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    age: int | None = None
    innate: str | None = None
    learned: str | None = None
    lifestyle: str | None = None
    living_area: str | None = None

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

    def get_str_iss(self) -> str:
        """Return a formatted Identity Stable Set string for LLM prompt injection."""
        parts: list[str] = []
        display_name = self.iss_name or "Unknown"
        parts.append(f"Name: {display_name}")
        if self.age is not None:
            parts.append(f"Age: {self.age}")
        if self.innate:
            parts.append(f"Innate traits: {self.innate}")
        if self.learned:
            parts.append(f"Background: {self.learned}")
        if self.lifestyle:
            parts.append(f"Lifestyle: {self.lifestyle}")
        if self.living_area:
            parts.append(f"Living area: {self.living_area}")
        if self.currently:
            parts.append(f"Currently: {self.currently}")
        return "\n".join(parts)

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
            "act_pronunciatio": self.act_pronunciatio,
            "act_event": list(self.act_event) if self.act_event else None,
            "planned_path": [list(i) for i in self.planned_path],
            "chatting_with": self.chatting_with,
            "chat": [[speaker, utterance] for speaker, utterance in self.chat],
            "chatting_with_buffer": dict(self.chatting_with_buffer),
            "chatting_end_step": self.chatting_end_step,
            "currently": self.currently,
            "last_day_index": int(self.last_day_index),
            "iss_name": self.iss_name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "innate": self.innate,
            "learned": self.learned,
            "lifestyle": self.lifestyle,
            "living_area": self.living_area,
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
            act_pronunciatio=(str(raw.get("act_pronunciatio")) if raw.get("act_pronunciatio") is not None else None),
            chatting_with=(str(raw.get("chatting_with")) if raw.get("chatting_with") is not None else None),
            chatting_end_step=(int(raw["chatting_end_step"]) if raw.get("chatting_end_step") is not None else None),
            currently=(str(raw.get("currently")) if raw.get("currently") is not None else None),
            last_day_index=int(raw.get("last_day_index") or -1),
            iss_name=(str(raw.get("iss_name")) if raw.get("iss_name") is not None else None),
            first_name=(str(raw.get("first_name")) if raw.get("first_name") is not None else None),
            last_name=(str(raw.get("last_name")) if raw.get("last_name") is not None else None),
            age=(int(raw["age"]) if raw.get("age") is not None else None),
            innate=(str(raw.get("innate")) if raw.get("innate") is not None else None),
            learned=(str(raw.get("learned")) if raw.get("learned") is not None else None),
            lifestyle=(str(raw.get("lifestyle")) if raw.get("lifestyle") is not None else None),
            living_area=(str(raw.get("living_area")) if raw.get("living_area") is not None else None),
            next_daily_plan_step=max(0, int(raw.get("next_daily_plan_step") or 0)),
            next_long_term_plan_step=max(0, int(raw.get("next_long_term_plan_step") or 0)),
        )
        curr_tile = raw.get("curr_tile")
        if isinstance(curr_tile, (list, tuple)) and len(curr_tile) == 2:
            try:
                scratch.curr_tile = (int(curr_tile[0]), int(curr_tile[1]))
            except Exception:
                scratch.curr_tile = None
        act_event_raw = raw.get("act_event")
        if isinstance(act_event_raw, (list, tuple)) and len(act_event_raw) == 3:
            scratch.act_event = (str(act_event_raw[0]), str(act_event_raw[1]), str(act_event_raw[2]))
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


CognitionFn = Any  # Optional: CognitionPlanner instance or None


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
    cognition: CognitionFn = field(default=None, repr=False)
    town_id: str | None = field(default=None, repr=False)

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
    def bootstrap(cls, *, agent_id: str, agent_name: str, step: int, persona: dict[str, Any] | None = None) -> "AgentMemory":
        memory = cls(agent_id=agent_id, agent_name=agent_name)
        memory.scratch.ensure_default_schedule(agent_name=agent_name)

        if persona and isinstance(persona, dict):
            # Persona-aware bootstrap — seed from GA-format persona
            innate = persona.get("innate", "")
            learned = persona.get("learned", "")
            lifestyle = persona.get("lifestyle", "")
            daily_req = persona.get("daily_req")
            currently = persona.get("currently")

            # Populate ISS fields
            p_first = str(persona.get("first_name", "")).strip()
            p_last = str(persona.get("last_name", "")).strip()
            memory.scratch.first_name = p_first or None
            memory.scratch.last_name = p_last or None
            memory.scratch.iss_name = f"{p_first} {p_last}".strip() or agent_name
            try:
                memory.scratch.age = int(persona["age"]) if persona.get("age") is not None else None
            except (ValueError, TypeError):
                memory.scratch.age = None
            memory.scratch.innate = str(innate).strip() or None
            memory.scratch.learned = str(learned).strip() or None
            memory.scratch.lifestyle = str(lifestyle).strip() or None
            memory.scratch.living_area = str(persona.get("living_area", "")).strip() or None

            if isinstance(daily_req, list) and daily_req:
                memory.scratch.daily_req = [str(r) for r in daily_req if str(r).strip()]

            if currently and str(currently).strip():
                memory.scratch.currently = str(currently).strip()

            if learned and str(learned).strip():
                memory.scratch.long_term_goals = [str(learned).strip()]

            # Seed identity thought
            identity_parts = [agent_name]
            if innate:
                identity_parts.append(f"is {innate}")
            if lifestyle:
                identity_parts.append(lifestyle)
            identity_desc = ". ".join(identity_parts)
            memory.add_node(
                kind="thought",
                step=step,
                subject=agent_name,
                predicate="is",
                object="a town resident",
                description=identity_desc,
                poignancy=5,
                evidence_ids=(),
                decrement_trigger=False,
            )

            # Seed backstory memory
            if learned and str(learned).strip():
                memory.add_node(
                    kind="thought",
                    step=step,
                    subject=agent_name,
                    predicate="has background",
                    object="life story",
                    description=str(learned).strip(),
                    poignancy=4,
                    evidence_ids=(),
                    decrement_trigger=False,
                )
        else:
            # Generic bootstrap — no persona
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
                self.kw_to_event.setdefault(kw, []).append(node.node_id)
                if f"{node.predicate} {node.object}" != "is idle":
                    self.kw_strength_event[kw] = int(self.kw_strength_event.get(kw, 0)) + 1
        elif node.kind in {"thought", "reflection"}:
            for kw in lowered:
                self.kw_to_thought.setdefault(kw, []).append(node.node_id)
                if f"{node.predicate} {node.object}" != "is idle":
                    self.kw_strength_thought[kw] = int(self.kw_strength_thought.get(kw, 0)) + 1
        elif node.kind == "chat":
            for kw in lowered:
                self.kw_to_chat.setdefault(kw, []).append(node.node_id)

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
        self.nodes.append(node)
        self.id_to_node[node.node_id] = node
        if node.kind == "event":
            self.seq_event.append(node.node_id)
        elif node.kind == "chat":
            self.seq_chat.append(node.node_id)
        elif node.kind == "reflection":
            self.seq_reflection.append(node.node_id)
            self.seq_thought.append(node.node_id)
        else:
            self.seq_thought.append(node.node_id)
        self._index_node_keywords(node=node)

        if decrement_trigger:
            self.scratch.importance_trigger_curr -= max(1, int(node.poignancy))
            self.scratch.importance_ele_n += 1
        return node

    def recent_signatures(self) -> set[tuple[str, str, str]]:
        signatures: set[tuple[str, str, str]] = set()
        ordered = list(reversed(self.seq_event)) + list(reversed(self.seq_chat))
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
        schedule_item, idx, offset = self.scratch.current_schedule_state(step=step, step_minutes=10)
        if schedule_item is None:
            return ("roaming", None)

        desc = str(schedule_item.description)
        duration = max(1, int(schedule_item.duration_mins))
        normalized_scope = self._normalize_scope(scope)

        # Try LLM decomposition for blocks >= 60 mins that haven't been decomposed yet
        if (
            self.cognition is not None
            and normalized_scope in {"short_action", "daily_plan"}
            and duration >= 60
            and desc.lower() not in {"sleep", "sleeping"}
            and not schedule_item.decomposed
        ):
            try:
                result = self.cognition.decompose_schedule({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "task_description": desc,
                    "total_duration_mins": duration,
                    "current_offset_mins": offset,
                    "daily_requirements": self.scratch.daily_req[:3],
                    "long_term_goals": self.scratch.long_term_goals[:2],
                })
                subtasks = result.get("subtasks") or []
                if subtasks and result.get("route") != "heuristic":
                    # Replace the current schedule item with decomposed subtasks
                    new_items: list[ScheduleItem] = []
                    total_sub = 0
                    for sub in subtasks:
                        if not isinstance(sub, dict):
                            continue
                        sub_desc = str(sub.get("description") or desc)
                        sub_dur = max(1, int(sub.get("duration_mins") or 10))
                        new_items.append(ScheduleItem(sub_desc, sub_dur, schedule_item.affordance_hint))
                        total_sub += sub_dur
                    if new_items and abs(total_sub - duration) <= duration * 0.2:
                        for item in new_items:
                            item.decomposed = True
                        self.scratch.daily_schedule[idx:idx + 1] = new_items
                        # Recalculate current position after decomposition
                        new_item, _, new_offset = self.scratch.current_schedule_state(step=step, step_minutes=10)
                        if new_item is not None:
                            return (str(new_item.description), new_item.affordance_hint)
            except Exception:
                pass

        # Deterministic fallback: 3-phase split for long activities
        if normalized_scope in {"short_action", "daily_plan"} and duration >= 90:
            progress = float(offset) / float(duration)
            if progress < 0.33:
                return (f"prepare for {desc}", schedule_item.affordance_hint)
            if progress < 0.66:
                return (f"doing {desc}", schedule_item.affordance_hint)
            return (f"wrap up {desc}", schedule_item.affordance_hint)
        return (desc, schedule_item.affordance_hint)

    def maybe_refresh_plans(self, *, step: int, scope: str, step_minutes: int = 10) -> list[MemoryNode]:
        created: list[MemoryNode] = []
        normalized_scope = self._normalize_scope(scope)

        # Day-boundary detection: generate new daily schedule via LLM
        day_minutes = max(1, self.scratch.day_minutes)
        current_day_index = (int(step) * int(step_minutes)) // day_minutes
        if current_day_index > self.scratch.last_day_index:
            self.scratch.last_day_index = current_day_index
            created.extend(self._generate_daily_schedule(step=step))

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
                    evidence_ids=tuple(self.seq_event[-2:]),
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
                    evidence_ids=tuple(self.seq_thought[-2:]),
                    decrement_trigger=False,
                    expiration_step=int(step) + (24 * 7),
                )
            )
            # ~12h cadence.
            self.scratch.next_long_term_plan_step = int(step) + 72

        return created

    def _generate_daily_schedule(self, *, step: int) -> list[MemoryNode]:
        """Generate a new daily schedule via LLM at day boundaries (GA's revise_identity + generate_hourly_schedule)."""
        created: list[MemoryNode] = []
        if self.cognition is None:
            return created

        try:
            # --- Detect first day (no previous schedule or very early step) ---
            is_first_day = not self.scratch.daily_schedule or step <= 1

            # --- Generate wake-up hour ---
            wake_up_hour = 7  # default
            if hasattr(self.cognition, "generate_wake_up_hour"):
                try:
                    wake_result = self.cognition.generate_wake_up_hour({
                        "agent_id": self.agent_id,
                        "agent_name": self.agent_name,
                        "lifestyle": self.scratch.lifestyle or "",
                        "identity": self.scratch.get_str_iss(),
                        "town_id": self.town_id or "",
                    })
                    wake_up_hour = max(0, min(23, int(wake_result.get("wake_up_hour") or 7)))
                except Exception:
                    pass

            # Gather context for schedule generation
            recent_thoughts = [
                node.description for node in self.nodes[-8:]
                if node.kind in {"thought", "reflection"} and "idle" not in node.embedding_key.lower()
            ][:4]
            recent_events = [
                node.description for node in self.nodes[-12:]
                if node.kind == "event" and "idle" not in node.embedding_key.lower()
            ][:4]
            top_rels = self.top_relationships(limit=3)
            rel_names = [str(r.get("agent_id", "")) for r in top_rels]

            # Retrieve plan-related memories for identity revision
            plan_memories: list[str] = []
            try:
                plan_results = self.retrieve_ranked(
                    focal_text=f"{self.agent_name} plan for today",
                    step=step,
                    limit=3,
                )
                for r in plan_results:
                    nid = str(r.get("node_id", ""))
                    node = self.id_to_node.get(nid)
                    if node is not None:
                        plan_memories.append(node.description)
            except Exception:
                pass

            # --- First day: use exploratory plan; subsequent days: full schedule ---
            if is_first_day and hasattr(self.cognition, "generate_first_daily_plan"):
                result = self.cognition.generate_first_daily_plan({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "identity": self.scratch.get_str_iss(),
                    "innate": self.scratch.innate or "",
                    "learned": self.scratch.learned or "",
                    "lifestyle": self.scratch.lifestyle or "",
                    "living_area": self.scratch.living_area or "",
                    "wake_up_hour": wake_up_hour,
                    "step": step,
                })
            else:
                result = self.cognition.generate_daily_schedule({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "identity": self.scratch.get_str_iss(),
                    "daily_requirements": self.scratch.daily_req[:5],
                    "long_term_goals": self.scratch.long_term_goals[:3],
                    "recent_thoughts": recent_thoughts,
                    "recent_events": recent_events,
                    "plan_memories": plan_memories,
                    "known_places": list(self.known_places.keys())[:6],
                    "relationships": rel_names,
                    "previous_currently": self.scratch.currently or "",
                    "wake_up_hour": wake_up_hour,
                    "step": step,
                })

            if result.get("route") == "heuristic":
                return created

            # Update daily schedule
            schedule_items = result.get("schedule") or []
            if schedule_items:
                new_schedule: list[ScheduleItem] = []
                total = 0
                for item in schedule_items:
                    if not isinstance(item, dict):
                        continue
                    desc = str(item.get("description") or "idle")
                    dur = max(1, int(item.get("duration_mins") or 60))
                    new_schedule.append(ScheduleItem(desc, dur))
                    total += dur

                # Prepend sleeping block based on wake-up hour
                sleep_mins = wake_up_hour * 60
                if sleep_mins > 0:
                    new_schedule.insert(0, ScheduleItem("sleeping", sleep_mins))
                    total += sleep_mins

                # Accept if total is within 10% of day length
                day_mins = self.scratch.day_minutes
                if new_schedule and abs(total - day_mins) <= day_mins * 0.1:
                    self.scratch.daily_schedule = new_schedule
                    self.scratch.daily_schedule_hourly_original = list(new_schedule)

            # Update identity/currently (GA's revise_identity)
            currently = str(result.get("currently") or "").strip()
            if currently:
                old_currently = self.scratch.currently or ""
                self.scratch.currently = currently
                created.append(
                    self.add_node(
                        kind="thought",
                        step=step,
                        subject=self.agent_name,
                        predicate="revises_identity",
                        object="self",
                        description=f"{self.agent_name}'s new focus: {currently}",
                        poignancy=5,
                        evidence_ids=tuple(self.seq_thought[-2:]),
                        decrement_trigger=False,
                        expiration_step=int(step) + (24 * 6),
                    )
                )

            # Update daily_req if LLM provides revised requirements
            new_daily_req = result.get("daily_req")
            if isinstance(new_daily_req, list) and new_daily_req:
                self.scratch.daily_req = [str(r) for r in new_daily_req[:8]]

            created.append(
                self.add_node(
                    kind="thought",
                    step=step,
                    subject=self.agent_name,
                    predicate="generates_daily_plan",
                    object="new_day",
                    description=f"{self.agent_name} plans a new day with {len(self.scratch.daily_schedule)} activities.",
                    poignancy=4,
                    evidence_ids=(),
                    decrement_trigger=False,
                    expiration_step=int(step) + (24 * 2),
                )
            )
        except Exception:
            pass

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
        self.scratch.chatting_with_buffer[target] = int(step) + 20
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

    def score_event_poignancy(self, *, description: str, event_type: str = "event") -> int:
        """Score an event's importance using LLM if available, else heuristic."""
        if self.cognition is not None:
            try:
                result = self.cognition.score_poignancy({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "description": description,
                    "event_type": event_type,
                })
                score = int(result.get("score") or 3)
                if result.get("route") != "heuristic":
                    return max(1, min(10, score))
            except Exception:
                pass
        # Deterministic fallback
        lower = description.lower()
        if "idle" in lower or "sleeping" in lower:
            return 1
        if any(kw in lower for kw in ("chat", "talk", "convers")):
            return 4
        if any(kw in lower for kw in ("reflect", "plan", "goal")):
            return 5
        return 3

    def recompose_schedule_after_interruption(
        self,
        *,
        step: int,
        inserted_description: str,
        inserted_duration_mins: int,
    ) -> None:
        """Recompose the remaining daily schedule after a chat or wait interruption."""
        _, idx, offset = self.scratch.current_schedule_state(step=step, step_minutes=10)
        if idx < 0 or idx >= len(self.scratch.daily_schedule):
            return

        current_item = self.scratch.daily_schedule[idx]
        remaining_in_block = max(0, int(current_item.duration_mins) - offset)

        # Insert the interruption activity into the schedule
        interrupted_items: list[ScheduleItem] = []
        if offset > 0:
            # Partial completion of current activity
            interrupted_items.append(ScheduleItem(
                f"interrupted: {current_item.description}",
                offset,
                current_item.affordance_hint,
            ))
        interrupted_items.append(ScheduleItem(
            inserted_description,
            inserted_duration_mins,
            "social",
        ))

        # Try LLM recomposition for remaining time
        if self.cognition is not None and remaining_in_block > 0:
            try:
                remaining_schedule = self.scratch.daily_schedule[idx + 1:]
                remaining_descs = [
                    f"{item.description} ({item.duration_mins}m)"
                    for item in remaining_schedule[:4]
                ]
                result = self.cognition.decompose_schedule({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "task_description": f"remainder of {current_item.description}",
                    "total_duration_mins": remaining_in_block,
                    "current_offset_mins": 0,
                    "interruption": inserted_description,
                    "upcoming_schedule": remaining_descs,
                })
                subtasks = result.get("subtasks") or []
                if subtasks and result.get("route") != "heuristic":
                    new_recomp_items: list[ScheduleItem] = []
                    total_recomp = 0
                    for sub in subtasks:
                        if not isinstance(sub, dict):
                            continue
                        sub_dur = max(1, int(sub.get("duration_mins") or remaining_in_block))
                        new_recomp_items.append(ScheduleItem(
                            str(sub.get("description") or current_item.description),
                            sub_dur,
                            current_item.affordance_hint,
                        ))
                        total_recomp += sub_dur
                    # Only accept if total duration is within 20% tolerance
                    if new_recomp_items and abs(total_recomp - remaining_in_block) <= remaining_in_block * 0.2:
                        for item in new_recomp_items:
                            item.decomposed = True
                        interrupted_items.extend(new_recomp_items)
                        self.scratch.daily_schedule[idx:idx + 1] = interrupted_items
                        return
            except Exception:
                pass

        # Deterministic fallback: just insert and keep remaining as-is
        if remaining_in_block > inserted_duration_mins:
            interrupted_items.append(ScheduleItem(
                f"resume {current_item.description}",
                remaining_in_block - inserted_duration_mins,
                current_item.affordance_hint,
            ))
        self.scratch.daily_schedule[idx:idx + 1] = interrupted_items

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
        for node_id in list(reversed(self.seq_event)) + list(reversed(self.seq_thought)):
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

    def _reflection_focal_points(self, *, step: int = 0) -> list[str]:
        candidates = self.nodes[-max(8, min(32, len(self.nodes))):]
        if not candidates:
            return []

        # Try LLM-based focal point generation
        if self.cognition is not None:
            try:
                recent_n = max(1, min(len(candidates), self.scratch.importance_ele_n or 8))
                recent = candidates[:recent_n]
                statements = "\n".join(
                    f"{i}. {node.description}" for i, node in enumerate(recent) if "idle" not in node.embedding_key.lower()
                )
                if statements:
                    result = self.cognition.generate_focal_points({
                        "agent_id": self.agent_id,
                        "agent_name": self.agent_name,
                        "town_id": self.town_id or "",
                        "recent_statements": statements,
                        "step": step,
                    })
                    questions = result.get("questions") or []
                    if questions and result.get("route") != "heuristic":
                        return [str(q) for q in questions[:3] if str(q).strip()]
            except Exception:
                pass

        # Deterministic fallback: keyword frequency
        keyword_counter: Counter[str] = Counter()
        for node in candidates:
            keyword_counter.update(node.keywords[:4])
        top_keywords = [kw for kw, _ in keyword_counter.most_common(5)]
        if top_keywords:
            return top_keywords[:3]
        return [node.description for node in candidates[:3]]

    def _conversation_follow_up_thought(self, *, step: int) -> list[MemoryNode]:
        end_step = self.scratch.chatting_end_step
        if end_step is None or int(step) < int(end_step):
            return []
        if not self.scratch.chat:
            return []

        partner = self.scratch.chatting_with or "chat"
        convo_excerpt = " | ".join(f"{speaker}: {line}" for speaker, line in self.scratch.chat[-4:])
        created: list[MemoryNode] = []
        chat_evidence = tuple(self.seq_chat[-2:])

        # Try LLM-based conversation summary
        if self.cognition is not None:
            try:
                full_transcript = "\n".join(f"{speaker}: {line}" for speaker, line in self.scratch.chat)
                result = self.cognition.summarize_conversation({
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "town_id": self.town_id or "",
                    "partner_name": partner,
                    "transcript": full_transcript,
                    "step": step,
                })
                memo = str(result.get("memo") or "").strip()
                planning = str(result.get("planning_thought") or "").strip()
                if memo and result.get("route") != "heuristic":
                    created.append(
                        self.add_node(
                            kind="thought",
                            step=step,
                            subject=self.agent_name,
                            predicate="memo_on_chat",
                            object=partner,
                            description=f"{self.agent_name} takeaway: {memo}",
                            poignancy=4,
                            evidence_ids=chat_evidence,
                            decrement_trigger=False,
                            expiration_step=int(step) + (24 * 6),
                        )
                    )
                if planning and result.get("route") != "heuristic":
                    created.append(
                        self.add_node(
                            kind="thought",
                            step=step,
                            subject=self.agent_name,
                            predicate="plans_from_chat",
                            object=partner,
                            description=f"For {self.agent_name}'s planning: {planning}",
                            poignancy=4,
                            evidence_ids=chat_evidence,
                            decrement_trigger=False,
                            expiration_step=int(step) + (24 * 4),
                        )
                    )
            except Exception:
                pass

        # Deterministic fallback if LLM didn't produce nodes
        if not created:
            created.append(
                self.add_node(
                    kind="thought",
                    step=step,
                    subject=self.agent_name,
                    predicate="memo_on_chat",
                    object=partner,
                    description=f"{self.agent_name} notes from recent chat: {convo_excerpt}",
                    poignancy=4,
                    evidence_ids=chat_evidence,
                    decrement_trigger=False,
                    expiration_step=int(step) + (24 * 6),
                )
            )

        self.scratch.chat = []
        self.scratch.chatting_with = None
        self.scratch.chatting_end_step = None
        return created

    def maybe_reflect(self, *, step: int) -> list[MemoryNode]:
        out: list[MemoryNode] = []
        if self.scratch.importance_trigger_curr <= 0 and self.nodes:
            focal_points = self._reflection_focal_points(step=step)
            for focal in focal_points[:3]:
                retrieved = self.retrieve_ranked(focal_text=focal, step=step, limit=5)
                retrieved_nodes = [self.id_to_node[str(item["node_id"])] for item in retrieved if item.get("node_id") and str(item["node_id"]) in self.id_to_node]

                # Try LLM insight generation
                if self.cognition is not None and retrieved_nodes:
                    try:
                        statements = "\n".join(
                            f"{i}. {node.description}" for i, node in enumerate(retrieved_nodes)
                        )
                        result = self.cognition.generate_reflection_insights({
                            "agent_id": self.agent_id,
                            "agent_name": self.agent_name,
                            "town_id": self.town_id or "",
                            "focal_point": focal,
                            "retrieved_statements": statements,
                            "step": step,
                        })
                        insights = result.get("insights") or []
                        if insights and result.get("route") != "heuristic":
                            for insight_data in insights[:3]:
                                if not isinstance(insight_data, dict):
                                    continue
                                insight_text = str(insight_data.get("insight") or "").strip()
                                if not insight_text:
                                    continue
                                evidence_indices = insight_data.get("evidence_indices") or [0]
                                evidence_ids = tuple(
                                    retrieved_nodes[idx].node_id
                                    for idx in evidence_indices
                                    if isinstance(idx, int) and 0 <= idx < len(retrieved_nodes)
                                ) or tuple(str(item["node_id"]) for item in retrieved[:1] if item.get("node_id"))
                                out.append(
                                    self.add_node(
                                        kind="reflection",
                                        step=step,
                                        subject=self.agent_name,
                                        predicate="reflects_on",
                                        object=focal,
                                        description=insight_text,
                                        poignancy=6,
                                        evidence_ids=evidence_ids,
                                        decrement_trigger=False,
                                        expiration_step=int(step) + (24 * 7),
                                    )
                                )
                            continue  # Skip deterministic fallback for this focal point
                    except Exception:
                        pass

                # Deterministic fallback
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

        follow_ups = self._conversation_follow_up_thought(step=step)
        out.extend(follow_ups)

        # Prune expired and over-cap nodes
        self.prune_expired(step=step)
        self._evict_to_cap(step=step)

        return out

    # ------------------------------------------------------------------
    # Memory management: pruning and eviction
    # ------------------------------------------------------------------

    MAX_MEMORY_NODES = 2000

    def prune_expired(self, *, step: int) -> int:
        """Remove nodes past their expiration_step. Returns count removed."""
        to_remove: set[str] = set()
        for node in self.nodes:
            if node.expiration_step is not None and node.expiration_step <= step:
                to_remove.add(node.node_id)
        if not to_remove:
            return 0
        return self._remove_nodes(to_remove)

    def _evict_to_cap(self, *, step: int) -> int:
        """If node count exceeds MAX_MEMORY_NODES, evict lowest-value nodes."""
        if len(self.nodes) <= self.MAX_MEMORY_NODES:
            return 0
        excess = len(self.nodes) - self.MAX_MEMORY_NODES
        # Build eviction candidates: never evict reflections or recent nodes
        recent_cutoff = max(0, step - 100)
        candidates: list[tuple[int, int, str]] = []
        for node in self.nodes:
            if node.kind == "reflection":
                continue
            if node.step >= recent_cutoff:
                continue
            candidates.append((node.poignancy, node.step, node.node_id))
        # Sort by lowest poignancy first, then oldest step first
        candidates.sort(key=lambda c: (c[0], c[1]))
        to_remove: set[str] = set()
        for _, _, node_id in candidates[:excess]:
            to_remove.add(node_id)
        if not to_remove:
            return 0
        return self._remove_nodes(to_remove)

    def _remove_nodes(self, node_ids: set[str]) -> int:
        """Remove a set of nodes from all data structures. Returns count removed."""
        self.nodes = [n for n in self.nodes if n.node_id not in node_ids]
        for nid in node_ids:
            self.id_to_node.pop(nid, None)
        self.seq_event = [nid for nid in self.seq_event if nid not in node_ids]
        self.seq_thought = [nid for nid in self.seq_thought if nid not in node_ids]
        self.seq_chat = [nid for nid in self.seq_chat if nid not in node_ids]
        self.seq_reflection = [nid for nid in self.seq_reflection if nid not in node_ids]
        # Rebuild keyword indices
        self.kw_to_event = {}
        self.kw_to_thought = {}
        self.kw_to_chat = {}
        self.kw_strength_event = {}
        self.kw_strength_thought = {}
        for node in self.nodes:
            self._index_node_keywords(node=node)
        return len(node_ids)

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
            "nodes": [node.as_dict() for node in self.nodes[-max(1, int(limit)):]],
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
        # Ensure oldest-first order (handles state files from before the ordering flip)
        memory.nodes.sort(key=lambda n: int(n.step))
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
        for node in memory.nodes:
            memory._index_node_keywords(node=node)
        memory.scratch.ensure_default_schedule(agent_name=memory.agent_name)
        if memory.scratch.importance_trigger_curr > memory.scratch.importance_trigger_max:
            memory.scratch.importance_trigger_curr = int(memory.scratch.importance_trigger_max)
        return memory
