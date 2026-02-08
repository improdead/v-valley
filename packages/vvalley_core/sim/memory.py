"""Lightweight memory system inspired by Generative Agents, adapted for V-Valley."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
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
    tokens = [tok for tok in _TOKEN_RE.findall(lowered) if tok not in _STOPWORDS]
    return tuple(tokens)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


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
        }


@dataclass
class AgentMemory:
    """Per-agent memory state for simulation ticks."""

    agent_id: str
    agent_name: str
    retention: int = 6
    attention_bandwidth: int = 3
    vision_radius: int = 4
    recency_decay: float = 0.93
    recency_weight: float = 0.5
    relevance_weight: float = 3.0
    importance_weight: float = 2.0
    reflection_trigger_max: int = 28
    reflection_trigger_curr: int = 28
    importance_events_since_reflection: int = 0
    nodes: list[MemoryNode] = field(default_factory=list)
    known_places: dict[str, int] = field(default_factory=dict)
    relationship_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def bootstrap(cls, *, agent_id: str, agent_name: str, step: int) -> "AgentMemory":
        memory = cls(agent_id=agent_id, agent_name=agent_name)
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
    ) -> MemoryNode:
        node = MemoryNode(
            node_id=f"mem_{uuid.uuid4().hex[:12]}",
            kind=str(kind),
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
            evidence_ids=tuple(str(i) for i in evidence_ids),
            last_accessed_step=int(step),
        )
        self.nodes.insert(0, node)
        if decrement_trigger:
            self.reflection_trigger_curr -= max(1, int(node.poignancy))
            self.importance_events_since_reflection += 1
        return node

    def recent_signatures(self) -> set[tuple[str, str, str]]:
        signatures: set[tuple[str, str, str]] = set()
        for node in self.nodes:
            if node.kind not in {"event", "chat"}:
                continue
            signatures.add((node.subject, node.predicate, node.object))
            if len(signatures) >= self.retention:
                break
        return signatures

    def observe_place(self, place_label: str) -> None:
        normalized = str(place_label or "").strip()
        if not normalized:
            return
        self.known_places[normalized] = int(self.known_places.get(normalized, 0)) + 1

    def add_social_interaction(
        self,
        *,
        step: int,
        target_agent_id: str,
        target_name: str,
        message: str,
    ) -> MemoryNode:
        self.relationship_scores[target_agent_id] = float(self.relationship_scores.get(target_agent_id, 0.0)) + 2.0
        return self.add_node(
            kind="chat",
            step=step,
            subject=self.agent_name,
            predicate="talks_with",
            object=target_name,
            description=message,
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

    def retrieve_ranked(self, *, focal_text: str, step: int, limit: int = 8) -> list[dict[str, object]]:
        if not self.nodes:
            return []
        focal_tokens = set(_tokenize(focal_text))
        scored: list[tuple[float, MemoryNode]] = []
        for idx, node in enumerate(self.nodes):
            recency = float(self.recency_decay) ** float(idx)
            importance = min(1.0, float(node.poignancy) / 10.0)
            node_tokens = set(node.keywords) if node.keywords else set(_tokenize(node.description))
            relevance = _jaccard(focal_tokens, node_tokens)
            score = (
                self.recency_weight * recency
                + self.relevance_weight * relevance
                + self.importance_weight * importance
            )
            scored.append((score, node))
        scored.sort(key=lambda item: (-float(item[0]), -int(item[1].step)))
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
                }
            )
        return out

    def maybe_reflect(self, *, step: int) -> list[MemoryNode]:
        if self.reflection_trigger_curr > 0 or not self.nodes:
            return []

        window = self.nodes[: max(6, min(20, len(self.nodes)))]
        keyword_counter: Counter[str] = Counter()
        evidence_ids: list[str] = []
        for node in window:
            evidence_ids.append(node.node_id)
            keyword_counter.update(node.keywords[:4])

        top_keywords = [kw for kw, _ in keyword_counter.most_common(3)]
        theme = ", ".join(top_keywords) if top_keywords else "the day"
        reflections: list[MemoryNode] = []
        reflections.append(
            self.add_node(
                kind="reflection",
                step=step,
                subject=self.agent_name,
                predicate="reflects_on",
                object=theme,
                description=f"{self.agent_name} reflects on {theme} and adjusts priorities.",
                poignancy=6,
                evidence_ids=evidence_ids[:4],
                decrement_trigger=False,
            )
        )

        top_rel = self.top_relationships(limit=1)
        if top_rel:
            top_agent = str(top_rel[0]["agent_id"])
            reflections.append(
                self.add_node(
                    kind="thought",
                    step=step,
                    subject=self.agent_name,
                    predicate="plans_with",
                    object=top_agent,
                    description=f"{self.agent_name} plans to spend more time coordinating with {top_agent}.",
                    poignancy=5,
                    evidence_ids=(reflections[0].node_id,),
                    decrement_trigger=False,
                )
            )

        self.reflection_trigger_curr = int(self.reflection_trigger_max)
        self.importance_events_since_reflection = 0
        return reflections

    def summary(self) -> dict[str, object]:
        event_count = 0
        thought_count = 0
        chat_count = 0
        reflection_count = 0
        for node in self.nodes:
            if node.kind == "event":
                event_count += 1
            elif node.kind == "chat":
                chat_count += 1
            elif node.kind == "reflection":
                reflection_count += 1
            elif node.kind == "thought":
                thought_count += 1
        return {
            "total_nodes": len(self.nodes),
            "event_count": event_count,
            "thought_count": thought_count,
            "chat_count": chat_count,
            "reflection_count": reflection_count,
            "known_places": len(self.known_places),
            "top_relationships": self.top_relationships(limit=3),
            "reflection_trigger_curr": int(self.reflection_trigger_curr),
            "reflection_trigger_max": int(self.reflection_trigger_max),
        }

    def snapshot(self, *, limit: int = 40) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "known_places": dict(sorted(self.known_places.items(), key=lambda item: (-item[1], item[0]))),
            "relationship_scores": {
                k: round(float(v), 3)
                for k, v in sorted(self.relationship_scores.items(), key=lambda item: (-item[1], item[0]))
            },
            "nodes": [node.as_dict() for node in self.nodes[: max(1, int(limit))]],
        }

