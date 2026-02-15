"""Shared Elo helpers for multi-player scenario matches."""

from __future__ import annotations

import math
from typing import Any


def k_factor(games_played: int) -> int:
    return 40 if int(games_played) < 10 else 20


def expected_score(r_self: int, r_other: int) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (float(r_other) - float(r_self)) / 400.0))


def compute_pairwise_elo_updates(
    *,
    ratings: dict[str, dict[str, Any]],
    participant_ids: list[str],
    winners: set[str],
    draws: bool,
) -> dict[str, dict[str, int]]:
    deltas_float: dict[str, float] = {agent_id: 0.0 for agent_id in participant_ids}

    for i, agent_id in enumerate(participant_ids):
        for j, other_id in enumerate(participant_ids):
            if i == j:
                continue
            rating_i = int(ratings[agent_id]["rating"])
            rating_j = int(ratings[other_id]["rating"])
            expected_i = expected_score(rating_i, rating_j)
            if draws:
                s_i = 0.5
            else:
                s_i = 1.0 if agent_id in winners and other_id not in winners else 0.0
                if (agent_id in winners) == (other_id in winners):
                    s_i = 0.5
            k = k_factor(int(ratings[agent_id].get("games_played") or 0))
            deltas_float[agent_id] += float(k) * (float(s_i) - expected_i)

    out: dict[str, dict[str, int]] = {}
    divisor = max(1.0, float(len(participant_ids) - 1))
    for agent_id in participant_ids:
        current = ratings[agent_id]
        before = int(current["rating"])
        delta = int(round(deltas_float[agent_id] / divisor))
        after = max(0, before + delta)
        out[agent_id] = {
            "before": before,
            "delta": delta,
            "after": after,
            "games_played": int(current.get("games_played") or 0) + 1,
            "wins": int(current.get("wins") or 0) + (1 if (not draws and agent_id in winners) else 0),
            "losses": int(current.get("losses") or 0) + (1 if (not draws and agent_id not in winners) else 0),
            "draws": int(current.get("draws") or 0) + (1 if draws else 0),
            "peak_rating": max(int(current.get("peak_rating") or before), after),
        }
    return out
