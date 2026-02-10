"""Town directory and observer-mode endpoints."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ..storage.agents import count_active_agents_in_town
from ..storage.map_versions import (
    get_active_version,
    get_affordances,
    list_town_ids,
    list_versions,
)

logger = logging.getLogger("vvalley_api.towns")


router = APIRouter(prefix="/api/v1/towns", tags=["towns"])


def _title_from_town_id(town_id: str) -> str:
    cleaned = town_id.replace("-", " ").replace("_", " ")
    return " ".join(part.capitalize() for part in cleaned.split())


def _normalize_version(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["version"] = int(out["version"])
    out["is_active"] = bool(out.get("is_active", 0))
    return out


def _town_summary(town_id: str) -> dict[str, Any]:
    active = get_active_version(town_id)
    active_norm: Optional[dict[str, Any]] = None
    if active:
        active_norm = _normalize_version(active)
    population = count_active_agents_in_town(town_id)
    return {
        "town_id": town_id,
        "name": _title_from_town_id(town_id),
        "description": "Autonomous AI town simulation",
        "status": "online",
        "mode": "public",
        "population": population,
        "max_agents": 25,
        "available_slots": max(0, 25 - population),
        "is_full": population >= 25,
        "active_map_version": active_norm,
        "observer_mode": True,
    }


@router.get("")
@router.get("/")
def list_towns() -> dict[str, Any]:
    logger.info("[TOWNS] List towns request")
    town_ids = list_town_ids()
    towns = [_town_summary(town_id) for town_id in town_ids]
    logger.info("[TOWNS] Towns listed: count=%d", len(towns))
    return {"count": len(towns), "towns": towns}


@router.get("/{town_id}")
def get_town(town_id: str) -> dict[str, Any]:
    logger.info("[TOWNS] Get town request: town_id='%s'", town_id)
    versions = [_normalize_version(v) for v in list_versions(town_id)]
    if not versions:
        logger.warning("[TOWNS] Town not found: town_id='%s'", town_id)
        raise HTTPException(status_code=404, detail=f"Town not found: {town_id}")

    town = _town_summary(town_id)
    active = town.get("active_map_version")
    if active:
        active["affordances"] = get_affordances(active["id"])

    logger.info("[TOWNS] Town retrieved: town_id='%s', versions=%d", town_id, len(versions))
    return {
        "town": town,
        "map_versions": versions,
    }
