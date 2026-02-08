"""Legacy generative_agents integration endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.vvalley_core.maps.map_utils import map_sha256, parse_location_objects
from packages.vvalley_core.maps.nav import build_nav_payload
from packages.vvalley_core.maps.validator import validate_map
from packages.vvalley_core.sim.legacy_map_import import build_the_ville_map
from packages.vvalley_core.sim.legacy_replay import list_legacy_simulations, replay_events

from ..storage.map_versions import next_version, publish_version
from ..storage.map_versions import activate_version


router = APIRouter(prefix="/api/v1/legacy", tags=["legacy"])
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


class ReplayRequest(BaseModel):
    start_step: int = Field(default=0, ge=0)
    max_steps: int = Field(default=120, ge=1, le=5000)


class ImportTheVilleRequest(BaseModel):
    town_id: str = Field(default="the_ville_legacy", min_length=2, max_length=64)
    map_name: Optional[str] = None
    notes: Optional[str] = None
    publish_version_record: bool = True
    activate_after_publish: bool = True


def _rel_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path.resolve())


def _normalize_version(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["is_active"] = bool(out.get("is_active", 0))
    out["version"] = int(out["version"])
    return out


@router.get("/simulations")
def legacy_simulations() -> dict[str, Any]:
    sims = list_legacy_simulations()
    return {"count": len(sims), "simulations": sims}


@router.post("/simulations/{simulation_id}/events")
def legacy_simulation_events(simulation_id: str, req: ReplayRequest) -> dict[str, Any]:
    try:
        return replay_events(
            simulation_id=simulation_id,
            start_step=req.start_step,
            max_steps=req.max_steps,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Legacy simulation not found: {simulation_id}")


@router.post("/import-the-ville")
def import_the_ville(req: ImportTheVilleRequest) -> dict[str, Any]:
    source_visual_path = (
        WORKSPACE_ROOT
        / "generative_agents"
        / "environment"
        / "frontend_server"
        / "static_dirs"
        / "assets"
        / "the_ville"
        / "visuals"
        / "the_ville2.png"
    )
    map_data = build_the_ville_map()
    errors, warnings, summary = validate_map(map_data)
    if errors:
        return {
            "ok": False,
            "town_id": req.town_id,
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }

    town_dir = WORKSPACE_ROOT / "assets" / "maps" / "v_valley" / req.town_id
    town_dir.mkdir(parents=True, exist_ok=True)
    draft_map_path = town_dir / "legacy_the_ville.draft.map.json"
    draft_map_path.write_text(json.dumps(map_data, indent=2), encoding="utf-8")

    if not req.publish_version_record:
        return {
            "ok": True,
            "town_id": req.town_id,
            "map_path": str(draft_map_path),
            "warnings": warnings,
            "summary": summary,
            "published": False,
            "source_visual_path": str(source_visual_path),
        }

    version_number = next_version(req.town_id)
    map_snapshot = town_dir / f"v{version_number}.map.json"
    nav_snapshot = town_dir / f"v{version_number}.nav.json"

    map_snapshot.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
    source_sha = map_sha256(map_snapshot)
    nav_errors, nav_warnings, nav_summary, nav_payload = build_nav_payload(
        map_data,
        source_map_file=_rel_workspace(map_snapshot),
        source_sha256=source_sha,
    )
    if nav_errors:
        return {
            "ok": False,
            "town_id": req.town_id,
            "errors": nav_errors,
            "warnings": nav_warnings,
            "summary": nav_summary,
        }
    nav_snapshot.write_text(json.dumps(nav_payload, indent=2), encoding="utf-8")

    affordance_rows: list[dict[str, Any]] = []
    for location in parse_location_objects(map_data):
        for affordance in location.affordances:
            affordance_rows.append(
                {
                    "location_name": location.name,
                    "tile_x": location.x,
                    "tile_y": location.y,
                    "affordance": affordance,
                    "metadata_json": None,
                }
            )

    record = publish_version(
        town_id=req.town_id,
        version=version_number,
        map_name=req.map_name or f"legacy_the_ville_v{version_number}",
        map_json_path=_rel_workspace(map_snapshot),
        nav_data_path=_rel_workspace(nav_snapshot),
        source_sha256=source_sha,
        affordances=affordance_rows,
        notes=req.notes or "Imported from generative_agents the_ville matrix assets",
    )
    if req.activate_after_publish:
        active = activate_version(req.town_id, version_number)
        if active:
            record = active

    return {
        "ok": True,
        "town_id": req.town_id,
        "published": True,
        "draft_map_path": str(draft_map_path),
        "map_path": str(map_snapshot),
        "nav_path": str(nav_snapshot),
        "warnings": warnings,
        "summary": summary,
        "version": _normalize_version(record),
        "source_visual_path": str(source_visual_path),
    }
