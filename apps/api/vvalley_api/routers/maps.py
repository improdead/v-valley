"""Map validation and nav baking endpoints."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.vvalley_core.maps.customize import apply_customization, customize_map_file
from packages.vvalley_core.maps.templates import discover_templates
from packages.vvalley_core.maps.map_utils import (
    load_map,
    map_sha256,
    map_sha256_json,
    parse_location_objects,
)
from packages.vvalley_core.maps.nav import bake_nav_to_file, build_nav_payload
from packages.vvalley_core.maps.validator import validate_map

from ..storage.map_versions import (
    activate_version,
    get_active_version,
    get_affordances,
    get_version,
    list_versions,
    next_version,
    publish_version,
)

logger = logging.getLogger("vvalley_api.maps")

router = APIRouter(prefix="/api/v1/maps", tags=["maps"])
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
TEMPLATES_ROOT = WORKSPACE_ROOT / "assets" / "templates"
TOWN_MAP_ROOT = WORKSPACE_ROOT / "assets" / "maps" / "v_valley"
VALID_TOWN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$")


class ValidateMapRequest(BaseModel):
    map_data: dict[str, Any]


class ValidateMapFileRequest(BaseModel):
    map_path: str = Field(description="Absolute or workspace-relative path to map JSON")


class BakeMapRequest(BaseModel):
    map_data: dict[str, Any]
    source_map_file: str = "<inline>"


class BakeMapFileRequest(BaseModel):
    map_path: str
    out_path: Optional[str] = None
    allow_invalid: bool = False


class CustomizeMapFileRequest(BaseModel):
    map_path: str
    config_path: str
    out_path: Optional[str] = None
    validate_after: bool = True


class PublishMapVersionRequest(BaseModel):
    town_id: str
    map_path: str
    map_name: Optional[str] = None
    notes: Optional[str] = None


class ScaffoldTemplateRequest(BaseModel):
    template_id: str = Field(description="Template folder name under assets/templates")
    town_id: str = Field(description="Town slug used for output folder naming")
    out_map_path: Optional[str] = Field(
        default=None,
        description="Optional explicit output map JSON path",
    )
    out_nav_path: Optional[str] = Field(
        default=None,
        description="Optional explicit output nav JSON path",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Optional workspace path to customization JSON",
    )
    customization: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional inline customization patch object",
    )
    validate_after: bool = True


def _resolve_workspace_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Path must be within workspace")
    return candidate


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


def _resolve_template_dir(template_id: str) -> Path:
    template_dir = (TEMPLATES_ROOT / template_id).resolve()
    if template_dir != TEMPLATES_ROOT and TEMPLATES_ROOT not in template_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid template_id path")
    if not template_dir.exists() or not template_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")
    return template_dir


def _load_customization_from_request(req: ScaffoldTemplateRequest) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if req.config_path and req.customization:
        raise HTTPException(
            status_code=400,
            detail="Provide either config_path or customization, not both",
        )

    if req.config_path:
        config_path = _resolve_workspace_path(req.config_path)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid customization JSON: {exc}") from exc
        return parsed, str(config_path)

    return req.customization, None


def _ensure_valid_town_id(town_id: str) -> None:
    if not VALID_TOWN_ID_RE.fullmatch(town_id):
        raise HTTPException(
            status_code=400,
            detail="town_id must match [a-zA-Z0-9][a-zA-Z0-9_-]{1,63}",
        )


@router.post("/validate")
def validate_map_inline(req: ValidateMapRequest) -> dict[str, Any]:
    logger.info("[MAPS] Inline map validation request")
    errors, warnings, summary = validate_map(req.map_data)
    logger.info("[MAPS] Inline validation complete: ok=%s, errors=%d, warnings=%d", 
                len(errors) == 0, len(errors), len(warnings))
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


@router.post("/validate-file")
def validate_map_file(req: ValidateMapFileRequest) -> dict[str, Any]:
    logger.info("[MAPS] File map validation request: map_path='%s'", req.map_path)
    map_path = _resolve_workspace_path(req.map_path)
    if not map_path.exists():
        logger.warning("[MAPS] Validation failed - map file not found: '%s'", map_path)
        raise HTTPException(status_code=404, detail=f"Map file not found: {map_path}")

    logger.debug("[MAPS] Loading map file: '%s'", map_path)
    map_data = load_map(map_path)
    errors, warnings, summary = validate_map(map_data)
    logger.info("[MAPS] File validation complete: path='%s', ok=%s, errors=%d, warnings=%d", 
                map_path, len(errors) == 0, len(errors), len(warnings))
    return {
        "ok": len(errors) == 0,
        "map_path": str(map_path),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


@router.post("/bake")
def bake_map_inline(req: BakeMapRequest) -> dict[str, Any]:
    errors, warnings, summary, nav_payload = build_nav_payload(
        req.map_data,
        source_map_file=req.source_map_file,
    )
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "nav": nav_payload,
    }


@router.post("/bake-file")
def bake_map_file(req: BakeMapFileRequest) -> dict[str, Any]:
    map_path = _resolve_workspace_path(req.map_path)
    if not map_path.exists():
        raise HTTPException(status_code=404, detail=f"Map file not found: {map_path}")

    out_path = _resolve_workspace_path(req.out_path) if req.out_path else None

    errors, warnings, summary, written = bake_nav_to_file(
        map_path,
        out_path=out_path,
        allow_invalid=req.allow_invalid,
    )

    if errors and not req.allow_invalid:
        return {
            "ok": False,
            "map_path": str(map_path),
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }

    return {
        "ok": len(errors) == 0,
        "map_path": str(map_path),
        "out_path": str(written),
        "source_sha256": map_sha256(map_path),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


@router.post("/customize-file")
def customize_map(req: CustomizeMapFileRequest) -> dict[str, Any]:
    map_path = _resolve_workspace_path(req.map_path)
    config_path = _resolve_workspace_path(req.config_path)
    if not map_path.exists():
        raise HTTPException(status_code=404, detail=f"Map file not found: {map_path}")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

    out_path = _resolve_workspace_path(req.out_path) if req.out_path else None

    notes, patch_errors, validation_errors, written = customize_map_file(
        map_path,
        config_path,
        out_path=out_path,
        validate_after=req.validate_after,
    )

    return {
        "ok": (len(patch_errors) == 0 and len(validation_errors) == 0),
        "map_path": str(map_path),
        "config_path": str(config_path),
        "out_path": str(written),
        "notes": notes,
        "patch_errors": patch_errors,
        "validation_errors": validation_errors,
    }


@router.get("/templates")
def list_map_templates() -> dict[str, Any]:
    logger.info("[MAPS] List templates request")
    templates = discover_templates(TEMPLATES_ROOT, WORKSPACE_ROOT)
    logger.info("[MAPS] Templates discovered: count=%d", len(templates))
    return {"count": len(templates), "templates": templates}


@router.post("/templates/scaffold")
def scaffold_map_template(req: ScaffoldTemplateRequest) -> dict[str, Any]:
    logger.info("[MAPS] Scaffold template request: template_id='%s', town_id='%s'", 
                req.template_id, req.town_id)
    _ensure_valid_town_id(req.town_id)

    template_dir = _resolve_template_dir(req.template_id)
    template_map_path = template_dir / "map.json"
    if not template_map_path.exists():
        logger.warning("[MAPS] Scaffold failed - template map not found: '%s'", template_map_path)
        raise HTTPException(status_code=404, detail=f"Template map not found: {template_map_path}")

    logger.debug("[MAPS] Loading template: '%s'", template_map_path)
    map_data = load_map(template_map_path)
    notes: list[str] = []
    patch_errors: list[str] = []

    customization_config, resolved_config_path = _load_customization_from_request(req)
    if customization_config is not None:
        logger.info("[MAPS] Applying customization to template")
        notes, patch_errors = apply_customization(map_data, customization_config)
        if patch_errors:
            logger.warning("[MAPS] Customization produced %d patch errors", len(patch_errors))

    errors, warnings, summary = validate_map(map_data)
    if patch_errors:
        logger.warning("[MAPS] Scaffold failed due to patch errors: %d errors", len(patch_errors))
        return {
            "ok": False,
            "template_id": req.template_id,
            "town_id": req.town_id,
            "notes": notes,
            "patch_errors": patch_errors,
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }
    if errors and req.validate_after:
        logger.warning("[MAPS] Scaffold failed validation: %d errors", len(errors))
        return {
            "ok": False,
            "template_id": req.template_id,
            "town_id": req.town_id,
            "notes": notes,
            "patch_errors": [],
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }

    out_map_path = (
        _resolve_workspace_path(req.out_map_path)
        if req.out_map_path
        else (TOWN_MAP_ROOT / req.town_id / "draft.map.json").resolve()
    )
    out_nav_path = (
        _resolve_workspace_path(req.out_nav_path)
        if req.out_nav_path
        else out_map_path.with_suffix(".nav.json")
    )

    source_sha = map_sha256_json(map_data)
    nav_errors, nav_warnings, nav_summary, nav_payload = build_nav_payload(
        map_data,
        source_map_file=_rel_workspace(out_map_path),
        source_sha256=source_sha,
    )
    if nav_errors and req.validate_after:
        logger.warning("[MAPS] Scaffold failed nav validation: %d errors", len(nav_errors))
        return {
            "ok": False,
            "template_id": req.template_id,
            "town_id": req.town_id,
            "map_path": str(out_map_path),
            "notes": notes,
            "patch_errors": [],
            "errors": nav_errors,
            "warnings": nav_warnings,
            "summary": nav_summary,
        }

    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    out_nav_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_path.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
    out_nav_path.write_text(json.dumps(nav_payload, indent=2), encoding="utf-8")
    
    logger.info("[MAPS] Template scaffolded successfully: town_id='%s', map_path='%s', nav_path='%s'", 
                req.town_id, out_map_path, out_nav_path)
    
    return {
        "ok": True,
        "template_id": req.template_id,
        "town_id": req.town_id,
        "map_path": str(out_map_path),
        "nav_path": str(out_nav_path),
        "config_path": resolved_config_path,
        "notes": notes,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "source_sha256": source_sha,
    }


@router.post("/publish-version")
def publish_map_version(req: PublishMapVersionRequest) -> dict[str, Any]:
    logger.info("[MAPS] Publish version request: town_id='%s', map_path='%s'", 
                req.town_id, req.map_path)
    
    map_path = _resolve_workspace_path(req.map_path)
    if not map_path.exists():
        logger.warning("[MAPS] Publish failed - map file not found: '%s'", map_path)
        raise HTTPException(status_code=404, detail=f"Map file not found: {map_path}")

    logger.debug("[MAPS] Loading map for publish: '%s'", map_path)
    map_data = load_map(map_path)
    errors, warnings, summary = validate_map(map_data)
    if errors:
        logger.warning("[MAPS] Publish failed - validation errors: %d errors", len(errors))
        return {
            "ok": False,
            "map_path": str(map_path),
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        }

    version_number = next_version(req.town_id)
    logger.info("[MAPS] Creating version %d for town '%s'", version_number, req.town_id)
    
    map_store_dir = WORKSPACE_ROOT / "assets" / "maps" / "v_valley" / req.town_id
    map_store_dir.mkdir(parents=True, exist_ok=True)

    map_snapshot = map_store_dir / f"v{version_number}.map.json"
    nav_snapshot = map_store_dir / f"v{version_number}.nav.json"

    map_snapshot.write_text(json.dumps(map_data, indent=2), encoding="utf-8")
    logger.debug("[MAPS] Map snapshot written: '%s'", map_snapshot)

    nav_errors, nav_warnings, nav_summary, nav_payload = build_nav_payload(
        map_data,
        source_map_file=_rel_workspace(map_snapshot),
        source_sha256=map_sha256(map_snapshot),
    )
    if nav_errors:
        logger.warning("[MAPS] Publish failed - nav validation errors: %d errors", len(nav_errors))
        return {
            "ok": False,
            "map_path": str(map_path),
            "errors": nav_errors,
            "warnings": nav_warnings,
            "summary": nav_summary,
        }

    nav_snapshot.write_text(json.dumps(nav_payload, indent=2), encoding="utf-8")
    logger.debug("[MAPS] Nav snapshot written: '%s'", nav_snapshot)

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
        map_name=req.map_name or map_snapshot.stem,
        map_json_path=_rel_workspace(map_snapshot),
        nav_data_path=_rel_workspace(nav_snapshot),
        source_sha256=map_sha256(map_snapshot),
        affordances=affordance_rows,
        notes=req.notes,
    )
    normalized = _normalize_version(record)
    
    logger.info("[MAPS] Version published successfully: town_id='%s', version=%d, affordances=%d", 
                req.town_id, version_number, len(affordance_rows))
    
    return {
        "ok": True,
        "town_id": req.town_id,
        "version": normalized,
        "summary": summary,
        "affordance_count": len(affordance_rows),
    }


@router.get("/versions/{town_id}")
def list_map_versions(town_id: str) -> dict[str, Any]:
    logger.info("[MAPS] List versions request: town_id='%s'", town_id)
    rows = [_normalize_version(row) for row in list_versions(town_id)]
    logger.info("[MAPS] Versions listed: town_id='%s', count=%d", town_id, len(rows))
    return {"town_id": town_id, "versions": rows}


@router.get("/versions/{town_id}/active")
def active_map_version(town_id: str) -> dict[str, Any]:
    logger.info("[MAPS] Get active version request: town_id='%s'", town_id)
    row = get_active_version(town_id)
    if not row:
        logger.warning("[MAPS] No active version found: town_id='%s'", town_id)
        raise HTTPException(status_code=404, detail=f"No active map version for town: {town_id}")
    normalized = _normalize_version(row)
    normalized["affordances"] = get_affordances(row["id"])
    logger.info("[MAPS] Active version retrieved: town_id='%s', version=%s", town_id, row.get("version"))
    return {"town_id": town_id, "active": normalized}


@router.get("/versions/{town_id}/{version}")
def get_map_version(town_id: str, version: int) -> dict[str, Any]:
    logger.info("[MAPS] Get version request: town_id='%s', version=%d", town_id, version)
    row = get_version(town_id, version)
    if not row:
        logger.warning("[MAPS] Version not found: town_id='%s', version=%d", town_id, version)
        raise HTTPException(status_code=404, detail=f"Map version not found: {town_id} v{version}")
    normalized = _normalize_version(row)
    normalized["affordances"] = get_affordances(row["id"])
    logger.info("[MAPS] Version retrieved: town_id='%s', version=%d", town_id, version)
    return {"town_id": town_id, "version": normalized}


@router.post("/versions/{town_id}/{version}/activate")
def activate_map_version(town_id: str, version: int) -> dict[str, Any]:
    logger.info("[MAPS] Activate version request: town_id='%s', version=%d", town_id, version)
    row = activate_version(town_id, version)
    if not row:
        logger.warning("[MAPS] Activate failed - version not found: town_id='%s', version=%d", town_id, version)
        raise HTTPException(status_code=404, detail=f"Map version not found: {town_id} v{version}")
    normalized = _normalize_version(row)
    logger.info("[MAPS] Version activated successfully: town_id='%s', version=%d", town_id, version)
    return {"ok": True, "town_id": town_id, "active": normalized}
