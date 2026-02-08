"""Navigation baking helpers for V-Valley maps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .map_utils import (
    collision_grid,
    load_map,
    map_sha256,
    map_sha256_json,
    parse_location_objects,
    parse_spawns,
    walkable_grid,
)
from .validator import validate_map

logger = logging.getLogger("vvalley_core.maps.nav")


def _flatten(grid: list[list[int]]) -> list[int]:
    return [cell for row in grid for cell in row]


def build_nav_payload(
    map_data: dict[str, Any],
    *,
    source_map_file: str = "<inline>",
    source_sha256: str | None = None,
) -> tuple[list[str], list[str], dict[str, Any], dict[str, Any]]:
    logger.debug("[NAV] Building nav payload for: %s", source_map_file)
    errors, warnings, summary = validate_map(map_data)

    collision = collision_grid(map_data)
    walkable = walkable_grid(collision)
    locations = parse_location_objects(map_data)
    spawns = parse_spawns(map_data)

    nav_payload = {
        "format": "vvalley-nav-v1",
        "source": {
            "map_file": source_map_file,
            "sha256": source_sha256 or map_sha256_json(map_data),
        },
        "dimensions": {
            "width": int(map_data["width"]),
            "height": int(map_data["height"]),
            "tilewidth": int(map_data["tilewidth"]),
            "tileheight": int(map_data["tileheight"]),
        },
        "walkable": _flatten(walkable),
        "collision": _flatten(collision),
        "neighbors": {
            "mode": "4dir",
            "deltas": [[1, 0], [-1, 0], [0, 1], [0, -1]],
        },
        "locations": [
            {
                "name": loc.name,
                "x": loc.x,
                "y": loc.y,
                "affordances": list(loc.affordances),
            }
            for loc in locations
        ],
        "spawns": [{"name": s.name, "x": s.x, "y": s.y} for s in spawns],
        "validation": {
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
        },
    }

    return errors, warnings, summary, nav_payload


def bake_nav_to_file(
    map_path: Path,
    *,
    out_path: Path | None = None,
    allow_invalid: bool = False,
) -> tuple[list[str], list[str], dict[str, Any], Path]:
    logger.info("[NAV] Baking navigation to file: map_path='%s', out_path='%s', allow_invalid=%s",
                map_path, out_path, allow_invalid)
    map_data = load_map(map_path)

    errors, warnings, summary, nav_payload = build_nav_payload(
        map_data,
        source_map_file=str(map_path),
        source_sha256=map_sha256(map_path),
    )

    if errors and not allow_invalid:
        logger.warning("[NAV] Baking failed - validation errors and allow_invalid=False: %d errors", len(errors))
        return errors, warnings, summary, out_path or map_path.with_suffix(".nav.json")

    out = out_path or map_path.with_suffix(".nav.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nav_payload, indent=2), encoding="utf-8")
    logger.info("[NAV] Navigation baked successfully: out='%s', errors=%d, warnings=%d", 
                out, len(errors), len(warnings))
    return errors, warnings, summary, out
