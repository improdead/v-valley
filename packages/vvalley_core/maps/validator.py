"""Map validation logic for V-Valley."""

from __future__ import annotations

from collections import deque
from logging import getLogger
from typing import Any

from .map_utils import (
    REQUIRED_AFFORDANCES,
    REQUIRED_TILE_LAYERS,
    collision_grid,
    get_layer,
    layer_names,
    neighbors4,
    parse_location_objects,
    parse_spawns,
    validate_tile_layer_shape,
    walkable_grid,
)

logger = getLogger("vvalley_core.maps.validator")


def bfs_reachable(start: tuple[int, int], walkable: list[list[int]]) -> set[tuple[int, int]]:
    height = len(walkable)
    width = len(walkable[0]) if height else 0
    seen: set[tuple[int, int]] = set()
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        if (x, y) in seen:
            continue
        seen.add((x, y))
        for nx, ny in neighbors4(x, y, width, height):
            if walkable[ny][nx] and (nx, ny) not in seen:
                queue.append((nx, ny))
    return seen


def validate_map(map_data: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    logger.debug("[VALIDATOR] Starting map validation")
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("width", "height", "tilewidth", "tileheight", "layers"):
        if key not in map_data:
            errors.append(f"Missing top-level key: {key}")

    if errors:
        logger.warning("[VALIDATOR] Validation failed - missing required keys: %s", errors)
        return errors, warnings, {}

    width = int(map_data["width"])
    height = int(map_data["height"])

    missing_layers = [name for name in REQUIRED_TILE_LAYERS if name not in layer_names(map_data)]
    if missing_layers:
        errors.append("Missing required layers: " + ", ".join(missing_layers))

    for name in REQUIRED_TILE_LAYERS:
        layer = get_layer(map_data, name)
        if not layer:
            continue
        errors.extend(validate_tile_layer_shape(layer, width, height))

    if errors:
        return errors, warnings, {}

    collision = collision_grid(map_data)
    walkable = walkable_grid(collision)

    walkable_count = sum(sum(row) for row in walkable)
    blocked_count = width * height - walkable_count
    if walkable_count == 0:
        errors.append("Map has zero walkable tiles")

    locations = parse_location_objects(map_data)
    if not locations:
        errors.append("Missing 'locations' object layer or no locations defined")

    affordance_to_locations: dict[str, list[tuple[int, int]]] = {k: [] for k in REQUIRED_AFFORDANCES}

    for loc in locations:
        if not loc.affordances:
            warnings.append(f"Location '{loc.name}' has no affordances")
            continue

        if not walkable[loc.y][loc.x]:
            errors.append(f"Location '{loc.name}' is placed on blocked tile ({loc.x}, {loc.y})")

        for affordance in loc.affordances:
            if affordance in affordance_to_locations:
                affordance_to_locations[affordance].append((loc.x, loc.y))

    for affordance, points in affordance_to_locations.items():
        if not points:
            errors.append(f"Missing required affordance location: {affordance}")

    spawns = parse_spawns(map_data)
    if not spawns:
        errors.append("Missing 'spawns' object layer or no spawn points defined")
    else:
        for spawn in spawns:
            if not walkable[spawn.y][spawn.x]:
                errors.append(
                    f"Spawn '{spawn.name}' is placed on blocked tile ({spawn.x}, {spawn.y})"
                )

    if not errors:
        representative_points: list[tuple[str, tuple[int, int]]] = []
        for affordance in REQUIRED_AFFORDANCES:
            points = affordance_to_locations.get(affordance, [])
            if points:
                representative_points.append((affordance, points[0]))

        if representative_points:
            anchor_name, anchor = representative_points[0]
            reachable = bfs_reachable(anchor, walkable)

            for affordance, point in representative_points[1:]:
                if point not in reachable:
                    errors.append(
                        "Connectivity check failed: "
                        f"'{affordance}' at {point} is unreachable from '{anchor_name}' at {anchor}"
                    )

    summary = {
        "dimensions": {"width": width, "height": height},
        "tiles": {
            "walkable": walkable_count,
            "blocked": blocked_count,
            "walkable_ratio": round(walkable_count / (width * height), 4) if width and height else 0,
        },
        "locations": len(locations),
        "spawns": len(spawns),
        "required_affordances": {
            key: len(value) for key, value in affordance_to_locations.items()
        },
    }

    if errors:
        logger.warning("[VALIDATOR] Validation completed with errors: %d errors, %d warnings", 
                      len(errors), len(warnings))
    else:
        logger.info("[VALIDATOR] Validation completed successfully: %d warnings, walkable=%d, blocked=%d", 
                   len(warnings), walkable_count, blocked_count)

    return errors, warnings, summary
