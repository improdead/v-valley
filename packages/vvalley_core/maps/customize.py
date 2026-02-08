"""No-code map customization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple, List

from .map_utils import get_layer, load_map
from .validator import validate_map


def set_property(obj: dict, name: str, value: Any) -> None:
    props = obj.setdefault("properties", [])
    for prop in props:
        if prop.get("name") == name:
            prop["value"] = value
            return

    prop_type = "string"
    if isinstance(value, bool):
        prop_type = "bool"
    elif isinstance(value, int):
        prop_type = "int"
    elif isinstance(value, float):
        prop_type = "float"

    props.append({"name": name, "type": prop_type, "value": value})


def find_object_by_name(layer: dict, name: str):
    for obj in layer.get("objects", []):
        if str(obj.get("name", "")) == name:
            return obj
    return None


def apply_customization(map_data: dict, config: dict) -> Tuple[List[str], List[str]]:
    notes: List[str] = []
    errors: List[str] = []

    tile_w = int(map_data["tilewidth"])
    tile_h = int(map_data["tileheight"])

    map_props = config.get("map_properties", {})
    if not isinstance(map_props, dict):
        errors.append("map_properties must be an object")
    else:
        for key, value in map_props.items():
            set_property(map_data, key, value)
            notes.append(f"Set map property: {key}")

    locations_layer = get_layer(map_data, "locations")
    if locations_layer and "locations" in config:
        for patch in config.get("locations", []):
            name = patch.get("name")
            if not name:
                errors.append("location patch missing name")
                continue
            obj = find_object_by_name(locations_layer, name)
            if not obj:
                errors.append(f"location not found: {name}")
                continue

            if "affordances" in patch:
                affordances = patch["affordances"]
                if not isinstance(affordances, list) or not all(isinstance(v, str) for v in affordances):
                    errors.append(f"location '{name}' affordances must be a list of strings")
                else:
                    set_property(obj, "affordances", ",".join(affordances))
                    notes.append(f"Updated affordances for location: {name}")

            if "hours" in patch:
                set_property(obj, "hours", str(patch["hours"]))
                notes.append(f"Updated hours for location: {name}")

            if "zone_rules" in patch:
                rules = patch["zone_rules"]
                if isinstance(rules, list):
                    set_property(obj, "zone_rules", ",".join(str(r) for r in rules))
                    notes.append(f"Updated zone_rules for location: {name}")
                else:
                    errors.append(f"location '{name}' zone_rules must be a list")
    elif "locations" in config:
        errors.append("map has no locations layer")

    spawns_layer = get_layer(map_data, "spawns")
    if spawns_layer and "spawns" in config:
        for patch in config.get("spawns", []):
            name = patch.get("name")
            if not name:
                errors.append("spawn patch missing name")
                continue
            obj = find_object_by_name(spawns_layer, name)
            if not obj:
                errors.append(f"spawn not found: {name}")
                continue

            if "x" in patch and "y" in patch:
                x = int(patch["x"])
                y = int(patch["y"])
                obj["x"] = x * tile_w
                obj["y"] = y * tile_h
                notes.append(f"Moved spawn '{name}' to ({x}, {y})")
            else:
                errors.append(f"spawn '{name}' requires both x and y")
    elif "spawns" in config:
        errors.append("map has no spawns layer")

    return notes, errors


def customize_map_file(
    map_path: Path,
    config_path: Path,
    *,
    out_path: Optional[Path] = None,
    validate_after: bool = True,
) -> Tuple[List[str], List[str], List[str], Path]:
    map_data = load_map(map_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    notes, patch_errors = apply_customization(map_data, config)
    validation_errors: List[str] = []

    if not patch_errors and validate_after:
        validation_errors, _, _ = validate_map(map_data)

    out = out_path or map_path.with_name(map_path.stem + ".custom.json")
    if not patch_errors and not validation_errors:
        out.write_text(json.dumps(map_data, indent=2), encoding="utf-8")

    return notes, patch_errors, validation_errors, out
