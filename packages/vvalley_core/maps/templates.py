"""Template discovery helpers for operator-facing town customization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .map_utils import load_map
from .validator import validate_map


def _map_properties(map_data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in map_data.get("properties", []):
        name = prop.get("name")
        if not name:
            continue
        out[str(name)] = prop.get("value")
    return out


def discover_templates(templates_root: Path, workspace_root: Path) -> list[dict[str, Any]]:
    """Return metadata for every template folder that contains a map.json file."""

    if not templates_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for child in sorted(templates_root.iterdir()):
        if not child.is_dir():
            continue

        map_path = child / "map.json"
        if not map_path.exists():
            continue

        map_data = load_map(map_path)
        errors, warnings, summary = validate_map(map_data)

        customization_path = child / "customization.example.json"
        readme_path = child / "README.md"

        def rel(path: Path) -> str:
            return str(path.resolve().relative_to(workspace_root))

        records.append(
            {
                "template_id": child.name,
                "title": str(map_data.get("name") or child.name),
                "map_path": rel(map_path),
                "customization_example_path": rel(customization_path)
                if customization_path.exists()
                else None,
                "readme_path": rel(readme_path) if readme_path.exists() else None,
                "properties": _map_properties(map_data),
                "ok": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "summary": summary,
            }
        )

    return records
