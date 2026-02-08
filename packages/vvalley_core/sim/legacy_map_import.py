"""Import legacy generative_agents map assets into V-Valley map format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json

from packages.vvalley_core.maps.validator import validate_map


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEGACY_MATRIX_ROOT = (
    WORKSPACE_ROOT
    / "generative_agents"
    / "environment"
    / "frontend_server"
    / "static_dirs"
    / "assets"
    / "the_ville"
    / "matrix"
)

AFFORDANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "home": ("home", "apartment", "house", "dorm", "room"),
    "work": ("college", "school", "office", "store", "market", "shop", "supply"),
    "food": ("cafe", "pub", "kitchen", "restaurant", "bar"),
    "social": ("park", "common room", "plaza", "square", "lounge"),
    "leisure": ("park", "pub", "music", "guitar", "recreation", "common room"),
}


def _split_csv_line(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_special_block_map(path: Path, value_index: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) <= value_index:
            continue
        out[parts[0]] = parts[value_index]
    return out


def _pick_affordance_locations(
    *,
    width: int,
    height: int,
    collision_flat: list[int],
    sector_flat: list[str],
    arena_flat: list[str],
) -> dict[str, tuple[int, int]]:
    def neighbors(x: int, y: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        if x > 0:
            out.append((x - 1, y))
        if x + 1 < width:
            out.append((x + 1, y))
        if y > 0:
            out.append((x, y - 1))
        if y + 1 < height:
            out.append((x, y + 1))
        return out

    def reachable_from(start: tuple[int, int]) -> set[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        stack = [start]
        while stack:
            x, y = stack.pop()
            if (x, y) in seen:
                continue
            idx = y * width + x
            if collision_flat[idx]:
                continue
            seen.add((x, y))
            for nx, ny in neighbors(x, y):
                if (nx, ny) not in seen:
                    stack.append((nx, ny))
        return seen

    walkable_tiles: list[tuple[int, int]] = []
    candidates: dict[str, list[tuple[int, int]]] = {k: [] for k in AFFORDANCE_KEYWORDS}

    for idx, blocked in enumerate(collision_flat):
        x = idx % width
        y = idx // width
        if blocked:
            continue
        walkable_tiles.append((x, y))
        text = f"{sector_flat[idx]} {arena_flat[idx]}".lower()
        for affordance, keywords in AFFORDANCE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                candidates[affordance].append((x, y))

    chosen: dict[str, tuple[int, int]] = {}

    if not walkable_tiles:
        for affordance in AFFORDANCE_KEYWORDS:
            chosen[affordance] = (0, 0)
        return chosen

    home_candidates = candidates.get("home") or []
    home_tile = home_candidates[0] if home_candidates else walkable_tiles[0]
    reachable = reachable_from(home_tile)
    if home_tile not in reachable and reachable:
        home_tile = next(iter(reachable))
    chosen["home"] = home_tile

    if not reachable:
        reachable = set(walkable_tiles)

    for affordance in AFFORDANCE_KEYWORDS:
        if affordance == "home":
            continue
        picks = candidates.get(affordance) or []
        reachable_pick = next((tile for tile in picks if tile in reachable), None)
        if reachable_pick:
            chosen[affordance] = reachable_pick
            continue
        fallback = next(iter(reachable)) if reachable else home_tile
        chosen[affordance] = fallback
    return chosen


def _build_object_layer(
    *,
    name: str,
    tilewidth: int,
    tileheight: int,
    width: int,
    height: int,
    objects: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "type": "objectgroup",
        "opacity": 1,
        "visible": True,
        "x": 0,
        "y": 0,
        "draworder": "topdown",
        "objects": objects,
        "width": width,
        "height": height,
        "tilewidth": tilewidth,
        "tileheight": tileheight,
    }


def _build_tile_layer(name: str, width: int, height: int, data: list[int]) -> dict[str, Any]:
    return {
        "name": name,
        "type": "tilelayer",
        "width": width,
        "height": height,
        "opacity": 1,
        "visible": True,
        "x": 0,
        "y": 0,
        "data": data,
    }


def build_the_ville_map(matrix_root: Optional[Path] = None) -> dict[str, Any]:
    root = (matrix_root or DEFAULT_LEGACY_MATRIX_ROOT).resolve()
    meta = json.loads((root / "maze_meta_info.json").read_text(encoding="utf-8"))

    width = int(meta["maze_width"])
    height = int(meta["maze_height"])
    tilewidth = int(meta.get("sq_tile_size") or 32)
    tileheight = int(meta.get("sq_tile_size") or 32)
    total = width * height

    maze_root = root / "maze"
    collision_codes = _split_csv_line(maze_root / "collision_maze.csv")
    sector_codes = _split_csv_line(maze_root / "sector_maze.csv")
    arena_codes = _split_csv_line(maze_root / "arena_maze.csv")
    spawn_codes = _split_csv_line(maze_root / "spawning_location_maze.csv")

    if not (
        len(collision_codes) == len(sector_codes) == len(arena_codes) == len(spawn_codes) == total
    ):
        raise ValueError("Legacy maze CSV dimensions do not match maze_meta_info")

    blocks_root = root / "special_blocks"
    sector_map = _load_special_block_map(blocks_root / "sector_blocks.csv", 2)
    arena_map = _load_special_block_map(blocks_root / "arena_blocks.csv", 3)
    spawn_map = _load_special_block_map(blocks_root / "spawning_location_blocks.csv", 4)

    collision_flat = [1 if code != "0" else 0 for code in collision_codes]
    sector_flat = [sector_map.get(code, "") for code in sector_codes]
    arena_flat = [arena_map.get(code, "") for code in arena_codes]
    affordance_tiles = _pick_affordance_locations(
        width=width,
        height=height,
        collision_flat=collision_flat,
        sector_flat=sector_flat,
        arena_flat=arena_flat,
    )

    locations: list[dict[str, Any]] = []
    obj_id = 1
    for affordance, (x, y) in affordance_tiles.items():
        locations.append(
            {
                "id": obj_id,
                "name": f"{affordance}_hub",
                "type": "location",
                "x": x * tilewidth,
                "y": y * tileheight,
                "width": tilewidth,
                "height": tileheight,
                "rotation": 0,
                "visible": True,
                "properties": [
                    {
                        "name": "affordances",
                        "type": "string",
                        "value": affordance,
                    }
                ],
            }
        )
        obj_id += 1

    spawn_seen: set[str] = set()
    spawns: list[dict[str, Any]] = []
    for idx, code in enumerate(spawn_codes):
        spawn_name = spawn_map.get(code)
        if not spawn_name or spawn_name in spawn_seen:
            continue
        if collision_flat[idx]:
            continue
        spawn_seen.add(spawn_name)
        x = idx % width
        y = idx // width
        spawns.append(
            {
                "id": obj_id,
                "name": spawn_name,
                "type": "spawn",
                "x": x * tilewidth,
                "y": y * tileheight,
                "width": tilewidth,
                "height": tileheight,
                "rotation": 0,
                "visible": True,
            }
        )
        obj_id += 1

    if not spawns:
        spawns.append(
            {
                "id": obj_id,
                "name": "spawn_a",
                "type": "spawn",
                "x": 0,
                "y": 0,
                "width": tilewidth,
                "height": tileheight,
                "rotation": 0,
                "visible": True,
            }
        )

    zeros = [0] * total
    ones = [1] * total

    return {
        "type": "map",
        "version": "1.10",
        "tiledversion": "1.10.2",
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "width": width,
        "height": height,
        "tilewidth": tilewidth,
        "tileheight": tileheight,
        "infinite": False,
        "layers": [
            _build_tile_layer("ground", width, height, ones),
            _build_tile_layer("collision", width, height, collision_flat),
            _build_tile_layer("interiors", width, height, zeros),
            _build_tile_layer("doors", width, height, zeros),
            _build_tile_layer("decor", width, height, zeros),
            _build_object_layer(
                name="locations",
                tilewidth=tilewidth,
                tileheight=tileheight,
                width=width,
                height=height,
                objects=locations,
            ),
            _build_object_layer(
                name="spawns",
                tilewidth=tilewidth,
                tileheight=tileheight,
                width=width,
                height=height,
                objects=spawns,
            ),
        ],
        "properties": [
            {"name": "town_theme", "type": "string", "value": "legacy_the_ville"},
            {"name": "source", "type": "string", "value": "generative_agents"},
        ],
    }


def import_the_ville_map(
    *,
    out_map_path: Path,
    matrix_root: Optional[Path] = None,
) -> dict[str, Any]:
    map_data = build_the_ville_map(matrix_root=matrix_root)
    errors, warnings, summary = validate_map(map_data)
    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_path.write_text(json.dumps(map_data, indent=2), encoding="utf-8")

    return {
        "ok": len(errors) == 0,
        "map_path": str(out_map_path.resolve()),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "map": {
            "width": int(map_data["width"]),
            "height": int(map_data["height"]),
            "tilewidth": int(map_data["tilewidth"]),
            "tileheight": int(map_data["tileheight"]),
        },
    }
