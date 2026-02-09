#!/usr/bin/env python3
"""Utility helpers for V-Valley map tooling.

These helpers operate on Tiled-style JSON map exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

REQUIRED_TILE_LAYERS = ("ground", "collision", "interiors", "doors", "decor")
REQUIRED_AFFORDANCES = ("home", "work", "food", "social", "leisure")


@dataclass(frozen=True)
class LocationObject:
    """Parsed location object with normalized affordances and tile position."""

    name: str
    x: int
    y: int
    affordances: tuple[str, ...]


@dataclass(frozen=True)
class SpawnPoint:
    """Spawn point in tile coordinates."""

    name: str
    x: int
    y: int


def load_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def map_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def map_sha256_json(map_data: dict[str, Any]) -> str:
    payload = json.dumps(map_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_layer(map_data: dict[str, Any], name: str) -> dict[str, Any] | None:
    for layer in map_data.get("layers", []):
        if layer.get("name") == name:
            return layer
    return None


def get_property(obj: dict[str, Any], key: str, default: Any = None) -> Any:
    for prop in obj.get("properties", []):
        if prop.get("name") == key:
            return prop.get("value", default)
    return default


def parse_affordances(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, list):
        values = [str(v).strip().lower() for v in raw if str(v).strip()]
        return tuple(dict.fromkeys(values))
    if isinstance(raw, str):
        values = [v.strip().lower() for v in raw.split(",") if v.strip()]
        return tuple(dict.fromkeys(values))
    value = str(raw).strip().lower()
    return (value,) if value else ()


def object_center_tile(obj: dict[str, Any], tile_w: int, tile_h: int, map_w: int, map_h: int) -> tuple[int, int]:
    # Tiled object x/y is top-left for rectangles, so use center for placement.
    x_px = float(obj.get("x", 0.0)) + float(obj.get("width", 0.0)) / 2.0
    y_px = float(obj.get("y", 0.0)) + float(obj.get("height", 0.0)) / 2.0

    x = int(x_px // tile_w)
    y = int(y_px // tile_h)

    x = max(0, min(map_w - 1, x))
    y = max(0, min(map_h - 1, y))
    return x, y


def collision_grid(map_data: dict[str, Any]) -> list[list[int]]:
    width = int(map_data["width"])
    height = int(map_data["height"])
    layer = get_layer(map_data, "collision")
    if not layer:
        raise ValueError("Missing collision layer")

    data = layer.get("data", [])
    if len(data) != width * height:
        raise ValueError("Collision layer has invalid tile count")

    grid: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]
    for idx, value in enumerate(data):
        y = idx // width
        x = idx % width
        grid[y][x] = 1 if int(value) != 0 else 0
    return grid


def walkable_grid(collision: list[list[int]]) -> list[list[int]]:
    return [[0 if cell else 1 for cell in row] for row in collision]


def in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def neighbors4(x: int, y: int, width: int, height: int) -> Iterable[tuple[int, int]]:
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, width, height):
            yield nx, ny


def parse_location_objects(map_data: dict[str, Any]) -> list[LocationObject]:
    tile_w = int(map_data["tilewidth"])
    tile_h = int(map_data["tileheight"])
    map_w = int(map_data["width"])
    map_h = int(map_data["height"])

    layer = get_layer(map_data, "locations")
    if not layer:
        return []

    out: list[LocationObject] = []
    for obj in layer.get("objects", []):
        x, y = object_center_tile(obj, tile_w, tile_h, map_w, map_h)
        affordances = parse_affordances(get_property(obj, "affordances"))

        if not affordances:
            role = get_property(obj, "role")
            affordances = parse_affordances(role)

        out.append(
            LocationObject(
                name=str(obj.get("name", "location")).strip() or "location",
                x=x,
                y=y,
                affordances=affordances,
            )
        )
    return out


def parse_spawns(map_data: dict[str, Any]) -> list[SpawnPoint]:
    tile_w = int(map_data["tilewidth"])
    tile_h = int(map_data["tileheight"])
    map_w = int(map_data["width"])
    map_h = int(map_data["height"])

    layer = get_layer(map_data, "spawns")
    if not layer:
        return []

    out: list[SpawnPoint] = []
    for obj in layer.get("objects", []):
        x, y = object_center_tile(obj, tile_w, tile_h, map_w, map_h)
        out.append(SpawnPoint(name=str(obj.get("name", "spawn")), x=x, y=y))
    return out


def layer_names(map_data: dict[str, Any]) -> set[str]:
    return {str(layer.get("name", "")) for layer in map_data.get("layers", [])}


def validate_tile_layer_shape(layer: dict[str, Any], width: int, height: int) -> list[str]:
    errors: list[str] = []
    if layer.get("type") != "tilelayer":
        errors.append(f"Layer '{layer.get('name')}' must be tilelayer")
        return errors

    lw = int(layer.get("width", -1))
    lh = int(layer.get("height", -1))
    if lw != width or lh != height:
        errors.append(
            f"Layer '{layer.get('name')}' dimensions {lw}x{lh} do not match map {width}x{height}"
        )

    data = layer.get("data")
    if not isinstance(data, list) or len(data) != width * height:
        errors.append(
            f"Layer '{layer.get('name')}' data length must be {width * height}"
        )
    return errors


# ---------------------------------------------------------------------------
# Per-tile spatial metadata (GA spatial grounding)
# ---------------------------------------------------------------------------

@dataclass
class TileSpatialMetadata:
    """Per-tile spatial metadata grids built from GA tile layers + embedded lookups.

    Each grid is width×height and maps (x,y) to a name string or None.
    """

    width: int
    height: int
    tile_sectors: list[list[str | None]]   # [y][x] -> sector name or None
    tile_arenas: list[list[str | None]]    # [y][x] -> arena name or None
    tile_objects: list[list[str | None]]   # [y][x] -> game object name or None

    def sector_at(self, x: int, y: int) -> str | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tile_sectors[y][x]
        return None

    def arena_at(self, x: int, y: int) -> str | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tile_arenas[y][x]
        return None

    def object_at(self, x: int, y: int) -> str | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tile_objects[y][x]
        return None

    def full_address_at(self, x: int, y: int) -> tuple[str | None, str | None, str | None]:
        """Return (sector, arena, object) for tile (x,y)."""
        return (self.sector_at(x, y), self.arena_at(x, y), self.object_at(x, y))

    def sectors_list(self) -> list[str]:
        """Unique sector names present on the map."""
        seen: set[str] = set()
        out: list[str] = []
        for row in self.tile_sectors:
            for v in row:
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
        return out

    def arenas_in_sector(self, sector: str) -> list[str]:
        """Unique arena names within a given sector."""
        seen: set[str] = set()
        out: list[str] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.tile_sectors[y][x] == sector:
                    arena = self.tile_arenas[y][x]
                    if arena and arena not in seen:
                        seen.add(arena)
                        out.append(arena)
        return out

    def objects_in_arena(self, sector: str, arena: str) -> list[str]:
        """Unique game object names within a given sector+arena."""
        seen: set[str] = set()
        out: list[str] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.tile_sectors[y][x] == sector and self.tile_arenas[y][x] == arena:
                    obj = self.tile_objects[y][x]
                    if obj and obj not in seen:
                        seen.add(obj)
                        out.append(obj)
        return out

    def find_tile_for_object(
        self, sector: str, arena: str, game_object: str
    ) -> tuple[int, int] | None:
        """Find any tile with the given sector+arena+object. Returns (x,y) or None."""
        for y in range(self.height):
            for x in range(self.width):
                if (
                    self.tile_sectors[y][x] == sector
                    and self.tile_arenas[y][x] == arena
                    and self.tile_objects[y][x] == game_object
                ):
                    return (x, y)
        return None


def build_spatial_metadata(map_data: dict[str, Any]) -> TileSpatialMetadata | None:
    """Build per-tile spatial grids from GA tile layers + embedded lookup tables.

    Returns None if the map doesn't have GA spatial layers or lookups.
    """
    width = int(map_data.get("width", 0))
    height = int(map_data.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    # Extract embedded spatial_lookups
    spatial_json = None
    for prop in map_data.get("properties", []):
        if prop.get("name") == "spatial_lookups":
            spatial_json = prop.get("value")
            break
    if not spatial_json:
        return None

    try:
        lookups = json.loads(spatial_json)
    except (json.JSONDecodeError, TypeError):
        return None

    sector_ids: dict[int, str] = {int(k): v for k, v in lookups.get("sector_ids", {}).items()}
    arena_ids: dict[int, tuple[str, str]] = {}
    for k, v in lookups.get("arena_ids", {}).items():
        if isinstance(v, list) and len(v) == 2:
            arena_ids[int(k)] = (str(v[0]), str(v[1]))
    object_ids: dict[int, str] = {int(k): v for k, v in lookups.get("game_object_ids", {}).items()}

    if not sector_ids:
        return None

    # Find GA spatial tile layers
    sector_layer = get_layer(map_data, "Sector Blocks")
    arena_layer = get_layer(map_data, "Arena Blocks")
    object_layer = get_layer(map_data, "Object Interaction Blocks")

    if not sector_layer:
        return None

    sector_data = sector_layer.get("data", [])
    arena_data = arena_layer.get("data", []) if arena_layer else []
    object_data = object_layer.get("data", []) if object_layer else []

    # Build grids
    tile_sectors: list[list[str | None]] = [[None] * width for _ in range(height)]
    tile_arenas: list[list[str | None]] = [[None] * width for _ in range(height)]
    tile_objects: list[list[str | None]] = [[None] * width for _ in range(height)]

    for idx in range(min(len(sector_data), width * height)):
        tid = int(sector_data[idx])
        if tid != 0 and tid in sector_ids:
            x = idx % width
            y = idx // width
            tile_sectors[y][x] = sector_ids[tid]

    for idx in range(min(len(arena_data), width * height)):
        tid = int(arena_data[idx])
        if tid != 0 and tid in arena_ids:
            x = idx % width
            y = idx // width
            sector_name, arena_name = arena_ids[tid]
            tile_arenas[y][x] = arena_name
            # Also fill in sector from arena if missing
            if tile_sectors[y][x] is None:
                tile_sectors[y][x] = sector_name

    # For objects, the arena they belong to is determined by the tile's arena,
    # so objects inherit sector/arena from overlapping arena tiles
    for idx in range(min(len(object_data), width * height)):
        tid = int(object_data[idx])
        if tid != 0 and tid in object_ids:
            x = idx % width
            y = idx // width
            tile_objects[y][x] = object_ids[tid]

    return TileSpatialMetadata(
        width=width,
        height=height,
        tile_sectors=tile_sectors,
        tile_arenas=tile_arenas,
        tile_objects=tile_objects,
    )
