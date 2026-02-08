#!/usr/bin/env python3
"""Convert the_ville_jan7.json (GA format) into a V-Valley compatible map template.

Reads the actual GA tile layers (Sector Blocks, Spawning Blocks, Collisions) and
the special_blocks CSV lookup tables to extract real location positions instead of
guessing coordinates.

Adds the required V-Valley layers (ground, collision, interiors, doors, decor,
locations, spawns) while preserving all original GA visual layers.
"""

import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "generative_agents" / "environment" / "frontend_server" / "static_dirs" / "assets" / "the_ville" / "visuals" / "the_ville_jan7.json"
DST = ROOT / "assets" / "templates" / "starter_town" / "map.json"

GA_MATRIX = ROOT / "generative_agents" / "environment" / "frontend_server" / "static_dirs" / "assets" / "the_ville" / "matrix"


# ----- GA sector → V-Valley affordances mapping -----
# Based on what each sector actually IS in Smallville
SECTOR_AFFORDANCES = {
    "Lin family's house": "home",
    "Moreno family's house": "home",
    "Moore family's house": "home",
    "Adam Smith's house": "home",
    "Yuriko Yamamoto's house": "home",
    "Tamara Taylor and Carmen Ortiz's house": "home",
    "artist's co-living space": "home",
    "Arthur Burton's apartment": "home",
    "Ryan Park's apartment": "home",
    "Isabella Rodriguez's apartment": "home",
    "Giorgio Rossi's apartment": "home",
    "Carlos Gomez's apartment": "home",
    "Dorm for Oak Hill College": "home",
    "Hobbs Cafe": "food,social",
    "The Rose and Crown Pub": "food,social,leisure",
    "Harvey Oak Supply Store": "work",
    "The Willows Market and Pharmacy": "work",
    "Oak Hill College": "work,leisure",
    "Johnson Park": "leisure,social",
}


def parse_blocks_csv(path):
    """Parse a GA special_blocks CSV file.

    Returns dict mapping tile_id (int) → name (str, last field in the row).
    """
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                tile_id = int(parts[0])
                # For sector_blocks: "32135, the Ville, artist's co-living space"
                # We want the sector name (last field)
                name = parts[-1]
                mapping[tile_id] = name
    return mapping


def parse_spawning_blocks_csv(path):
    """Parse the spawning_location_blocks CSV.

    Returns dict mapping tile_id → (sector_name, arena_name, spawn_label).
    """
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Format: "32285, the Ville, artist's co-living space, Latoya Williams's room, sp-A"
            if len(parts) >= 5:
                tile_id = int(parts[0])
                sector = parts[2]
                arena = parts[3]
                label = parts[4]
                mapping[tile_id] = (sector, arena, label)
    return mapping


def find_layer(layers, name):
    for layer in layers:
        if layer.get("name") == name:
            return layer
    return None


def extract_sector_tiles(layer_data, width, height, sector_blocks):
    """For each sector, find all (x, y) tiles that belong to it.

    Returns dict: sector_name → list[(x, y)].
    """
    sectors = defaultdict(list)
    for idx, tile_id in enumerate(layer_data):
        tile_id = int(tile_id)
        if tile_id == 0:
            continue
        if tile_id in sector_blocks:
            x = idx % width
            y = idx // width
            sectors[sector_blocks[tile_id]].append((x, y))
    return sectors


def extract_spawn_tiles(layer_data, width, height, spawn_blocks):
    """Find all spawn point tile positions.

    Returns list of (spawn_name, x, y, sector) tuples.
    Includes sector in name to avoid collisions (many sectors have "main room").
    """
    spawns = []
    for idx, tile_id in enumerate(layer_data):
        tile_id = int(tile_id)
        if tile_id == 0:
            continue
        if tile_id in spawn_blocks:
            x = idx % width
            y = idx // width
            sector, arena, label = spawn_blocks[tile_id]
            # Include sector in name to avoid "main_room_sp-a" collisions
            short_sector = sector.split("'s ")[0] if "'s " in sector else sector
            spawn_name = f"{short_sector}_{label}".replace(" ", "_").lower()
            spawns.append((spawn_name, x, y, sector))
    return spawns


def bfs_connected_components(collision_data, width, height):
    """Find connected walkable components using BFS.

    Returns (component_map, component_sizes) where component_map[y][x] = component_id
    and component_sizes[id] = count.
    """
    comp = [[-1] * width for _ in range(height)]
    sizes = {}
    current_id = 0

    for sy in range(height):
        for sx in range(width):
            idx = sy * width + sx
            if int(collision_data[idx]) != 0 or comp[sy][sx] != -1:
                continue
            # BFS from this tile
            queue = deque([(sx, sy)])
            count = 0
            while queue:
                x, y = queue.popleft()
                if comp[y][x] != -1:
                    continue
                comp[y][x] = current_id
                count += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        nidx = ny * width + nx
                        if int(collision_data[nidx]) == 0 and comp[ny][nx] == -1:
                            queue.append((nx, ny))
            sizes[current_id] = count
            current_id += 1

    return comp, sizes


def sector_bounding_box(tiles):
    """Return (min_x, min_y, max_x, max_y) for a list of tiles."""
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    return min(xs), min(ys), max(xs), max(ys)


def sector_center(tiles):
    """Return center (cx, cy) of a set of tiles."""
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    return sum(xs) // len(xs), sum(ys) // len(ys)


def find_walkable_in_component(tiles, collision_data, width, comp_map, target_comp, center):
    """Find the tile in `tiles` that is walkable and in the target component,
    closest to center. If none found, search outward from center."""
    cx, cy = center

    # First try: walkable sector tiles in the main component
    candidates = []
    for x, y in tiles:
        idx = y * width + x
        if int(collision_data[idx]) == 0 and comp_map[y][x] == target_comp:
            dist = abs(x - cx) + abs(y - cy)
            candidates.append((dist, x, y))

    if candidates:
        candidates.sort()
        return candidates[0][1], candidates[0][2]

    # Second try: any walkable tile near the sector boundary in the main component
    height = len(comp_map)
    bbox = sector_bounding_box(tiles)
    for radius in range(1, 20):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if int(collision_data[ny * width + nx]) == 0 and comp_map[ny][nx] == target_comp:
                        return nx, ny

    return None, None


def make_empty_tile_layer(name, width, height):
    return {
        "data": [0] * (width * height),
        "height": height,
        "id": 0,
        "name": name,
        "opacity": 1,
        "type": "tilelayer",
        "visible": True,
        "width": width,
        "x": 0,
        "y": 0,
    }


def clone_tile_layer_as(src_layer, new_name):
    layer = dict(src_layer)
    layer["name"] = new_name
    return layer


def make_object_group(name, objects):
    return {
        "draworder": "topdown",
        "id": 0,
        "name": name,
        "objects": objects,
        "opacity": 1,
        "type": "objectgroup",
        "visible": False,
        "x": 0,
        "y": 0,
    }


def make_location_object(obj_id, name, tile_x, tile_y, tile_w, tile_h, affordances_str, tw=32, th=32):
    """Create a Tiled-style rectangle object in pixel coordinates."""
    return {
        "id": obj_id,
        "name": name,
        "x": tile_x * tw,
        "y": tile_y * th,
        "width": tile_w * tw,
        "height": tile_h * th,
        "rotation": 0,
        "type": "",
        "visible": True,
        "properties": [
            {"name": "affordances", "type": "string", "value": affordances_str}
        ],
    }


def make_spawn_object(obj_id, name, tile_x, tile_y, tw=32, th=32):
    return {
        "id": obj_id,
        "name": name,
        "x": tile_x * tw + tw // 2,
        "y": tile_y * th + th // 2,
        "width": 0,
        "height": 0,
        "point": True,
        "rotation": 0,
        "type": "",
        "visible": True,
    }


def main():
    # ---- Load GA tilemap ----
    print(f"Reading tilemap: {SRC}")
    with SRC.open("r", encoding="utf-8") as f:
        data = json.load(f)

    width = data["width"]
    height = data["height"]
    tw = data["tilewidth"]
    th = data["tileheight"]
    print(f"Map: {width}x{height} tiles, {tw}x{th}px each")

    layers = data["layers"]

    # ---- Load GA special_blocks CSVs ----
    sector_blocks = parse_blocks_csv(GA_MATRIX / "special_blocks" / "sector_blocks.csv")
    spawn_blocks = parse_spawning_blocks_csv(GA_MATRIX / "special_blocks" / "spawning_location_blocks.csv")
    print(f"Loaded {len(sector_blocks)} sector definitions, {len(spawn_blocks)} spawn definitions")

    # ---- Extract actual positions from tile layers ----
    sector_layer = find_layer(layers, "Sector Blocks")
    if not sector_layer:
        print("ERROR: No 'Sector Blocks' layer found")
        sys.exit(1)
    sector_data = sector_layer["data"]
    sector_tiles = extract_sector_tiles(sector_data, width, height, sector_blocks)

    spawn_layer = find_layer(layers, "Spawning Blocks")
    if not spawn_layer:
        print("ERROR: No 'Spawning Blocks' layer found")
        sys.exit(1)
    spawn_data = spawn_layer["data"]
    ga_spawns = extract_spawn_tiles(spawn_data, width, height, spawn_blocks)

    collisions_layer = find_layer(layers, "Collisions")
    if not collisions_layer:
        print("ERROR: No 'Collisions' layer found")
        sys.exit(1)
    collision_data = collisions_layer["data"]

    walkable_count = sum(1 for v in collision_data if int(v) == 0)
    blocked_count = sum(1 for v in collision_data if int(v) != 0)
    print(f"Collision layer: {walkable_count} walkable, {blocked_count} blocked tiles")

    # ---- Find connected components ----
    comp_map, comp_sizes = bfs_connected_components(collision_data, width, height)
    main_comp = max(comp_sizes, key=comp_sizes.get)
    print(f"\nConnected components: {len(comp_sizes)}")
    for cid in sorted(comp_sizes, key=comp_sizes.get, reverse=True)[:5]:
        print(f"  Component {cid}: {comp_sizes[cid]} tiles{'  <-- MAIN' if cid == main_comp else ''}")

    # ---- Build V-Valley tile layers ----
    bottom = find_layer(layers, "Bottom Ground")
    ground_layer = clone_tile_layer_as(bottom, "ground") if bottom else make_empty_tile_layer("ground", width, height)

    collision_layer = clone_tile_layer_as(collisions_layer, "collision")

    interior = find_layer(layers, "Interior Ground")
    interiors_layer = clone_tile_layer_as(interior, "interiors") if interior else make_empty_tile_layer("interiors", width, height)

    doors_layer = make_empty_tile_layer("doors", width, height)

    ext_decor = find_layer(layers, "Exterior Decoration L1")
    decor_layer = clone_tile_layer_as(ext_decor, "decor") if ext_decor else make_empty_tile_layer("decor", width, height)

    # ---- Create location objects from real sector data ----
    print(f"\n--- Extracting {len(sector_tiles)} sectors from tile data ---")
    obj_id = 1000
    location_objects = []

    for sector_name in sorted(sector_tiles.keys()):
        tiles = sector_tiles[sector_name]
        affordances = SECTOR_AFFORDANCES.get(sector_name)
        if affordances is None:
            print(f"  SKIP: {sector_name} (no affordance mapping)")
            continue

        center = sector_center(tiles)
        bbox = sector_bounding_box(tiles)
        tile_count = len(tiles)

        # Find a walkable tile in the main component near this sector
        wx, wy = find_walkable_in_component(tiles, collision_data, width, comp_map, main_comp, center)

        if wx is None:
            print(f"  WARN: {sector_name} - no walkable tile in main component, "
                  f"trying any walkable tile")
            # Fallback: just use any walkable tile in the sector
            for x, y in tiles:
                idx = y * width + x
                if int(collision_data[idx]) == 0:
                    wx, wy = x, y
                    break

        if wx is None:
            print(f"  ERROR: {sector_name} - no walkable tiles at all! Skipping.")
            continue

        # Create a small 3x3 tile object centered on the walkable position
        loc = make_location_object(obj_id, sector_name,
                                   max(0, wx - 1), max(0, wy - 1),
                                   3, 3, affordances, tw, th)
        location_objects.append(loc)
        obj_id += 1

        in_main = comp_map[wy][wx] == main_comp
        print(f"  Location: {sector_name}")
        print(f"    tiles={tile_count} bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) "
              f"center=({center[0]},{center[1]})")
        print(f"    placed=({wx},{wy}) in_main_component={in_main} affordances={affordances}")

    locations_group = make_object_group("locations", location_objects)
    print(f"\nTotal locations created: {len(location_objects)}")

    # ---- Create spawn objects from real spawn data ----
    print(f"\n--- Extracting spawns from tile data ({len(ga_spawns)} raw) ---")

    # GA spawn points are all inside buildings (disconnected interior components).
    # For V-Valley, we need spawns on walkable tiles in the main connected component.
    # Strategy: one spawn per sector, placed on the nearest main-component walkable
    # tile to the GA spawn position.
    seen_sectors = set()
    spawn_objects = []

    # Sort to prefer sp-A over sp-B (alphabetical puts sp-A first)
    ga_spawns_sorted = sorted(ga_spawns, key=lambda s: s[0])

    for spawn_name, sx, sy, sector in ga_spawns_sorted:
        if sector in seen_sectors:
            continue

        # Find nearest walkable tile in main component to this GA spawn
        best = None
        best_dist = float("inf")
        for radius in range(0, 30):
            if best is not None:
                break
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        nidx = ny * width + nx
                        if int(collision_data[nidx]) == 0 and comp_map[ny][nx] == main_comp:
                            dist = abs(dx) + abs(dy)
                            if dist < best_dist:
                                best = (nx, ny)
                                best_dist = dist
            if best is not None:
                break

        if best is None:
            print(f"  SKIP spawn: {spawn_name} - no reachable tile near ({sx},{sy})")
            continue

        spawn_obj = make_spawn_object(obj_id, spawn_name, best[0], best[1], tw, th)
        spawn_objects.append(spawn_obj)
        obj_id += 1
        seen_sectors.add(sector)
        print(f"  Spawn: {spawn_name} ga_pos=({sx},{sy}) placed=({best[0]},{best[1]}) "
              f"sector={sector}")

    spawns_group = make_object_group("spawns", spawn_objects)
    print(f"Total spawns created: {len(spawn_objects)}")

    # ---- Assemble final layer list ----
    new_layers = list(layers)  # preserve all original GA layers
    new_layers.extend([
        ground_layer,
        collision_layer,
        interiors_layer,
        doors_layer,
        decor_layer,
        locations_group,
        spawns_group,
    ])

    for i, layer in enumerate(new_layers, start=1):
        layer["id"] = i

    data["layers"] = new_layers

    # ---- Write output ----
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = DST.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {DST} ({size_mb:.1f} MB)")
    print(f"Total layers: {len(new_layers)}")
    print("Done!")


if __name__ == "__main__":
    main()
