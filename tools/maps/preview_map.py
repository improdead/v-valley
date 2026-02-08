#!/usr/bin/env python3
"""Render a compact ASCII preview for a V-Valley map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.vvalley_core.maps.map_utils import collision_grid, load_map, parse_location_objects

MARKERS = {
    "home": "H",
    "work": "W",
    "food": "F",
    "social": "S",
    "leisure": "L",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview map as ASCII")
    parser.add_argument("map_path", type=Path)
    args = parser.parse_args()

    map_data = load_map(args.map_path)
    collision = collision_grid(map_data)
    height = len(collision)
    width = len(collision[0]) if height else 0

    canvas = [["." for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if collision[y][x]:
                canvas[y][x] = "#"

    for location in parse_location_objects(map_data):
        marker = "?"
        for affordance in location.affordances:
            if affordance in MARKERS:
                marker = MARKERS[affordance]
                break
        canvas[location.y][location.x] = marker

    print(f"Map: {args.map_path}")
    print("Legend: # blocked, . walkable, H/W/F/S/L location affordances")
    for row in canvas:
        print("".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
