#!/usr/bin/env python3
"""Bake lightweight navigation data from a V-Valley map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.vvalley_core.maps.nav import bake_nav_to_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake nav graph data for a V-Valley map")
    parser.add_argument("map_path", type=Path, help="Path to map JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output nav JSON path (default: <map_name>.nav.json next to map)",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Bake even if validator reports errors",
    )
    args = parser.parse_args()

    errors, _, _, out_path = bake_nav_to_file(
        args.map_path,
        out_path=args.out,
        allow_invalid=args.allow_invalid,
    )

    if errors and not args.allow_invalid:
        print("ERROR: map is invalid; refusing to bake nav data")
        for error in errors:
            print(f"ERR: {error}")
        return 1

    print(f"Wrote nav data to {out_path}")
    if errors:
        print("WARN: baked from invalid map due to --allow-invalid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
