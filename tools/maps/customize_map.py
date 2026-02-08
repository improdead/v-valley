#!/usr/bin/env python3
"""Apply no-code customization patches to a V-Valley map JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.vvalley_core.maps.customize import customize_map_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply no-code customizations to map JSON")
    parser.add_argument("map_path", type=Path, help="Source map JSON")
    parser.add_argument("config_path", type=Path, help="Customization JSON")
    parser.add_argument("--out", type=Path, default=None, help="Output path for customized map")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after patching",
    )
    args = parser.parse_args()

    notes, patch_errors, validation_errors, out = customize_map_file(
        args.map_path,
        args.config_path,
        out_path=args.out,
        validate_after=(not args.no_validate),
    )

    if patch_errors:
        for err in patch_errors:
            print(f"ERR: {err}")
        return 1

    if validation_errors:
        for err in validation_errors:
            print(f"ERR: {err}")
        return 1

    for note in notes:
        print(f"OK: {note}")
    print(f"Wrote customized map to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
