#!/usr/bin/env python3
"""Validate a V-Valley map for simulation and agent semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.vvalley_core.maps.map_utils import load_map
from packages.vvalley_core.maps.validator import validate_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a V-Valley map JSON")
    parser.add_argument("map_path", type=Path, help="Path to map JSON")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )
    args = parser.parse_args()

    map_data = load_map(args.map_path)
    errors, warnings, summary = validate_map(map_data)

    payload = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if payload["ok"]:
            print("OK: map validation passed")
        else:
            print("ERROR: map validation failed")

        for warning in warnings:
            print(f"WARN: {warning}")
        for error in errors:
            print(f"ERR: {error}")

        if summary:
            print("Summary:")
            print(json.dumps(summary, indent=2))

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
