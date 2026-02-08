#!/usr/bin/env python3
"""Validate all map templates under assets/templates."""

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
    parser = argparse.ArgumentParser(description="Validate all V-Valley templates")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("assets/templates"),
        help="Template root directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary",
    )
    args = parser.parse_args()

    map_paths = sorted(args.root.glob("*/map.json"))
    results: list[dict[str, object]] = []

    for map_path in map_paths:
        map_data = load_map(map_path)
        errors, warnings, summary = validate_map(map_data)
        results.append(
            {
                "template": str(map_path.parent),
                "map": str(map_path),
                "ok": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "summary": summary,
            }
        )

    failed = [r for r in results if not r["ok"]]
    payload = {
        "templates": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Validated {payload['templates']} template(s): "
            f"{payload['passed']} passed, {payload['failed']} failed"
        )
        for result in results:
            status = "OK" if result["ok"] else "FAIL"
            print(f"[{status}] {result['map']}")
            for warning in result["warnings"]:
                print(f"  WARN: {warning}")
            for error in result["errors"]:
                print(f"  ERR: {error}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
