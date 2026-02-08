#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from packages.vvalley_core.maps.map_utils import load_map
from packages.vvalley_core.maps.validator import validate_map

ROOT = Path(__file__).resolve().parents[3]
STARTER_MAP = ROOT / "assets/templates/starter_town/map.json"
BAKE_SCRIPT = ROOT / "tools/maps/bake_nav.py"
CUSTOMIZE_SCRIPT = ROOT / "tools/maps/customize_map.py"
CUSTOMIZE_CONFIG = ROOT / "assets/templates/starter_town/customization.example.json"


class MapToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.map_data = load_map(STARTER_MAP)

    def test_starter_template_is_valid(self) -> None:
        errors, warnings, summary = validate_map(self.map_data)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertGreater(summary["tiles"]["walkable"], 0)

    def test_missing_required_affordance_fails(self) -> None:
        map_data = deepcopy(self.map_data)
        for layer in map_data.get("layers", []):
            if layer.get("name") != "locations":
                continue
            for obj in layer.get("objects", []):
                props = obj.get("properties", [])
                for prop in props:
                    if prop.get("name") == "affordances" and "work" in str(prop.get("value", "")):
                        prop["value"] = "social"

        errors, _, _ = validate_map(map_data)
        self.assertTrue(any("Missing required affordance location: work" in e for e in errors))

    def test_spawn_on_blocked_tile_fails(self) -> None:
        map_data = deepcopy(self.map_data)
        for layer in map_data.get("layers", []):
            if layer.get("name") == "spawns":
                layer["objects"][0]["x"] = 2 * int(map_data["tilewidth"])
                layer["objects"][0]["y"] = 2 * int(map_data["tileheight"])

        errors, _, _ = validate_map(map_data)
        self.assertTrue(any("placed on blocked tile" in e for e in errors))

    def test_bake_nav_produces_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "map.nav.json"
            result = subprocess.run(
                [
                    "python3",
                    str(BAKE_SCRIPT),
                    str(STARTER_MAP),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "vvalley-nav-v1")
            self.assertIn("walkable", payload)
            self.assertIn("collision", payload)
            self.assertEqual(len(payload["walkable"]), int(payload["dimensions"]["width"]) * int(payload["dimensions"]["height"]))

    def test_customize_map_applies_patch_and_stays_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_map = Path(td) / "map.custom.json"
            result = subprocess.run(
                [
                    "python3",
                    str(CUSTOMIZE_SCRIPT),
                    str(STARTER_MAP),
                    str(CUSTOMIZE_CONFIG),
                    "--out",
                    str(out_map),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            customized = load_map(out_map)
            errors, _, _ = validate_map(customized)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
