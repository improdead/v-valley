# V-Valley Map Tools

CLI tooling for authoring and validating pixel-town maps.

## Core commands

Validate map structure and semantic affordances:

```bash
python3 tools/maps/validate_map.py <map.json>
```

Bake runtime navigation artifact:

```bash
python3 tools/maps/bake_nav.py <map.json> --out <map.nav.json>
```

ASCII map preview:

```bash
python3 tools/maps/preview_map.py <map.json>
```

Apply customization patch without hand-editing full JSON:

```bash
python3 tools/maps/customize_map.py <map.json> <customization.json> --out <map.custom.json>
```

Validate all templates:

```bash
python3 tools/maps/validate_templates.py
```

Run first-map end-to-end API workflow:

```bash
python3 tools/maps/run_first_map_e2e.py --town-id oakville_e2e
```

Run map tool tests:

```bash
python3 -m unittest discover -s tools/maps/tests -p 'test_*.py'
```

## Validation gates (`validate_map.py`)

- required layers exist and dimensions match
- affordance coverage exists (`home/work/food/social/leisure`)
- spawn and location tiles are walkable
- representative affordance locations are mutually reachable
- summary metrics are computed for operator review

## Recommended workflow

1. start from template
2. apply customization
3. validate
4. bake nav
5. publish map version
6. activate version for town runtime
