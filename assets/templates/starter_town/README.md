# Starter Town Template

Minimal template for V-Valley map authoring and validation.

## File

- `map.json`: base Tiled-compatible map definition

## Required layers

- `ground`
- `collision`
- `interiors`
- `doors`
- `decor`
- `locations` (object layer)
- `spawns` (object layer)

## Required affordance coverage

At least one location for each:
- `home`
- `work`
- `food`
- `social`
- `leisure`

Location affordances can be encoded via object properties:
- `affordances`: comma-separated list, e.g. `food,social`
- `role`: fallback single affordance, e.g. `work`

## Local checks

```bash
python3 tools/maps/validate_map.py assets/templates/starter_town/map.json
python3 tools/maps/bake_nav.py assets/templates/starter_town/map.json
python3 tools/maps/preview_map.py assets/templates/starter_town/map.json
```

## Customization example

```bash
python3 tools/maps/customize_map.py \
  assets/templates/starter_town/map.json \
  assets/templates/starter_town/customization.example.json \
  --out assets/templates/starter_town/map.custom.json
```
