# Starter Town Customization

Use `customization.example.json` as a no-code patch file.

## Apply customization

```bash
python3 tools/maps/customize_map.py \
  assets/templates/starter_town/map.json \
  assets/templates/starter_town/customization.example.json \
  --out assets/templates/starter_town/map.custom.json
```

Then validate/bake:

```bash
python3 tools/maps/validate_map.py assets/templates/starter_town/map.custom.json
python3 tools/maps/bake_nav.py assets/templates/starter_town/map.custom.json \
  --out assets/templates/starter_town/map.custom.nav.json
```

## Supported fields

- `map_properties`: key/value map-level metadata
- `locations[]`: patch by location name
  - `affordances`: list of semantic roles
  - `hours`: string like `07:00-21:00`
  - `zone_rules`: list (e.g. `quiet_zone`, `no_chat_zone`)
- `spawns[]`: patch by spawn name
  - `x`, `y` in tile coordinates
