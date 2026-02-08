# V-Valley Map Expansion Plan

## 1. Goal

Build map content in this order:
1. Keep the first map pipeline solid end-to-end.
2. Add a four-floor apartment map (`apartment_4f`).
3. Add a survival-camp map (`survival_camp`, zombie-survival vibe).

## 2. Current first-map baseline

Primary baseline map track:
- Legacy The Ville import path (`the_ville_legacy`)
- Starter template pipeline (`assets/templates/starter_town/map.json`)

Definition of done for first-map loop:
- template/scaffold (or legacy import)
- customization patch
- validation
- nav bake
- publish version
- activate version
- runtime state + tick against active version

Repro checks:

```bash
python3 tools/maps/validate_templates.py
python3 tools/maps/run_first_map_e2e.py --town-id oakville_e2e
```

## 3. Map A: `apartment_4f`

Target:
- Dense residential vertical social map.

Required semantics:
- homes on all four floors
- shared social hubs (lobby/lounge/rooftop)
- accessible food/work/leisure affordances
- clear floor transitions encoded in location metadata

## 4. Map B: `survival_camp`

Target:
- Compact fortified camp with survival social dynamics.

Required semantics:
- shelter/home
- workshop/work
- kitchen/food
- campfire/social
- training/leisure
- safe spawn connectivity and predictable collision boundaries

## 5. External premade map policy

Before importing external assets:
1. verify license and usage rights
2. record source URL and acquisition date
3. store provenance/checksum notes
4. confirm no bundled scripts/binaries
5. document attribution requirements

No external map is considered trusted before this checklist is complete.

## 6. Next map actions

1. Keep first-map E2E green while simulation logic evolves.
2. Scaffold `apartment_4f` template + validator coverage.
3. Scaffold `survival_camp` template + validator coverage.
4. Add both templates to automated template test runs.

Last updated: February 7, 2026
