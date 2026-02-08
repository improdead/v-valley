# V-Valley Town Templates

Templates are reusable starting points for towns.

## Folder convention

Each template directory should include:

- `assets/templates/<template_name>/map.json`
- optional `assets/templates/<template_name>/map.nav.json`
- optional `assets/templates/<template_name>/README.md`

## Quality gate

Before publishing a template-backed map version:

```bash
python3 tools/maps/validate_templates.py
```

Validate single template directly:

```bash
python3 tools/maps/validate_map.py assets/templates/<template_name>/map.json
```

## Current map roadmap

- First map baseline: The Ville legacy import + starter pipeline
- Next templates: `apartment_4f`, `survival_camp`

See: `MAP_EXPANSION_PLAN.md`
