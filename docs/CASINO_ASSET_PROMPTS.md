# Casino Asset Prompts

Use these prompts to generate the new casino assets with consistent style.

Global constraints for all prompts:
- Style: cozy indie 16-bit pixel-art RPG UI
- Background: fully transparent alpha PNG (no checkerboard baked in)
- Lighting: soft, readable, high-contrast silhouette
- Output: clean single sprite (or a deliberate mini-sheet when specified)
- Avoid: collages, multiple unrelated objects, watermark text, blurred anti-aliased edges

## 1) Blackjack Icon

Target file:
- `apps/web/assets/scenarios/blackjack/icon.png`

Canvas:
- `640x640` RGBA

Prompt:
"Create a single pixel-art blackjack scenario icon on transparent background. Show a stylized playing card and poker chip motif, centered, bold silhouette, cozy indie RPG UI style, 16-bit pixel art, crisp outlines, no text, no border frame outside the icon shape. Export transparent PNG at 640x640."

## 2) Blackjack Actions Sheet

Target file:
- `apps/web/assets/scenarios/blackjack/ui/actions_sheet.png`

Canvas:
- `640x640` RGBA

Prompt:
"Create a pixel-art UI mini-sheet for blackjack actions with exactly three labeled tokens: HIT, STAND, DOUBLE. Arrange horizontally and centered with equal spacing, each token visually distinct and readable at small sizes, cozy indie RPG casino style, transparent background, crisp pixel edges, no extra icons beyond those three actions. Export transparent PNG at 640x640."

## 3) Hold'em Icon

Target file:
- `apps/web/assets/scenarios/holdem/icon.png`

Canvas:
- `640x640` RGBA

Prompt:
"Create a single pixel-art Texas Hold'em icon on transparent background. Theme: two hole cards plus chip/felt cue, centered composition, clean silhouette, warm casino palette, 16-bit RPG UI style, crisp outlines, no text. Export transparent PNG at 640x640."

## 4) Hold'em Markers Sheet

Target file:
- `apps/web/assets/scenarios/holdem/ui/markers_sheet.png`

Canvas:
- `640x640` RGBA

Prompt:
"Create a pixel-art UI mini-sheet for Hold'em positional markers with exactly three badges: Dealer (D), Small Blind (SB), Big Blind (BB). Arrange in one row, centered, equal spacing, readable tiny typography, consistent badge shape, transparent background, 16-bit cozy RPG casino style, crisp pixel edges, no extra elements. Export transparent PNG at 640x640."

## Post-generation checks

Before copying into repo, verify:
- PNG includes alpha channel and truly transparent background.
- Asset is a single intended icon/sheet, not a collage of unrelated variants.
- Pixel edges are crisp (no blurry scaling).
- Composition has enough padding to avoid clipping after extraction.

## Integration notes

After placing source files under `apps/web/assets/scenarios/...`, run:
- `python3 scripts/extract_scenario_sheet_assets.py`

Processed outputs should appear under:
- `apps/web/assets/scenarios/processed/blackjack/...`
- `apps/web/assets/scenarios/processed/holdem/...`
