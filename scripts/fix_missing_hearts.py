#!/usr/bin/env python3
"""Generate correct 6H and 9H card faces from existing hearts cards.

The hearts source sheet is missing the 6 and 9 of hearts, so the extraction
script mapped them to 7H and TH respectively.  This script creates proper
6H and 9H by:
  - 6H: copy 7H, erase the center heart, repaint rank "7" -> "6"
  - 9H: copy TH, erase one heart, repaint rank "10" -> "9"
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

FACES = Path(__file__).resolve().parents[1] / "apps/web/assets/scenarios/processed/anaconda/cards/faces"


def _sample_bg_color(img: Image.Image) -> tuple[int, int, int, int]:
    """Sample the dominant card background (cream) color from the center."""
    px = img.load()
    w, h = img.size
    samples: list[tuple[int, ...]] = []
    for y in range(h // 3, 2 * h // 3):
        for x in range(w // 3, 2 * w // 3):
            r, g, b, a = px[x, y]
            if a > 200 and r > 210 and g > 180 and b > 150:
                samples.append((r, g, b, a))
    if not samples:
        return (248, 243, 223, 255)
    # Return median color
    samples.sort()
    return samples[len(samples) // 2]  # type: ignore[return-value]


def _flood_fill_rect(img: Image.Image, x0: int, y0: int, x1: int, y1: int,
                     color: tuple[int, int, int, int],
                     threshold: int = 160) -> None:
    """Fill non-border (non-card-frame) pixels in a rect with color.

    Only fills pixels that are NOT part of the card border (dark red outline).
    Border pixels have R < threshold and are reddish.
    """
    px = img.load()
    w, h = img.size
    for y in range(max(0, y0), min(h, y1 + 1)):
        for x in range(max(0, x0), min(w, x1 + 1)):
            r, g, b, a = px[x, y]
            if a < 50:
                continue
            # Don't erase the card border (left/right edge columns)
            if x <= 5 or x >= w - 5:
                continue
            px[x, y] = color


def _paint_glyph(img: Image.Image, glyph: list[str], x0: int, y0: int,
                 fg: tuple[int, int, int, int]) -> None:
    """Paint a pixel-art glyph onto the image.  '#' = foreground pixel."""
    px = img.load()
    for dy, row in enumerate(glyph):
        for dx, ch in enumerate(row):
            if ch == '#':
                px[x0 + dx, y0 + dy] = fg


def _find_rank_color(img: Image.Image) -> tuple[int, int, int, int]:
    """Find the rank/heart text color (darkest red) in the top-left area."""
    px = img.load()
    darkest = (255, 0, 0, 255)
    min_brightness = 999
    for y in range(6, 16):
        for x in range(6, 16):
            r, g, b, a = px[x, y]
            if a > 100 and r < 180 and g < 100 and b < 100 and r > 30:
                brightness = r + g + b
                if brightness < min_brightness:
                    min_brightness = brightness
                    darkest = (r, g, b, a)
    return darkest


# Pixel-art glyphs for rank digits (approximately 6 wide x 7 tall)
GLYPH_6 = [
    "  ### ",
    " #    ",
    " #    ",
    " #### ",
    " #  # ",
    " #  # ",
    "  ### ",
]

GLYPH_9 = [
    " ###  ",
    " #  # ",
    " #  # ",
    " #### ",
    "    # ",
    "    # ",
    " ###  ",
]


def make_6h() -> None:
    src = Image.open(FACES / "7H.png").copy()
    bg = _sample_bg_color(src)
    fg = _find_rank_color(src)

    # 1) Erase the "7" rank label (top-left, rows 7-14, cols 6-15)
    _flood_fill_rect(src, 6, 7, 15, 15, bg)

    # 2) Paint "6" rank glyph
    _paint_glyph(src, GLYPH_6, 7, 8, fg)

    # 3) Erase the center single heart (rows 22-33, cols 18-38)
    _flood_fill_rect(src, 18, 22, 38, 33, bg)

    dst = FACES / "6H.png"
    src.save(dst)
    print(f"Wrote {dst}")


def make_9h() -> None:
    src = Image.open(FACES / "TH.png").copy()
    bg = _sample_bg_color(src)
    fg = _find_rank_color(src)

    # 1) Erase the "10" rank label (top-left, rows 7-16, cols 6-16)
    _flood_fill_rect(src, 6, 7, 16, 16, bg)

    # 2) Paint "9" rank glyph
    _paint_glyph(src, GLYPH_9, 7, 8, fg)

    # 3) TH layout is 2+1+2+2+1+2 = 10 hearts:
    #    Pair 1 (top):       rows 14-22, two columns
    #    Center heart 1:     rows 21-26, center column
    #    Pair 2:             rows 27-35, two columns
    #    Pair 3:             rows 39-48, two columns
    #    Center heart 2:     rows 46-52, center column
    #    Pair 4 (bottom):    rows 52-61, two columns
    #    Remove center heart 2 (rows 45-52, x 20-38) to get 9
    _flood_fill_rect(src, 20, 45, 38, 52, bg)

    dst = FACES / "9H.png"
    src.save(dst)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    make_6h()
    make_9h()
