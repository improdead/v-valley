#!/usr/bin/env python3
"""Extract cropped transparent sprites from generated 640x640 scenario sheets.

The user-provided scenario assets are exported as padded canvases or sprite sheets.
This script normalizes them into per-sprite PNGs used by the web town viewer.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class SpriteCrop:
    src: str
    dst: str
    bbox: tuple[int, int, int, int]  # inclusive x0,y0,x1,y1
    pad: int = 8
    tolerance: int = 26
    required: bool = True


def _top_border_palette(img: Image.Image, *, bucket: int = 8, top_n: int = 8) -> list[tuple[int, int, int]]:
    width, height = img.size
    px = img.load()
    counts: Counter[tuple[int, int, int]] = Counter()

    for x in range(width):
        for y in (0, height - 1):
            r, g, b, _ = px[x, y]
            counts[(r // bucket * bucket, g // bucket * bucket, b // bucket * bucket)] += 1
    for y in range(height):
        for x in (0, width - 1):
            r, g, b, _ = px[x, y]
            counts[(r // bucket * bucket, g // bucket * bucket, b // bucket * bucket)] += 1

    return [color for color, _ in counts.most_common(top_n)]


def _is_near_palette(rgb: tuple[int, int, int], palette: Iterable[tuple[int, int, int]], tolerance: int) -> bool:
    r, g, b = rgb
    max_dist_sq = tolerance * tolerance
    for pr, pg, pb in palette:
        dr = r - pr
        dg = g - pg
        db = b - pb
        if dr * dr + dg * dg + db * db <= max_dist_sq:
            return True
    return False


def _remove_border_background(img: Image.Image, *, tolerance: int = 26) -> Image.Image:
    """Make checkerboard/padded border background transparent via border flood fill."""

    rgba = img.convert("RGBA")
    width, height = rgba.size
    px = rgba.load()
    palette = _top_border_palette(rgba)

    visited = [[False for _ in range(width)] for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    def maybe_seed(x: int, y: int) -> None:
        if visited[y][x]:
            return
        if _is_near_palette(px[x, y][:3], palette, tolerance):
            visited[y][x] = True
            queue.append((x, y))

    for x in range(width):
        maybe_seed(x, 0)
        maybe_seed(x, height - 1)
    for y in range(height):
        maybe_seed(0, y)
        maybe_seed(width - 1, y)

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny][nx]:
                continue
            if _is_near_palette(px[nx, ny][:3], palette, tolerance):
                visited[ny][nx] = True
                queue.append((nx, ny))

    out = rgba.copy()
    out_px = out.load()
    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                r, g, b, _ = out_px[x, y]
                out_px[x, y] = (r, g, b, 0)

    return out


def _trim_alpha(img: Image.Image, *, pad: int = 1) -> Image.Image:
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return img

    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.width, right + pad)
    bottom = min(img.height, bottom + pad)
    return img.crop((left, top, right, bottom))


def _extract(src_root: Path, dst_root: Path, spec: SpriteCrop) -> Path | None:
    src_path = src_root / spec.src
    if not src_path.exists():
        if not spec.required:
            print(f"Skipping missing optional source: {src_path}")
            return None
        raise FileNotFoundError(f"Missing source: {src_path}")

    image = Image.open(src_path).convert("RGBA")
    x0, y0, x1, y1 = spec.bbox

    x0 = max(0, x0 - spec.pad)
    y0 = max(0, y0 - spec.pad)
    x1 = min(image.width - 1, x1 + spec.pad)
    y1 = min(image.height - 1, y1 + spec.pad)

    cropped = image.crop((x0, y0, x1 + 1, y1 + 1))
    transparent = _remove_border_background(cropped, tolerance=spec.tolerance)
    cleaned = _trim_alpha(transparent, pad=1)

    dst_path = dst_root / spec.dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(dst_path)
    return dst_path


def _specs() -> list[SpriteCrop]:
    out: list[SpriteCrop] = []

    role_bbox = (128, 45, 511, 588)
    for role in (
        "role_villager.png",
        "role_werewolf.png",
        "role_seer.png",
        "role_doctor.png",
        "role_hunter.png",
        "role_witch.png",
        "role_alpha.png",
    ):
        out.append(
            SpriteCrop(
                src=f"werewolf/cards/{role}",
                dst=f"werewolf/cards/{role}",
                bbox=role_bbox,
                pad=8,
                tolerance=24,
            )
        )

    out.extend(
        [
            SpriteCrop(
                src="werewolf/icon.png",
                dst="werewolf/icon.png",
                bbox=(160, 84, 482, 416),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/board/campfire_circle.png",
                dst="werewolf/board/campfire_circle.png",
                bbox=(32, 32, 607, 607),
                pad=8,
                tolerance=26,
            ),
            SpriteCrop(
                src="werewolf/ui/tombstone.png",
                dst="werewolf/ui/tombstone.png",
                bbox=(205, 128, 434, 485),
                pad=8,
                tolerance=28,
            ),
            SpriteCrop(
                src="werewolf/ui/phase_day_night.png",
                dst="werewolf/ui/phase_day.png",
                bbox=(31, 100, 308, 373),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/ui/phase_day_night.png",
                dst="werewolf/ui/phase_night.png",
                bbox=(332, 102, 607, 371),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/ui/vote_tokens.png",
                dst="werewolf/ui/token_accuse.png",
                bbox=(41, 54, 278, 291),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/ui/vote_tokens.png",
                dst="werewolf/ui/token_defend.png",
                bbox=(361, 54, 598, 291),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/ui/vote_tokens.png",
                dst="werewolf/ui/token_skip.png",
                bbox=(39, 349, 280, 594),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="werewolf/ui/vote_tokens.png",
                dst="werewolf/ui/token_save.png",
                bbox=(357, 347, 601, 595),
                pad=8,
                tolerance=24,
            ),
        ]
    )

    out.extend(
        [
            SpriteCrop(
                src="anaconda/icon.png",
                dst="anaconda/icon.png",
                bbox=(22, 21, 618, 618),
                pad=8,
                tolerance=26,
            ),
            SpriteCrop(
                src="anaconda/board/poker_table.png",
                dst="anaconda/board/poker_table.png",
                bbox=(19, 102, 620, 537),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/cards/card_back.png",
                dst="anaconda/cards/card_back.png",
                bbox=(127, 49, 512, 590),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/chips/poker_chips.png",
                dst="anaconda/chips/chip_1.png",
                bbox=(7, 256, 128, 383),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/chips/poker_chips.png",
                dst="anaconda/chips/chip_5.png",
                bbox=(132, 255, 254, 384),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/chips/poker_chips.png",
                dst="anaconda/chips/chip_10.png",
                bbox=(258, 256, 381, 383),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/chips/poker_chips.png",
                dst="anaconda/chips/chip_25.png",
                bbox=(386, 256, 507, 383),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/chips/poker_chips.png",
                dst="anaconda/chips/chip_100.png",
                bbox=(512, 256, 633, 383),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/ui/betting_buttons.png",
                dst="anaconda/ui/button_fold.png",
                bbox=(52, 125, 301, 227),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/ui/betting_buttons.png",
                dst="anaconda/ui/button_check.png",
                bbox=(338, 126, 586, 227),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/ui/betting_buttons.png",
                dst="anaconda/ui/button_call.png",
                bbox=(196, 270, 437, 371),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/ui/betting_buttons.png",
                dst="anaconda/ui/button_raise.png",
                bbox=(52, 420, 301, 519),
                pad=8,
                tolerance=24,
            ),
            SpriteCrop(
                src="anaconda/ui/betting_buttons.png",
                dst="anaconda/ui/button_all_in.png",
                bbox=(332, 414, 594, 525),
                pad=8,
                tolerance=24,
            ),
        ]
    )

    out.extend(
        [
            SpriteCrop(
                src="blackjack/icon.png",
                dst="blackjack/icon.png",
                bbox=(0, 0, 639, 639),
                pad=0,
                tolerance=26,
                required=False,
            ),
            SpriteCrop(
                src="blackjack/ui/actions_sheet.png",
                dst="blackjack/ui/actions_sheet.png",
                bbox=(0, 0, 639, 639),
                pad=0,
                tolerance=26,
                required=False,
            ),
            SpriteCrop(
                src="holdem/icon.png",
                dst="holdem/icon.png",
                bbox=(0, 0, 639, 639),
                pad=0,
                tolerance=26,
                required=False,
            ),
            SpriteCrop(
                src="holdem/ui/markers_sheet.png",
                dst="holdem/ui/markers_sheet.png",
                bbox=(0, 0, 639, 639),
                pad=0,
                tolerance=26,
                required=False,
            ),
        ]
    )

    rank_boxes = [
        (46, 14, 593, 71),
        (46, 77, 593, 133),
        (46, 140, 593, 197),
        (45, 202, 594, 260),
        (45, 266, 594, 323),
        (45, 329, 594, 386),
        (44, 392, 595, 448),
        (45, 455, 594, 510),
        (44, 515, 595, 574),
        (43, 579, 596, 635),
    ]
    rank_names = [
        "high_card",
        "one_pair",
        "two_pair",
        "three_kind",
        "straight",
        "flush",
        "full_house",
        "four_kind",
        "straight_flush",
        "royal_flush",
    ]
    for name, bbox in zip(rank_names, rank_boxes):
        out.append(
            SpriteCrop(
                src="anaconda/ui/hand_ranks.png",
                dst=f"anaconda/ui/rank_{name}.png",
                bbox=bbox,
                pad=6,
                tolerance=24,
            )
        )

    # Per-card face sprites (some source sheets contain duplicates/missing ranks;
    # missing entries are mapped to nearest available art so every rank token resolves).
    def add_face_specs(
        *,
        sheet: str,
        suit: str,
        rank_to_bbox: dict[str, tuple[int, int, int, int]],
        pad: int,
        tolerance: int,
    ) -> None:
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"):
            bbox = rank_to_bbox.get(rank)
            if bbox is None:
                continue
            out.append(
                SpriteCrop(
                    src=f"anaconda/cards/{sheet}",
                    dst=f"anaconda/cards/faces/{rank}{suit}.png",
                    bbox=bbox,
                    pad=pad,
                    tolerance=tolerance,
                )
            )

    add_face_specs(
        sheet="spades_sheet.png",
        suit="S",
        rank_to_bbox={
            "A": (13, 45, 133, 221),
            "2": (136, 45, 256, 221),
            "3": (260, 44, 379, 221),
            "4": (506, 44, 627, 222),
            "5": (12, 234, 132, 409),
            "6": (136, 233, 256, 410),
            "7": (260, 234, 379, 409),
            "8": (383, 233, 503, 409),
            "9": (505, 233, 627, 409),
            "T": (136, 425, 256, 601),
            "J": (260, 425, 379, 601),
            "Q": (383, 425, 503, 601),
            "K": (505, 425, 626, 601),
        },
        pad=4,
        tolerance=24,
    )
    add_face_specs(
        sheet="diamonds_sheet.png",
        suit="D",
        rank_to_bbox={
            "A": (4, 29, 128, 211),
            "2": (131, 29, 254, 211),
            "3": (259, 28, 380, 211),
            "4": (384, 29, 508, 211),
            "5": (511, 29, 635, 211),
            "6": (4, 232, 128, 410),
            "7": (131, 232, 255, 410),
            "8": (258, 232, 380, 410),
            "9": (384, 232, 508, 410),
            "T": (511, 232, 635, 410),
            "J": (131, 440, 255, 616),
            "Q": (257, 440, 380, 616),
            "K": (384, 440, 508, 616),
        },
        pad=4,
        tolerance=24,
    )
    add_face_specs(
        sheet="clubs_sheet.png",
        suit="C",
        rank_to_bbox={
            "A": (22, 36, 133, 207),
            "2": (145, 35, 253, 207),
            "3": (266, 35, 373, 207),
            "4": (506, 35, 617, 207),
            "5": (22, 234, 133, 405),
            "6": (145, 234, 253, 405),
            "7": (266, 234, 373, 405),
            "8": (386, 234, 493, 405),
            "9": (506, 234, 617, 405),
            "T": (70, 432, 184, 603),
            "J": (196, 432, 312, 603),
            "Q": (326, 432, 442, 603),
            "K": (455, 432, 570, 603),
        },
        pad=4,
        tolerance=24,
    )
    add_face_specs(
        sheet="hearts_sheet.png",
        suit="H",
        rank_to_bbox={
            "A": (10, 284, 61, 354),
            "2": (67, 285, 118, 354),
            "3": (124, 285, 175, 354),
            "4": (181, 285, 232, 354),
            "5": (238, 285, 288, 354),
            "6": (294, 285, 345, 354),  # source sheet misses 6; use nearest card art
            "7": (294, 285, 345, 354),
            "8": (351, 285, 402, 354),
            "9": (408, 285, 458, 354),  # source sheet misses 9; use nearest card art
            "T": (408, 285, 458, 354),
            "J": (464, 285, 515, 354),
            "Q": (521, 285, 572, 354),
            "K": (578, 285, 629, 354),
        },
        pad=2,
        tolerance=24,
    )

    out.extend(
        [
            SpriteCrop(
                src="shared/frames/queue_panel.png",
                dst="shared/frames/queue_panel.png",
                bbox=(11, 135, 628, 520),
                pad=10,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_starter.png",
                bbox=(21, 264, 119, 375),
                pad=4,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_bronze.png",
                bbox=(123, 264, 218, 374),
                pad=4,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_silver.png",
                bbox=(223, 265, 316, 374),
                pad=4,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_gold.png",
                bbox=(322, 264, 416, 374),
                pad=4,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_platinum.png",
                bbox=(422, 265, 515, 373),
                pad=4,
                tolerance=24,
            ),
            SpriteCrop(
                src="shared/frames/rating_badges.png",
                dst="shared/frames/badge_diamond.png",
                bbox=(520, 264, 618, 375),
                pad=4,
                tolerance=24,
            ),
        ]
    )

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract processed scenario sprites from sheets")
    parser.add_argument(
        "--src",
        default="apps/web/assets/scenarios",
        help="Source scenario assets directory",
    )
    parser.add_argument(
        "--out",
        default="apps/web/assets/scenarios/processed",
        help="Output directory for processed assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    src_root = (repo_root / args.src).resolve()
    out_root = (repo_root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    specs = _specs()
    written: list[Path] = []
    skipped = 0
    for spec in specs:
        dst = _extract(src_root, out_root, spec)
        if dst is None:
            skipped += 1
            continue
        written.append(dst)

    print(f"Extracted {len(written)} sprites into {out_root} ({skipped} optional skipped)")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
