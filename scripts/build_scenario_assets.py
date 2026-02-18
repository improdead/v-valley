#!/usr/bin/env python3
"""Build deterministic pixel-art atlases for scenario spectator UI.

This script intentionally avoids third-party dependencies so it can run in CI
and local dev without imaging toolchains.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

Color = tuple[int, int, int, int]


PALETTE: dict[str, Color] = {
    "transparent": (0, 0, 0, 0),
    "night_0": (8, 18, 32, 255),
    "night_1": (15, 31, 52, 255),
    "night_2": (34, 54, 80, 255),
    "wood": (95, 67, 42, 255),
    "ember": (233, 69, 96, 255),
    "fire": (255, 168, 76, 255),
    "sun": (255, 213, 107, 255),
    "cyan": (83, 216, 251, 255),
    "leaf_0": (10, 47, 34, 255),
    "leaf_1": (22, 73, 55, 255),
    "leaf_2": (48, 124, 90, 255),
    "silver": (176, 194, 215, 255),
    "slate": (59, 89, 122, 255),
    "white": (239, 244, 252, 255),
    "ink": (19, 29, 45, 255),
}


@dataclass(frozen=True)
class FrameSpec:
    name: str
    draw: Callable[["Canvas"], None]


class Canvas:
    def __init__(self, width: int, height: int, fill: Color) -> None:
        self.width = int(width)
        self.height = int(height)
        self.pixels: list[Color] = [fill for _ in range(self.width * self.height)]

    def set_px(self, x: int, y: int, color: Color) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        self.pixels[y * self.width + x] = color

    def fill(self, color: Color) -> None:
        for i in range(len(self.pixels)):
            self.pixels[i] = color

    def fill_rect(self, x: int, y: int, w: int, h: int, color: Color) -> None:
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.set_px(xx, yy, color)

    def stroke_rect(self, x: int, y: int, w: int, h: int, color: Color) -> None:
        for xx in range(x, x + w):
            self.set_px(xx, y, color)
            self.set_px(xx, y + h - 1, color)
        for yy in range(y, y + h):
            self.set_px(x, yy, color)
            self.set_px(x + w - 1, yy, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x = x0
        y = y0
        while True:
            self.set_px(x, y, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def circle(self, cx: int, cy: int, r: int, color: Color, *, fill: bool = False) -> None:
        if r <= 0:
            return
        if fill:
            for y in range(cy - r, cy + r + 1):
                for x in range(cx - r, cx + r + 1):
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r:
                        self.set_px(x, y, color)
            return
        x = r
        y = 0
        err = 0
        while x >= y:
            points = (
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            )
            for px, py in points:
                self.set_px(px, py, color)
            y += 1
            if err <= 0:
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

    def blit(self, other: "Canvas", x: int, y: int) -> None:
        for yy in range(other.height):
            for xx in range(other.width):
                color = other.pixels[yy * other.width + xx]
                if color[3] == 0:
                    continue
                self.set_px(x + xx, y + yy, color)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: Iterable[Color]) -> None:
    raw = bytearray()
    px = list(pixels)
    for y in range(height):
        raw.append(0)  # no filter
        row = px[y * width : (y + 1) * width]
        for r, g, b, a in row:
            raw.extend((r & 0xFF, g & 0xFF, b & 0xFF, a & 0xFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    data = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(data)


def bg(canvas: Canvas, top: Color, bottom: Color) -> None:
    h = max(1, canvas.height - 1)
    for y in range(canvas.height):
        t = y / h
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        canvas.fill_rect(0, y, canvas.width, 1, (r, g, b, 255))


def icon_campfire(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(10, 20, 12, 3, PALETTE["wood"])
    c.fill_rect(8, 23, 16, 2, PALETTE["ink"])
    c.fill_rect(14, 10, 4, 10, PALETTE["fire"])
    c.fill_rect(13, 14, 6, 4, PALETTE["sun"])
    c.circle(16, 17, 3, PALETTE["ember"], fill=True)


def icon_moon(c: Canvas) -> None:
    bg(c, PALETTE["night_1"], PALETTE["night_0"])
    c.circle(16, 14, 8, PALETTE["sun"], fill=True)
    c.circle(19, 12, 8, PALETTE["night_1"], fill=True)
    c.circle(16, 14, 8, PALETTE["silver"])


def icon_wolf(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(9, 9, 14, 12, PALETTE["ink"])
    c.line(9, 9, 4, 3, PALETTE["ink"])
    c.line(22, 9, 27, 3, PALETTE["ink"])
    c.fill_rect(12, 13, 2, 2, PALETTE["ember"])
    c.fill_rect(18, 13, 2, 2, PALETTE["ember"])
    c.fill_rect(15, 18, 2, 2, PALETTE["white"])


def icon_villager(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_1"])
    c.fill_rect(10, 7, 12, 18, PALETTE["leaf_2"])
    c.stroke_rect(10, 7, 12, 18, PALETTE["white"])
    c.circle(16, 14, 3, PALETTE["white"], fill=True)
    c.fill_rect(15, 17, 2, 5, PALETTE["white"])


def icon_seer(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(6, 12, 20, 8, PALETTE["white"])
    c.circle(16, 16, 4, PALETTE["cyan"], fill=True)
    c.circle(16, 16, 2, PALETTE["ink"], fill=True)
    c.stroke_rect(6, 12, 20, 8, PALETTE["ink"])


def icon_doctor(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_1"])
    c.fill_rect(11, 8, 10, 16, PALETTE["white"])
    c.fill_rect(8, 11, 16, 10, PALETTE["white"])
    c.fill_rect(14, 8, 4, 16, PALETTE["ember"])
    c.fill_rect(8, 14, 16, 4, PALETTE["ember"])


def icon_vote(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_1"])
    c.fill_rect(8, 20, 16, 4, PALETTE["wood"])
    c.fill_rect(20, 8, 5, 13, PALETTE["wood"])
    c.fill_rect(11, 6, 9, 8, PALETTE["silver"])
    c.stroke_rect(11, 6, 9, 8, PALETTE["ink"])


def icon_skull(c: Canvas) -> None:
    bg(c, PALETTE["night_1"], PALETTE["night_0"])
    c.circle(16, 14, 8, PALETTE["silver"], fill=True)
    c.fill_rect(11, 18, 10, 6, PALETTE["silver"])
    c.fill_rect(13, 12, 2, 3, PALETTE["ink"])
    c.fill_rect(17, 12, 2, 3, PALETTE["ink"])
    c.fill_rect(15, 16, 2, 2, PALETTE["ink"])


def icon_day_banner(c: Canvas) -> None:
    bg(c, PALETTE["cyan"], PALETTE["silver"])
    c.fill_rect(2, 10, 28, 12, PALETTE["sun"])
    c.stroke_rect(2, 10, 28, 12, PALETTE["ink"])
    c.line(5, 13, 27, 13, PALETTE["ink"])
    c.line(5, 18, 27, 18, PALETTE["ink"])


def icon_night_banner(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(2, 10, 28, 12, PALETTE["slate"])
    c.stroke_rect(2, 10, 28, 12, PALETTE["silver"])
    c.circle(11, 16, 3, PALETTE["sun"], fill=True)
    c.circle(12, 15, 3, PALETTE["slate"], fill=True)


def icon_table(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(4, 9, 24, 14, PALETTE["leaf_1"])
    c.stroke_rect(4, 9, 24, 14, PALETTE["sun"])


def icon_blackjack(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(7, 5, 12, 20, PALETTE["white"])
    c.stroke_rect(7, 5, 12, 20, PALETTE["ink"])
    c.fill_rect(10, 8, 6, 6, PALETTE["ember"])
    c.fill_rect(19, 9, 8, 8, PALETTE["sun"])
    c.circle(23, 13, 4, PALETTE["sun"], fill=True)
    c.circle(23, 13, 4, PALETTE["ink"])


def icon_holdem(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(5, 7, 10, 16, PALETTE["white"])
    c.fill_rect(17, 9, 10, 16, PALETTE["white"])
    c.stroke_rect(5, 7, 10, 16, PALETTE["ink"])
    c.stroke_rect(17, 9, 10, 16, PALETTE["ink"])
    c.fill_rect(8, 10, 4, 4, PALETTE["cyan"])
    c.fill_rect(20, 12, 4, 4, PALETTE["ember"])
    c.circle(16, 24, 4, PALETTE["sun"], fill=True)
    c.circle(16, 24, 4, PALETTE["ink"])


def icon_bj_hit(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(5, 14, 22, 4, PALETTE["cyan"])
    c.fill_rect(14, 5, 4, 22, PALETTE["cyan"])
    c.stroke_rect(4, 4, 24, 24, PALETTE["ink"])


def icon_bj_stand(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.line(6, 17, 14, 24, PALETTE["sun"])
    c.line(14, 24, 26, 8, PALETTE["sun"])
    c.line(6, 16, 14, 23, PALETTE["sun"])
    c.line(14, 23, 26, 7, PALETTE["sun"])
    c.stroke_rect(4, 4, 24, 24, PALETTE["ink"])


def icon_bj_double(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.circle(12, 16, 6, PALETTE["ember"], fill=True)
    c.circle(20, 16, 6, PALETTE["sun"], fill=True)
    c.circle(12, 16, 6, PALETTE["ink"])
    c.circle(20, 16, 6, PALETTE["ink"])
    c.stroke_rect(4, 4, 24, 24, PALETTE["ink"])


def icon_bj_actions_sheet(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(2, 12, 7, 2, PALETTE["cyan"])  # HIT bar
    c.fill_rect(4, 10, 2, 7, PALETTE["cyan"])  # HIT stem
    c.line(12, 13, 14, 15, PALETTE["sun"])     # STAND check
    c.line(14, 15, 18, 10, PALETTE["sun"])
    c.circle(24, 13, 4, PALETTE["ember"], fill=True)  # DOUBLE chips
    c.circle(27, 17, 4, PALETTE["sun"], fill=True)
    c.circle(24, 13, 4, PALETTE["ink"])
    c.circle(27, 17, 4, PALETTE["ink"])
    c.stroke_rect(1, 1, 30, 30, PALETTE["ink"])


def icon_marker_dealer(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.circle(16, 16, 10, PALETTE["sun"], fill=True)
    c.circle(16, 16, 10, PALETTE["ink"])
    c.fill_rect(10, 11, 2, 10, PALETTE["ink"])
    c.line(12, 11, 19, 14, PALETTE["ink"])
    c.line(12, 21, 19, 18, PALETTE["ink"])


def icon_marker_sb(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.circle(16, 16, 10, PALETTE["silver"], fill=True)
    c.circle(16, 16, 10, PALETTE["ink"])
    c.fill_rect(9, 11, 2, 10, PALETTE["ink"])
    c.fill_rect(12, 11, 7, 2, PALETTE["ink"])
    c.fill_rect(12, 15, 6, 2, PALETTE["ink"])
    c.fill_rect(12, 19, 7, 2, PALETTE["ink"])


def icon_marker_bb(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.circle(16, 16, 10, PALETTE["cyan"], fill=True)
    c.circle(16, 16, 10, PALETTE["ink"])
    c.fill_rect(9, 11, 2, 10, PALETTE["ink"])
    c.fill_rect(12, 11, 6, 2, PALETTE["ink"])
    c.fill_rect(12, 15, 5, 2, PALETTE["ink"])
    c.fill_rect(12, 19, 6, 2, PALETTE["ink"])
    c.fill_rect(18, 13, 2, 6, PALETTE["ink"])


def icon_holdem_markers_sheet(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.circle(7, 16, 6, PALETTE["sun"], fill=True)
    c.circle(16, 16, 6, PALETTE["silver"], fill=True)
    c.circle(25, 16, 6, PALETTE["cyan"], fill=True)
    c.circle(7, 16, 6, PALETTE["ink"])
    c.circle(16, 16, 6, PALETTE["ink"])
    c.circle(25, 16, 6, PALETTE["ink"])
    c.fill_rect(5, 14, 2, 4, PALETTE["ink"])   # D
    c.fill_rect(14, 13, 3, 2, PALETTE["ink"])  # SB
    c.fill_rect(14, 17, 3, 2, PALETTE["ink"])
    c.fill_rect(23, 13, 3, 2, PALETTE["ink"])  # BB
    c.fill_rect(23, 17, 3, 2, PALETTE["ink"])
    c.fill_rect(25, 15, 1, 2, PALETTE["ink"])
    c.stroke_rect(1, 1, 30, 30, PALETTE["ink"])


def icon_card_back(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.fill_rect(8, 5, 16, 22, PALETTE["white"])
    c.stroke_rect(8, 5, 16, 22, PALETTE["ink"])
    c.fill_rect(11, 8, 10, 16, PALETTE["cyan"])
    c.stroke_rect(11, 8, 10, 16, PALETTE["ink"])


def icon_chip_stack(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.circle(12, 19, 5, PALETTE["ember"], fill=True)
    c.circle(16, 15, 5, PALETTE["sun"], fill=True)
    c.circle(20, 11, 5, PALETTE["cyan"], fill=True)
    c.circle(12, 19, 5, PALETTE["ink"])
    c.circle(16, 15, 5, PALETTE["ink"])
    c.circle(20, 11, 5, PALETTE["ink"])


def icon_pot(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.fill_rect(10, 10, 12, 14, PALETTE["wood"])
    c.stroke_rect(10, 10, 12, 14, PALETTE["ink"])
    c.line(10, 10, 16, 5, PALETTE["wood"])
    c.line(22, 10, 16, 5, PALETTE["wood"])
    c.fill_rect(14, 14, 4, 4, PALETTE["sun"])


def icon_fold(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(8, 5, 16, 22, PALETTE["white"])
    c.stroke_rect(8, 5, 16, 22, PALETTE["ink"])
    c.line(7, 25, 25, 7, PALETTE["ember"])
    c.line(7, 24, 24, 7, PALETTE["ember"])


def icon_all_in(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(14, 6, 4, 14, PALETTE["sun"])
    c.fill_rect(10, 10, 12, 4, PALETTE["sun"])
    c.line(10, 20, 16, 27, PALETTE["sun"])
    c.line(16, 27, 22, 20, PALETTE["sun"])


def icon_reveal(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(6, 12, 20, 8, PALETTE["white"])
    c.circle(16, 16, 3, PALETTE["ink"], fill=True)
    c.stroke_rect(6, 12, 20, 8, PALETTE["ink"])


def icon_arrow_left(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(8, 14, 16, 4, PALETTE["cyan"])
    c.line(8, 16, 14, 10, PALETTE["cyan"])
    c.line(8, 16, 14, 22, PALETTE["cyan"])


def icon_arrow_right(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_0"])
    c.fill_rect(8, 14, 16, 4, PALETTE["cyan"])
    c.line(24, 16, 18, 10, PALETTE["cyan"])
    c.line(24, 16, 18, 22, PALETTE["cyan"])


def icon_crown(c: Canvas) -> None:
    bg(c, PALETTE["leaf_1"], PALETTE["leaf_0"])
    c.fill_rect(7, 14, 18, 10, PALETTE["sun"])
    c.line(7, 14, 11, 7, PALETTE["sun"])
    c.line(16, 14, 16, 6, PALETTE["sun"])
    c.line(25, 14, 21, 7, PALETTE["sun"])
    c.stroke_rect(7, 14, 18, 10, PALETTE["ink"])


def icon_queue_panel(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.fill_rect(5, 5, 22, 22, PALETTE["slate"])
    c.stroke_rect(5, 5, 22, 22, PALETTE["cyan"])
    c.line(8, 11, 24, 11, PALETTE["white"])
    c.line(8, 16, 24, 16, PALETTE["white"])
    c.line(8, 21, 20, 21, PALETTE["white"])


def icon_badge(color: Color) -> Callable[[Canvas], None]:
    def _draw(c: Canvas) -> None:
        bg(c, PALETTE["night_2"], PALETTE["night_0"])
        c.circle(16, 16, 10, color, fill=True)
        c.circle(16, 16, 10, PALETTE["white"])
        c.fill_rect(15, 9, 2, 14, PALETTE["ink"])
        c.fill_rect(9, 15, 14, 2, PALETTE["ink"])

    return _draw


def icon_thinking(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.circle(16, 15, 9, PALETTE["white"], fill=True)
    c.fill_rect(12, 12, 2, 2, PALETTE["ink"])
    c.fill_rect(18, 12, 2, 2, PALETTE["ink"])
    c.line(12, 19, 20, 17, PALETTE["ink"])
    c.circle(24, 7, 2, PALETTE["cyan"], fill=True)


def icon_winning(c: Canvas) -> None:
    bg(c, PALETTE["leaf_2"], PALETTE["leaf_1"])
    c.circle(16, 15, 9, PALETTE["white"], fill=True)
    c.fill_rect(12, 12, 2, 2, PALETTE["ink"])
    c.fill_rect(18, 12, 2, 2, PALETTE["ink"])
    c.line(11, 18, 21, 18, PALETTE["ink"])
    c.line(11, 19, 21, 19, PALETTE["ink"])


def icon_losing(c: Canvas) -> None:
    bg(c, PALETTE["ember"], PALETTE["night_0"])
    c.circle(16, 15, 9, PALETTE["white"], fill=True)
    c.fill_rect(12, 12, 2, 2, PALETTE["ink"])
    c.fill_rect(18, 12, 2, 2, PALETTE["ink"])
    c.line(11, 20, 21, 17, PALETTE["ink"])


def icon_spectate(c: Canvas) -> None:
    bg(c, PALETTE["night_2"], PALETTE["night_0"])
    c.circle(11, 16, 5, PALETTE["cyan"], fill=True)
    c.circle(21, 16, 5, PALETTE["cyan"], fill=True)
    c.circle(11, 16, 5, PALETTE["ink"])
    c.circle(21, 16, 5, PALETTE["ink"])
    c.fill_rect(13, 14, 6, 4, PALETTE["ink"])


PACKS: dict[str, list[FrameSpec]] = {
    "werewolf": [
        FrameSpec("campfire", icon_campfire),
        FrameSpec("moon", icon_moon),
        FrameSpec("wolf_head", icon_wolf),
        FrameSpec("villager_badge", icon_villager),
        FrameSpec("seer_eye", icon_seer),
        FrameSpec("doctor_cross", icon_doctor),
        FrameSpec("vote_gavel", icon_vote),
        FrameSpec("elimination_skull", icon_skull),
        FrameSpec("day_banner", icon_day_banner),
        FrameSpec("night_banner", icon_night_banner),
    ],
    "anaconda": [
        FrameSpec("table_felt", icon_table),
        FrameSpec("card_back", icon_card_back),
        FrameSpec("chip_stack", icon_chip_stack),
        FrameSpec("pot_icon", icon_pot),
        FrameSpec("fold_icon", icon_fold),
        FrameSpec("all_in_icon", icon_all_in),
        FrameSpec("reveal_eye", icon_reveal),
        FrameSpec("pass_arrow_left", icon_arrow_left),
        FrameSpec("pass_arrow_right", icon_arrow_right),
        FrameSpec("winner_crown", icon_crown),
    ],
    "blackjack": [
        FrameSpec("icon", icon_blackjack),
        FrameSpec("actions_sheet", icon_bj_actions_sheet),
        FrameSpec("action_hit", icon_bj_hit),
        FrameSpec("action_stand", icon_bj_stand),
        FrameSpec("action_double", icon_bj_double),
    ],
    "holdem": [
        FrameSpec("icon", icon_holdem),
        FrameSpec("markers_sheet", icon_holdem_markers_sheet),
        FrameSpec("marker_dealer", icon_marker_dealer),
        FrameSpec("marker_sb", icon_marker_sb),
        FrameSpec("marker_bb", icon_marker_bb),
    ],
    "shared": [
        FrameSpec("queue_panel", icon_queue_panel),
        FrameSpec("rating_bronze", icon_badge(PALETTE["wood"])),
        FrameSpec("rating_silver", icon_badge(PALETTE["silver"])),
        FrameSpec("rating_gold", icon_badge(PALETTE["sun"])),
        FrameSpec("rating_platinum", icon_badge(PALETTE["cyan"])),
        FrameSpec("rating_diamond", icon_badge(PALETTE["white"])),
        FrameSpec("thinking", icon_thinking),
        FrameSpec("winning", icon_winning),
        FrameSpec("losing", icon_losing),
        FrameSpec("spectate", icon_spectate),
    ],
}


def build_pack(out_root: Path, pack_name: str, tile: int) -> dict[str, object]:
    frames = PACKS[pack_name]
    out_dir = out_root / pack_name
    frame_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    cols = max(1, math.ceil(math.sqrt(len(frames))))
    rows = math.ceil(len(frames) / cols)
    atlas = Canvas(cols * tile, rows * tile, PALETTE["transparent"])
    atlas_frames: list[dict[str, object]] = []

    for i, spec in enumerate(frames):
        frame = Canvas(tile, tile, PALETTE["transparent"])
        spec.draw(frame)
        col = i % cols
        row = i // cols
        x = col * tile
        y = row * tile
        atlas.blit(frame, x, y)
        write_png(frame_dir / f"{spec.name}.png", tile, tile, frame.pixels)
        atlas_frames.append(
            {
                "filename": spec.name,
                "frame": {"x": x, "y": y, "w": tile, "h": tile},
                "anchor": {"x": 0.5, "y": 0.5},
            }
        )

    write_png(out_dir / "atlas.png", atlas.width, atlas.height, atlas.pixels)
    atlas_payload = {
        "frames": atlas_frames,
        "meta": {
            "pack": pack_name,
            "generator": "scripts/build_scenario_assets.py",
            "tile_size": tile,
            "columns": cols,
            "rows": rows,
            "palette": list(PALETTE.keys()),
        },
    }
    (out_dir / "atlas.json").write_text(json.dumps(atlas_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "pack": pack_name,
        "atlas_path": str((out_dir / "atlas.png").as_posix()),
        "frame_count": len(frames),
        "tile_size": tile,
        "columns": cols,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic scenario atlases")
    parser.add_argument(
        "--out",
        default="apps/web/assets/scenarios",
        help="Output directory for generated assets",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=32,
        help="Per-frame tile size (pixels)",
    )
    parser.add_argument(
        "--pack",
        action="append",
        choices=sorted(PACKS.keys()),
        help="Optional specific pack(s) to build; defaults to all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    packs = args.pack or sorted(PACKS.keys())

    manifest: dict[str, object] = {
        "generator": "scripts/build_scenario_assets.py",
        "palette": PALETTE,
        "packs": [],
    }
    for pack in packs:
        result = build_pack(out_root, pack_name=pack, tile=int(args.tile_size))
        cast_packs = manifest["packs"]
        assert isinstance(cast_packs, list)
        cast_packs.append(result)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
