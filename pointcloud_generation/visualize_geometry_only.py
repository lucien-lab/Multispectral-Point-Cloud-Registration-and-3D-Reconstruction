#!/usr/bin/env python3
"""Generate monochrome SVG previews and geometry-only PLY files without dependencies."""

from __future__ import annotations

import csv
import binascii
import math
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pointcloud_output"
OUTPUT_DIR = ROOT / "geometry_only"
SIDES = ("left", "right")
WIDTH = 900
HEIGHT = 700


def load_valid_points(side: str) -> list[tuple[float, float, float]]:
    path = INPUT_DIR / f"{side}_pointcloud_all.csv"
    points: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["valid"])) != 1:
                continue
            points.append((float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])))
    return points


def project(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    azimuth = math.radians(-50.0)
    elevation = math.radians(20.0)
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    result = []
    for x, y, z in points:
        x -= cx; y -= cy; z -= cz
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        vertical = ce * z - se * yr
        depth = se * z + ce * yr
        result.append((xr, vertical, depth))
    return result


def panel_svg(points: list[tuple[float, float, float]], side: str, x_offset: int = 0) -> str:
    projected = project(points)
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    margin_x, margin_y = 55, 85
    scale = min(
        (WIDTH - 2 * margin_x) / max(umax - umin, 1.0),
        (HEIGHT - 2 * margin_y) / max(vmax - vmin, 1.0),
    )
    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)
    ordered = sorted(projected, key=lambda item: item[2])
    dots = []
    for u, v, _ in ordered:
        px = x_offset + WIDTH / 2 + (u - uc) * scale
        py = HEIGHT / 2 - (v - vc) * scale
        dots.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.55"/>')

    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    label = (
        f"{side}.bin：{len(points):,} 个有效点　"
        f"X {mins[0]:.0f}–{maxs[0]:.0f} mm　"
        f"Y {mins[1]:.0f}–{maxs[1]:.0f} mm　"
        f"Z {mins[2]:.0f}–{maxs[2]:.0f} mm"
    )
    return "\n".join(
        [
            f'<rect x="{x_offset}" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#fafafa"/>',
            f'<rect x="{x_offset + 24}" y="52" width="{WIDTH - 48}" height="{HEIGHT - 104}" fill="white" stroke="#c8c8c8"/>',
            f'<text x="{x_offset + WIDTH / 2}" y="32" text-anchor="middle" font-size="22" font-family="sans-serif">{side}.bin 无颜色几何点云</text>',
            f'<g fill="#30343b" fill-opacity="0.78">',
            *dots,
            "</g>",
            f'<text x="{x_offset + WIDTH / 2}" y="{HEIGHT - 20}" text-anchor="middle" font-size="13" fill="#444" font-family="sans-serif">{label}</text>',
        ]
    )


def projected_pixels(
    points: list[tuple[float, float, float]], x_offset: int = 0
) -> list[tuple[int, int, float]]:
    projected = project(points)
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    margin_x, margin_y = 55, 55
    scale = min(
        (WIDTH - 2 * margin_x) / max(umax - umin, 1.0),
        (HEIGHT - 2 * margin_y) / max(vmax - vmin, 1.0),
    )
    uc = 0.5 * (umin + umax)
    vc = 0.5 * (vmin + vmax)
    return [
        (
            round(x_offset + WIDTH / 2 + (u - uc) * scale),
            round(HEIGHT / 2 - (v - vc) * scale),
            depth,
        )
        for u, v, depth in projected
    ]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data))


def write_png(
    path: Path,
    panels: list[tuple[list[tuple[float, float, float]], int]],
    width: int,
) -> None:
    pixels = bytearray([250, 250, 250] * width * HEIGHT)

    def set_pixel(x: int, y: int, gray: int) -> None:
        if 0 <= x < width and 0 <= y < HEIGHT:
            index = 3 * (y * width + x)
            pixels[index:index + 3] = bytes((gray, gray, gray))

    for points, x_offset in panels:
        for x in range(x_offset + 24, x_offset + WIDTH - 24):
            set_pixel(x, 24, 185); set_pixel(x, HEIGHT - 25, 185)
        for y in range(24, HEIGHT - 24):
            set_pixel(x_offset + 24, y, 185); set_pixel(x_offset + WIDTH - 25, y, 185)
        for x, y, _ in sorted(projected_pixels(points, x_offset), key=lambda item: item[2]):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx * dx + dy * dy <= 2:
                        set_pixel(x + dx, y + dy, 45)

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3:(row + 1) * width * 3])
        for row in range(HEIGHT)
    )
    header = struct.pack(">IIBBBBB", width, HEIGHT, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
    data += png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + png_chunk(b"IEND", b"")
    path.write_bytes(data)


def write_svg(path: Path, panels: list[str], width: int) -> None:
    content = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{HEIGHT}" viewBox="0 0 {width} {HEIGHT}">',
            *panels,
            "</svg>",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment Geometry only; no color properties\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {side: load_valid_points(side) for side in SIDES}
    for side, points in data.items():
        write_svg(OUTPUT_DIR / f"{side}_geometry_preview.svg", [panel_svg(points, side)], WIDTH)
        write_png(OUTPUT_DIR / f"{side}_geometry_preview.png", [(points, 0)], WIDTH)
        write_ply(OUTPUT_DIR / f"{side}_geometry_only.ply", points)
    write_svg(
        OUTPUT_DIR / "left_right_geometry_comparison.svg",
        [panel_svg(data["left"], "left", 0), panel_svg(data["right"], "right", WIDTH)],
        WIDTH * 2,
    )
    write_png(
        OUTPUT_DIR / "left_right_geometry_comparison.png",
        [(data["left"], 0), (data["right"], WIDTH)],
        WIDTH * 2,
    )
    for side in SIDES:
        print(f"{side}: {len(data[side])} valid geometry points")
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
