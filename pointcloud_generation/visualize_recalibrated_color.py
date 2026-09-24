#!/usr/bin/env python3
"""Create color previews using the updated white-board calibration, without dependencies."""

from __future__ import annotations

import binascii
import csv
import math
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pointcloud_output"
OUTPUT_DIR = ROOT / "recalibrated_color"
WIDTH = 900
HEIGHT = 700
SIDES = ("left", "right")

WAVELENGTHS = (495, 696, 600, 803, 545, 642)
OLD_WHITE = (124.1064096774194, 120.2428838709678, 115.0127903225807,
             166.3828838709677, 115.8136612903226, 226.6485516129032)
NEW_WHITE = (531.2975, 396.4109, 425.0606, 361.2250, 572.4702, 477.3115)
CIE_XYZ = (
    (0.014700000000, 0.2586000000000, 0.353300000000),
    (0.014790640000, 0.0053430590000, 0.000000000000),
    (1.062200000000, 0.6310000000000, 0.000800000000),
    (0.000008308358, 0.0000030003000, 0.000000000000),
    (0.359700000000, 0.9803000000000, 0.013400000000),
    (0.412098000000, 0.1596464000000, 0.000016200000),
)
VISIBLE = (0, 1, 2, 4, 5)
XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def gamma_encode(value: float) -> float:
    value = max(value, 0.0)
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


def spectral_rgb(reflectance: list[float]) -> tuple[int, int, int]:
    normalization = sum(CIE_XYZ[index][1] for index in VISIBLE)
    xyz = [
        sum(reflectance[index] * CIE_XYZ[index][component] for index in VISIBLE)
        / normalization
        for component in range(3)
    ]
    linear = [sum(matrix_row[i] * xyz[i] for i in range(3)) for matrix_row in XYZ_TO_SRGB]
    encoded = [min(max(gamma_encode(value), 0.0), 1.0) for value in linear]
    return tuple(round(value * 255) for value in encoded)


def load_side(side: str):
    points = []
    with (INPUT_DIR / f"{side}_pointcloud_all.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["valid"])) != 1:
                continue
            old_rfl = [float(row[f"rfl_{wavelength}_nm"]) for wavelength in WAVELENGTHS]
            new_rfl = [value * old / new for value, old, new in zip(old_rfl, OLD_WHITE, NEW_WHITE)]
            points.append(
                (
                    float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]),
                    spectral_rgb(new_rfl),
                )
            )
    return points


def projected_pixels(points, x_offset=0):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    azimuth = math.radians(-50.0)
    elevation = math.radians(20.0)
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    projected = []
    for x, y, z, color in points:
        x -= cx; y -= cy; z -= cz
        xr = ca * x - sa * y
        yr = sa * x + ca * y
        vertical = ce * z - se * yr
        depth = se * z + ce * yr
        projected.append((xr, vertical, depth, color))
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    margin = 55
    scale = min(
        (WIDTH - 2 * margin) / max(max(us) - min(us), 1.0),
        (HEIGHT - 2 * margin) / max(max(vs) - min(vs), 1.0),
    )
    uc = 0.5 * (min(us) + max(us))
    vc = 0.5 * (min(vs) + max(vs))
    return [
        (round(x_offset + WIDTH / 2 + (u - uc) * scale),
         round(HEIGHT / 2 - (v - vc) * scale), depth, color)
        for u, v, depth, color in projected
    ]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data))


def write_png(path: Path, panels, width: int) -> None:
    pixels = bytearray([250, 250, 250] * width * HEIGHT)

    def set_pixel(x: int, y: int, color) -> None:
        if 0 <= x < width and 0 <= y < HEIGHT:
            index = 3 * (y * width + x)
            pixels[index:index + 3] = bytes(color)

    for points, x_offset in panels:
        for x in range(x_offset + 24, x_offset + WIDTH - 24):
            set_pixel(x, 24, (185, 185, 185)); set_pixel(x, HEIGHT - 25, (185, 185, 185))
        for y in range(24, HEIGHT - 24):
            set_pixel(x_offset + 24, y, (185, 185, 185)); set_pixel(x_offset + WIDTH - 25, y, (185, 185, 185))
        for x, y, _, color in sorted(projected_pixels(points, x_offset), key=lambda item: item[2]):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx * dx + dy * dy <= 2:
                        set_pixel(x + dx, y + dy, color)

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3:(row + 1) * width * 3])
        for row in range(HEIGHT)
    )
    header = struct.pack(">IIBBBBB", width, HEIGHT, 8, 2, 0, 0, 0)
    content = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
    content += png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + png_chunk(b"IEND", b"")
    path.write_bytes(content)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {side: load_side(side) for side in SIDES}
    for side, points in data.items():
        write_png(OUTPUT_DIR / f"{side}_updated_color.png", [(points, 0)], WIDTH)
        averages = tuple(sum(p[3][i] for p in points) / len(points) for i in range(3))
        print(f"{side}: {len(points)} points, mean RGB8={averages}")
    write_png(
        OUTPUT_DIR / "left_right_updated_color.png",
        [(data["left"], 0), (data["right"], WIDTH)], WIDTH * 2,
    )
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
