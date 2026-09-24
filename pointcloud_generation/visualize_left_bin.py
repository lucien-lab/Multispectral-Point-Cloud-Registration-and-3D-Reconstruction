#!/usr/bin/env python3
"""Visualize the raw ADC structure in left.bin without loading it all into RAM."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "left.bin"
OUTPUT = ROOT / "left_bin_visualization.png"
SAMPLE_LENGTH = 13_312
FRAME_BINS = 300
SAMPLE_BINS = 512


def block_max_heatmap(raw: np.memmap) -> np.ndarray:
    """Downsample with block maxima so narrow laser pulses remain visible."""
    frame_edges = np.linspace(0, raw.shape[0], FRAME_BINS + 1, dtype=int)
    sample_edges = np.linspace(0, raw.shape[1], SAMPLE_BINS + 1, dtype=int)
    heatmap = np.empty((FRAME_BINS, SAMPLE_BINS), dtype=np.float32)
    for row, (frame_start, frame_end) in enumerate(zip(frame_edges[:-1], frame_edges[1:])):
        block = np.asarray(raw[frame_start:frame_end], dtype=np.int16)
        for column, (sample_start, sample_end) in enumerate(
            zip(sample_edges[:-1], sample_edges[1:])
        ):
            heatmap[row, column] = block[:, sample_start:sample_end].max()
    return heatmap


def main() -> None:
    value_count = INPUT.stat().st_size // np.dtype("<i2").itemsize
    if value_count % SAMPLE_LENGTH:
        raise ValueError(f"{INPUT} is not an integer number of {SAMPLE_LENGTH}-sample frames")
    frame_count = value_count // SAMPLE_LENGTH
    raw = np.memmap(INPUT, dtype="<i2", mode="r", shape=(frame_count, SAMPLE_LENGTH))

    representative = np.linspace(0, frame_count - 1, 5, dtype=int)
    heatmap = block_max_heatmap(raw)
    frame_max = np.asarray(raw.max(axis=1))
    peak_position = np.asarray(raw.argmax(axis=1))

    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=170, constrained_layout=True)

    ax = axes[0, 0]
    for frame in representative:
        ax.plot(raw[frame], linewidth=0.65, alpha=0.8, label=f"帧 {frame + 1}")
    ax.set(title="代表帧原始 ADC 波形", xlabel="采样点", ylabel="ADC 值")
    ax.legend(ncols=2, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    low, high = np.percentile(heatmap, [2, 99.5])
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=(0, SAMPLE_LENGTH, 1, frame_count),
        cmap="turbo",
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    ax.set(title="全部 9,000 帧 ADC 峰值热力图", xlabel="采样点", ylabel="帧编号")
    fig.colorbar(image, ax=ax, label="分块最大 ADC 值")

    ax = axes[1, 0]
    ax.plot(np.arange(1, frame_count + 1), frame_max, color="#d62728", linewidth=0.7)
    ax.set(title="逐帧最大 ADC 值", xlabel="帧编号", ylabel="最大 ADC 值")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.scatter(
        np.arange(1, frame_count + 1),
        peak_position,
        c=frame_max,
        s=3,
        cmap="viridis",
        linewidths=0,
    )
    ax.set(title="逐帧全局峰位置", xlabel="帧编号", ylabel="峰值采样点")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"left.bin 原始数据可视化（{frame_count:,} 帧 × {SAMPLE_LENGTH:,} 点，int16 小端）",
        fontsize=15,
    )
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)

    print(f"输入: {INPUT}")
    print(f"形状: {frame_count} × {SAMPLE_LENGTH}")
    print(f"ADC 范围: {int(raw.min())} ~ {int(raw.max())}")
    print(f"输出: {OUTPUT}")


if __name__ == "__main__":
    main()
