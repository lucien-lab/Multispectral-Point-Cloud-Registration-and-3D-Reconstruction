"""三种粗配准方法的平衡搜索空间与确定性选参规则。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np


VOXEL_DISTANCE_PAIRS = (
    (0.10, 0.15),
    (0.12, 0.18),
    (0.16, 0.24),
)
TUNING_TRIALS = frozenset(range(1, 6))
HELDOUT_TRIALS = frozenset(range(6, 11))
FAILURE_PENALTY = 1_000_000.0


def _base_config(common: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in common.items()
        if key != "methods"
    }


def generate_search_configs(
    method: str,
    common: dict[str, object],
) -> list[dict[str, object]]:
    """按批准的18/27/27组搜索空间生成完整粗配准配置。"""
    if method not in {"geometry", "rgb", "spectral6"}:
        raise ValueError(f"未知调参方法: {method}")
    configs: list[dict[str, object]] = []
    for voxel_size, max_correspondence in VOXEL_DISTANCE_PAIRS:
        if method == "geometry":
            for normal_multiplier in (2.0, 3.0):
                for overlap_weight in (0.10, 0.20, 0.30):
                    config = _base_config(common)
                    config.update(
                        {
                            "modality": method,
                            "config_id": (
                                f"geometry_v{voxel_size:.2f}"
                                f"_n{normal_multiplier:.1f}"
                                f"_o{overlap_weight:.2f}"
                            ),
                            "voxel_size_m": voxel_size,
                            "max_correspondence_m": max_correspondence,
                            "normal_radius_multiplier": normal_multiplier,
                            "lambda_geo": 1.0 - overlap_weight,
                            "lambda_attribute": 0.0,
                            "lambda_overlap": overlap_weight,
                        }
                    )
                    configs.append(config)
            continue

        feature_weights = (
            (0.5, 1.0, 2.0)
            if method == "rgb"
            else (1.0, 2.0, 3.0)
        )
        data_fractions = (
            (0.20, 0.40, 0.60)
            if method == "rgb"
            else (0.30, 0.50, 0.70)
        )
        for feature_weight in feature_weights:
            for data_fraction in data_fractions:
                attribute_weight = 0.80 * data_fraction
                geometry_weight = 0.80 - attribute_weight
                config = _base_config(common)
                config.update(
                    {
                        "modality": method,
                        "config_id": (
                            f"{method}_v{voxel_size:.2f}"
                            f"_f{feature_weight:.1f}"
                            f"_d{data_fraction:.2f}"
                        ),
                        "voxel_size_m": voxel_size,
                        "max_correspondence_m": max_correspondence,
                        "normal_radius_multiplier": 2.0,
                        "attribute_feature_weight": feature_weight,
                        "lambda_geo": geometry_weight,
                        "lambda_attribute": attribute_weight,
                        "lambda_overlap": 0.20,
                    }
                )
                configs.append(config)
    return configs


def summarize_search_results(
    rows: Iterable[dict[str, object]],
    *,
    tuning_trials: frozenset[int] = TUNING_TRIALS,
    success_rotation_deg: float = 2.0,
    success_translation_m: float = 0.10,
) -> list[dict[str, object]]:
    """只使用调参trial聚合每套配置，错误运行按大惩罚计入。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        trial = int(row["trial"])
        if trial in tuning_trials:
            grouped[str(row["config_id"])].append(row)

    summaries: list[dict[str, object]] = []
    for config_id, config_rows in sorted(grouped.items()):
        success_count = 0
        normalized_errors: list[float] = []
        translations: list[float] = []
        runtimes: list[float] = []
        for row in config_rows:
            runtime = float(row.get("runtime_s", 0.0))
            runtimes.append(runtime if np.isfinite(runtime) else FAILURE_PENALTY)
            if row.get("status") != "ok":
                normalized_errors.append(FAILURE_PENALTY)
                translations.append(FAILURE_PENALTY)
                continue
            rotation = float(row["rotation_error_deg"])
            translation = float(row["translation_error_m"])
            normalized_errors.append(
                rotation / success_rotation_deg
                + translation / success_translation_m
            )
            translations.append(translation)
            if (
                rotation <= success_rotation_deg
                and translation <= success_translation_m
            ):
                success_count += 1
        summaries.append(
            {
                "config_id": config_id,
                "trial_count": len(config_rows),
                "success_count": success_count,
                "normalized_error_mean": float(np.mean(normalized_errors)),
                "translation_error_m_median": float(median(translations)),
                "runtime_s_mean": float(np.mean(runtimes)),
            }
        )
    return summaries


def select_best_config(
    summaries: Iterable[dict[str, object]],
) -> dict[str, object]:
    """按批准的五级稳定排序选择最优配置摘要。"""
    candidates = list(summaries)
    if not candidates:
        raise ValueError("没有可用于选参的调参集结果")
    return min(
        candidates,
        key=lambda item: (
            -int(item["success_count"]),
            float(item["normalized_error_mean"]),
            float(item["translation_error_m_median"]),
            float(item["runtime_s_mean"]),
            str(item["config_id"]),
        ),
    )


def write_tuning_outputs(
    output_dir: Path,
    method: str,
    configs: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """保存逐trial搜索结果、获胜完整配置及可读摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with (output_dir / "search_results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in fieldnames}
            for row in rows
        )

    summaries = summarize_search_results(rows)
    selected = select_best_config(summaries)
    configs_by_id = {
        str(config["config_id"]): config
        for config in configs
    }
    selected_config = configs_by_id[str(selected["config_id"])]
    payload = {
        "method": method,
        "tuning_trials": sorted(TUNING_TRIALS),
        "heldout_trials": sorted(HELDOUT_TRIALS),
        "selection_rule": (
            "success_count desc, normalized_error_mean asc, "
            "translation_error_m_median asc, runtime_s_mean asc, "
            "config_id asc"
        ),
        "selection": selected,
        "config": selected_config,
    }
    (output_dir / "selected_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ranked = sorted(
        summaries,
        key=lambda item: (
            -int(item["success_count"]),
            float(item["normalized_error_mean"]),
            float(item["translation_error_m_median"]),
            float(item["runtime_s_mean"]),
            str(item["config_id"]),
        ),
    )
    lines = [
        f"# {method} coarse tuning",
        "",
        "- 选参trial：1–5",
        "- 留出trial：6–10，不参与本表选参",
        f"- 最优配置：`{selected['config_id']}`",
        "",
        "| rank | config_id | success | normalized error | median translation / m | mean runtime / s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, summary in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {summary['config_id']} | "
            f"{summary['success_count']}/{summary['trial_count']} | "
            f"{float(summary['normalized_error_mean']):.6g} | "
            f"{float(summary['translation_error_m_median']):.6g} | "
            f"{float(summary['runtime_s_mean']):.6g} |"
        )
    (output_dir / "tuning_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return selected
