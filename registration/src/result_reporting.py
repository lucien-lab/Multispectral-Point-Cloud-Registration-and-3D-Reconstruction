"""三种配准方法的算法参数汇总。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("geometry", "rgb", "spectral6")
METHOD_LABELS = {
    "geometry": "Geometry point-to-plane ICP",
    "rgb": "RGB Colored ICP",
    "spectral6": "几何 + 六波段 ICP",
}


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_method_parameter_rows(
    config: dict[str, Any],
    wavelengths_nm: np.ndarray,
) -> list[dict[str, object]]:
    """从实际运行配置生成三种方法的可比较参数行。"""
    coarse = config.get("coarse_registration", {"enabled": False})
    method_configs = coarse.get("methods", {})
    common = {
        "fine_voxel_sizes_m": _compact_json(config["voxel_sizes_m"]),
        "fine_max_correspondence_distances_m": _compact_json(
            config["max_correspondence_distances_m"]
        ),
        "fine_max_iterations": _compact_json(config["max_iterations"]),
        "evaluation_max_correspondence_m": float(
            config["evaluation_max_correspondence_m"]
        ),
        "success_rotation_deg": float(config["success_rotation_deg"]),
        "success_translation_m": float(config["success_translation_m"]),
        "overlap_thresholds_m": "[0.1,0.25]",
    }
    wavelengths = _compact_json(
        np.asarray(wavelengths_nm, dtype=int).tolist()
    )
    rows: list[dict[str, object]] = []
    for method in METHODS:
        method_coarse = method_configs.get(method, {})
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "coarse_modality": method,
            "coarse_enabled": bool(coarse.get("enabled", False)),
            "coarse_config_id": method_coarse.get(
                "config_id",
                f"default_{method}",
            ),
            "coarse_voxel_size_m": method_coarse.get("voxel_size_m", ""),
            "coarse_max_correspondence_m": method_coarse.get(
                "max_correspondence_m",
                "",
            ),
            "coarse_normal_radius_multiplier": method_coarse.get(
                "normal_radius_multiplier",
                "",
            ),
            "coarse_attribute_feature_weight": method_coarse.get(
                "attribute_feature_weight",
                "",
            ),
            "coarse_lambda_geo": method_coarse.get("lambda_geo", ""),
            "coarse_lambda_attribute": method_coarse.get(
                "lambda_attribute",
                "",
            ),
            "coarse_lambda_overlap": method_coarse.get("lambda_overlap", ""),
            "coarse_candidate_count": coarse.get("candidate_count", ""),
            "coarse_ransac_n": coarse.get("ransac_n", ""),
            "coarse_max_iterations": coarse.get("max_iterations", ""),
            "coarse_confidence": coarse.get("confidence", ""),
            "coarse_max_rotation_deg": coarse.get("max_rotation_deg", ""),
            "coarse_max_translation_m": coarse.get("max_translation_m", ""),
            "fine_estimation": (
                "point_to_plane" if method == "geometry" else method
            ),
            "rgb_geometric_weight": (
                float(config["lambda_geometric_rgb"])
                if method == "rgb"
                else ""
            ),
            "rgb_weight": (
                1.0 - float(config["lambda_geometric_rgb"])
                if method == "rgb"
                else ""
            ),
            "spectral_geometric_weight": (
                float(config["lambda_geo_spectral"])
                if method == "spectral6"
                else ""
            ),
            "spectral_weight": (
                float(config["lambda_spec_spectral"])
                if method == "spectral6"
                else ""
            ),
            "wavelengths_nm": wavelengths if method == "spectral6" else "",
            **common,
        }
        rows.append(row)
    return rows


def write_method_parameter_summary(
    results_dir: Path,
    config: dict[str, Any],
    wavelengths_nm: np.ndarray,
) -> list[dict[str, object]]:
    """写出三种算法参数的CSV与Markdown汇总。"""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = build_method_parameter_rows(config, wavelengths_nm)
    fields = list(rows[0])
    with (results_dir / "method_parameters_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 三种配准方法参数汇总",
        "",
        "| 方法 | 粗配准模态/config | 粗体素/对应距离 m | 粗权重 geo/attr/overlap | 细体素 m | 细对应距离 m | 细迭代 | 细配准属性权重 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["method"] == "rgb":
            fine_weight = (
                f"geo={float(row['rgb_geometric_weight']):.3f}, "
                f"RGB={float(row['rgb_weight']):.3f}"
            )
        elif row["method"] == "spectral6":
            fine_weight = (
                f"geo={float(row['spectral_geometric_weight']):.3f}, "
                f"六波段={float(row['spectral_weight']):.3f}; "
                f"{row['wavelengths_nm']} nm"
            )
        else:
            fine_weight = "仅几何 point-to-plane"
        lines.append(
            f"| {row['method_label']} | {row['coarse_modality']} / "
            f"`{row['coarse_config_id']}` | "
            f"{row['coarse_voxel_size_m']} / "
            f"{row['coarse_max_correspondence_m']} | "
            f"{row['coarse_lambda_geo']} / "
            f"{row['coarse_lambda_attribute']} / "
            f"{row['coarse_lambda_overlap']} | "
            f"{row['fine_voxel_sizes_m']} | "
            f"{row['fine_max_correspondence_distances_m']} | "
            f"{row['fine_max_iterations']} | {fine_weight} |"
        )
    lines.extend(
        [
            "",
            "统一评价参数：",
            "",
            f"- 最大评价对应距离：{config['evaluation_max_correspondence_m']} m",
            f"- 成功阈值：旋转不超过 {config['success_rotation_deg']} deg，"
            f"平移不超过 {config['success_translation_m']} m",
            "- 双向重叠率阈值：0.10 m、0.25 m",
        ]
    )
    (results_dir / "method_parameters_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return rows
