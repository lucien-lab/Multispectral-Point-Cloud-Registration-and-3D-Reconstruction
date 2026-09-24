#!/usr/bin/env python3
"""运行左右几何、RGB和真实六波段点云配准实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import scipy

from src.coarse_registration import (
    CoarseCandidate,
    CoarseRegistrationResult,
    register_geometry_coarse,
    register_rgb_coarse,
    register_spectral_coarse,
)
from src.data_prepare import load_prepared, prepare_snapshot
from src.evaluation import evaluate_registration, full_metrics
from src.icp_geometry import ICPResult, register_geometry_icp
from src.icp_rgb import register_rgb_icp
from src.icp_spectral import register_spectral_icp
from src.result_reporting import write_method_parameter_summary
from src.transform import (
    Perturbation,
    apply_transform,
    generate_perturbations,
    invert_transform,
)
from src.visualization import (
    save_before_coarse_after,
    save_coarse_and_fine_error_boxplots,
    save_error_boxplots,
    save_method_comparison,
    write_coarse_registration_clouds,
    write_registration_clouds,
)


METHOD_DIRS = {
    "geometry": "icp_geometry",
    "rgb": "icp_rgb",
    "spectral6": "icp_spectral6",
}
METHOD_LABELS = {
    "geometry": "Point-to-plane geometry ICP",
    "rgb": "Open3D RGB Colored ICP",
    "spectral6": "Geometry + measured six-band spectral ICP",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_tuned_coarse_configs(
    config: dict[str, Any],
    root: Path,
) -> None:
    """将已生成的分方法获胜配置覆盖到运行时配置。"""
    coarse = config.setdefault("coarse_registration", {"enabled": False})
    methods = coarse.setdefault("methods", {})
    for method in METHOD_DIRS:
        selected_path = (
            root
            / "results"
            / "coarse_tuning"
            / method
            / "selected_config.json"
        )
        if not selected_path.is_file():
            continue
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        if payload.get("method") != method:
            raise ValueError(
                f"{selected_path} 的method与目录不一致"
            )
        selected = payload.get("config")
        if not isinstance(selected, dict):
            raise ValueError(f"{selected_path} 缺少config对象")
        methods[method] = dict(selected)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _configure_logging(root: Path) -> tuple[logging.Logger, Path]:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"experiment_{timestamp}.log"
    logger = logging.getLogger(f"multimodal_registration_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, log_path


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _save_transform(
    path: Path,
    matrix: np.ndarray,
    **metadata: Any,
) -> None:
    _save_json(path, {"matrix": np.asarray(matrix).tolist(), **metadata})


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("status\nno_history\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _candidate_payload(candidate: CoarseCandidate) -> dict[str, object]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "candidate_id": candidate.candidate_id,
        "seed": candidate.seed,
        "transformation": candidate.transformation.tolist(),
        "rotation_magnitude_deg": finite_or_none(candidate.rotation_magnitude_deg),
        "translation_magnitude_m": finite_or_none(
            candidate.translation_magnitude_m
        ),
        "coverage": finite_or_none(candidate.coverage),
        "geometry_error": finite_or_none(candidate.geometry_error),
        "modality": candidate.modality,
        "attribute_error": finite_or_none(candidate.attribute_error),
        "rgb_error": (
            None
            if candidate.rgb_error is None
            else finite_or_none(candidate.rgb_error)
        ),
        "spectral_error": (
            None
            if candidate.spectral_error is None
            else finite_or_none(candidate.spectral_error)
        ),
        "overlap_error": finite_or_none(candidate.overlap_error),
        "joint_score": finite_or_none(candidate.joint_score),
        "accepted": candidate.accepted,
        "rejection_reason": candidate.rejection_reason,
    }


def _write_coarse_candidates(
    output_dir: Path,
    candidates: list[CoarseCandidate],
) -> None:
    payload = [_candidate_payload(candidate) for candidate in candidates]
    _save_json(output_dir / "coarse_candidates.json", {"candidates": payload})
    fields = [
        "candidate_id",
        "seed",
        "rotation_magnitude_deg",
        "translation_magnitude_m",
        "coverage",
        "geometry_error",
        "modality",
        "attribute_error",
        "rgb_error",
        "spectral_error",
        "overlap_error",
        "joint_score",
        "accepted",
        "rejection_reason",
        "transformation_json",
    ]
    with (output_dir / "coarse_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in payload:
            row = {
                key: item.get(key)
                for key in fields
                if key != "transformation_json"
            }
            row["transformation_json"] = json.dumps(
                item["transformation"],
                separators=(",", ":"),
            )
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_inputs(
    config: dict[str, Any],
    root: Path,
    logger: logging.Logger,
) -> tuple[dict[str, object], dict[str, object]]:
    data = config["data"]
    left_input = _resolve(root, data["left_input"])
    right_input = _resolve(root, data["right_input"])
    left_output = _resolve(root, data["left_prepared"])
    right_output = _resolve(root, data["right_prepared"])
    logger.info("准备左点云: %s", left_input)
    left_summary = prepare_snapshot(left_input, left_output, "left")
    logger.info("准备右点云: %s", right_input)
    right_summary = prepare_snapshot(right_input, right_output, "right")
    return left_summary, right_summary


def _run_method(
    method: str,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_spectral: np.ndarray,
    target_spectral: np.ndarray,
    config: dict[str, Any],
    init: np.ndarray | None = None,
) -> ICPResult:
    validation_config = config["registration_validation"]
    common = (
        list(config["voxel_sizes_m"]),
        list(config["max_correspondence_distances_m"]),
        list(config["max_iterations"]),
    )
    if method == "geometry":
        return register_geometry_icp(
            source_xyz,
            target_xyz,
            *common,
            init=init,
            validation_config=validation_config,
        )
    if method == "rgb":
        return register_rgb_icp(
            source_xyz,
            target_xyz,
            source_rgb,
            target_rgb,
            *common,
            lambda_geometric=float(config["lambda_geometric_rgb"]),
            init=init,
            validation_config=validation_config,
        )
    if method == "spectral6":
        return register_spectral_icp(
            source_xyz,
            target_xyz,
            source_spectral,
            target_spectral,
            *common,
            lambda_geo=float(config["lambda_geo_spectral"]),
            lambda_spec=float(config["lambda_spec_spectral"]),
            init=init,
            validation_config=validation_config,
        )
    raise ValueError(f"未知方法: {method}")


def _coarse_config_for_method(
    config: dict[str, Any],
    method: str,
) -> dict[str, object]:
    coarse = dict(config.get("coarse_registration", {"enabled": False}))
    methods = coarse.pop("methods", {})
    if method not in METHOD_DIRS:
        raise ValueError(f"未知粗配准方法: {method}")
    method_config = dict(methods.get(method, {}))
    coarse.update(method_config)
    coarse["modality"] = method
    coarse.setdefault("config_id", f"default_{method}")
    if method == "spectral6":
        if "attribute_feature_weight" not in coarse:
            coarse["attribute_feature_weight"] = coarse.get(
                "spectral_feature_weight",
                1.0,
            )
        if "lambda_attribute" not in coarse:
            coarse["lambda_attribute"] = coarse.get("lambda_spec", 0.0)
    return coarse


def _run_coarse_registration(
    method: str,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    config: dict[str, Any],
    trial_number: int,
) -> CoarseRegistrationResult:
    coarse = _coarse_config_for_method(config, method)
    seed = int(config["seed"]) + trial_number
    if not bool(coarse["enabled"]):
        evaluated = evaluate_registration(
            source_xyz,
            target_xyz,
            np.eye(4),
            float(config["evaluation_max_correspondence_m"]),
        )
        identity_score = 1.0 - float(evaluated["fitness"])
        identity = CoarseCandidate(
            candidate_id="identity_baseline",
            seed=None,
            transformation=np.eye(4),
            rotation_magnitude_deg=0.0,
            translation_magnitude_m=0.0,
            coverage=float(evaluated["fitness"]),
            geometry_error=0.0,
            spectral_error=0.0 if method == "spectral6" else None,
            overlap_error=identity_score,
            joint_score=identity_score,
            accepted=True,
            rejection_reason="",
            modality=method,
            attribute_error=0.0,
            rgb_error=0.0 if method == "rgb" else None,
        )
        return CoarseRegistrationResult(
            transformation=np.eye(4),
            fitness=float(evaluated["fitness"]),
            inlier_rmse=float(evaluated["inlier_rmse_m"]),
            source_downsampled_count=len(source_xyz),
            target_downsampled_count=len(target_xyz),
            parameters={
                "enabled": False,
                "modality": method,
                "config_id": coarse["config_id"],
                "seed": seed,
                "identity_baseline_score": identity_score,
                "selected_score": identity_score,
                "valid_candidate_count": 0,
            },
            candidates=[identity],
            selected_candidate_id="identity_baseline",
            fallback_identity=True,
            selection_reason="coarse_registration_disabled",
        )

    if method == "geometry":
        result = register_geometry_coarse(
            source_xyz,
            target_xyz,
            config=coarse,
            seed=seed,
        )
    elif method == "rgb":
        result = register_rgb_coarse(
            source_xyz,
            target_xyz,
            source_rgb,
            target_rgb,
            config=coarse,
            seed=seed,
        )
    else:
        result = register_spectral_coarse(
            source_xyz,
            target_xyz,
            source_spectral6,
            target_spectral6,
            config=coarse,
            seed=seed,
        )
    result.parameters["enabled"] = True
    return result


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method in METHOD_DIRS:
        method_rows = [row for row in rows if row["method"] == method]
        valid = [row for row in method_rows if row["status"] == "ok"]
        if not method_rows:
            continue
        summary: dict[str, object] = {
            "method": method,
            "trial_count": len(method_rows),
            "completed_count": len(valid),
            "success_rate": float(
                sum(bool(row.get("success", False)) for row in method_rows)
                / len(method_rows)
            ),
        }
        for key in (
            "rotation_error_deg",
            "translation_error_m",
            "overlap_100mm_pct",
            "overlap_250mm_pct",
            "fitness",
            "inlier_rmse_m",
            "runtime_s",
        ):
            values = np.array(
                [
                    float(row[key])
                    for row in valid
                    if row.get(key) is not None
                ],
                dtype=float,
            )
            summary[f"{key}_mean"] = float(values.mean()) if values.size else None
            summary[f"{key}_std"] = float(values.std()) if values.size else None
        summaries.append(summary)
    return summaries


def write_result_tables(
    root: Path,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    row_fields = [
        "method",
        "trial",
        "status",
        "rotation_error_deg",
        "translation_error_m",
        "overlap_100mm_pct",
        "overlap_250mm_pct",
        "fitness",
        "inlier_rmse_m",
        "success",
        "runtime_s",
        "coarse_method",
        "coarse_config_id",
        "coarse_rotation_error_deg",
        "coarse_translation_error_m",
        "coarse_fitness",
        "coarse_inlier_rmse_m",
        "coarse_runtime_s",
        "error_message",
    ]
    with (results_dir / "results_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in row_fields} for row in rows)

    summaries = _summary_rows(rows)
    _write_method_summary_files(results_dir, "method_summary", summaries)
    _write_method_summary_files(
        results_dir,
        "all_trials_method_summary",
        summaries,
    )
    heldout = _summary_rows(
        [row for row in rows if 6 <= int(row["trial"]) <= 10]
    )
    _write_method_summary_files(
        results_dir,
        "heldout_method_summary",
        heldout,
    )
    return summaries


def _write_method_summary_files(
    results_dir: Path,
    basename: str,
    summaries: list[dict[str, object]],
) -> None:
    fields = [
        "method",
        "trial_count",
        "completed_count",
        "success_rate",
    ]
    for key in (
        "rotation_error_deg",
        "translation_error_m",
        "overlap_100mm_pct",
        "overlap_250mm_pct",
        "fitness",
        "inlier_rmse_m",
        "runtime_s",
    ):
        fields.extend((f"{key}_mean", f"{key}_std"))
    with (results_dir / f"{basename}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: item.get(key) for key in fields}
            for item in summaries
        )
    lines = [
        f"# {basename.replace('_', ' ').title()}",
        "",
        "| method | rotation error / deg | translation error / m | overlap 100 mm / % | overlap 250 mm / % | fitness | RMSE / m | success rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        def mean_std(key: str) -> str:
            mean = item[f"{key}_mean"]
            std = item[f"{key}_std"]
            return "N/A" if mean is None else f"{float(mean):.6g} ± {float(std):.3g}"

        lines.append(
            f"| {item['method']} | {mean_std('rotation_error_deg')} | "
            f"{mean_std('translation_error_m')} | "
            f"{mean_std('overlap_100mm_pct')} | "
            f"{mean_std('overlap_250mm_pct')} | "
            f"{mean_std('fitness')} | "
            f"{mean_std('inlier_rmse_m')} | {100 * float(item['success_rate']):.1f}% |"
        )
    if not summaries:
        lines.extend(["", "_No matching trials were run._"])
    (results_dir / f"{basename}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_experiment_log(
    root: Path,
    config: dict[str, Any],
    data_summaries: tuple[dict[str, object], dict[str, object]],
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    log_path: Path,
    perturbation_rows: list[tuple[int, Perturbation]],
) -> None:
    checksums: list[tuple[str, str]] = []
    for value in config["data"].values():
        path = _resolve(root, value)
        if path.is_file():
            checksums.append((str(path.relative_to(root)), _sha256(path)))
    for path in sorted((root / "data" / "raw").glob("*")) if (root / "data" / "raw").exists() else []:
        if path.is_file():
            checksums.append((str(path.relative_to(root)), _sha256(path)))

    lines = [
        "# 多模态点云配准实验日志",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- Python：{platform.python_version()}",
        f"- 平台：{platform.platform()}",
        f"- Conda 环境：{Path(sys.prefix).name}",
        f"- Open3D：{o3d.__version__}",
        f"- NumPy：{np.__version__}",
        f"- SciPy：{scipy.__version__}",
        f"- 运行日志：`{log_path.relative_to(root)}`",
        "",
        "## 数据来源",
        "",
        "- source：left.bin 对应的 6.3–10 m 左侧点云",
        "- target：right.bin 对应的 6.3–10 m 右侧点云",
        "- 坐标单位：m",
        "- 六波段顺序：495、696、600、803、545、642 nm",
        f"- 左侧点数：{data_summaries[0]['point_count']}",
        f"- 右侧点数：{data_summaries[1]['point_count']}",
        "",
        "## 粗配准选参与独立评价",
        "",
        "- trial 1–5：仅用于三种粗配准方法的独立参数选择。",
        "- trial 6–10：独立留出评价集，不参与配置排序。",
        "- T_gt 只用于离线选参与最终评价，不参与RANSAC候选生成、候选评分或细配准逐层回退。",
        "- 同一trial的扰动source与target完全相同；geometry、RGB、spectral6分别运行自身粗配准。",
        "",
        "### 运行时粗配准配置",
        "",
    ]
    for method_name in METHOD_DIRS:
        method_config = _coarse_config_for_method(config, method_name)
        lines.append(
            f"- `{method_name}`：`{method_config['config_id']}`"
        )
    lines.extend(
        [
        "",
        "### SHA-256",
        "",
        ]
    )
    lines.extend(f"- `{path}`：`{checksum}`" for path, checksum in checksums)
    lines.extend(
        [
            "",
            "## SE(3) 与 ICP 参数",
            "",
            "```json",
            json.dumps(config, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 每次随机扰动",
            "",
            "| trial | rx / deg | ry / deg | rz / deg | tx / m | ty / m | tz / m |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trial_number, perturbation in perturbation_rows:
        rx, ry, rz = perturbation.euler_deg
        tx, ty, tz = perturbation.translation_m
        lines.append(
            f"| {trial_number} | {rx:.9f} | {ry:.9f} | {rz:.9f} | "
            f"{tx:.9f} | {ty:.9f} | {tz:.9f} |"
        )
    lines.extend(
        [
            "",
            "## 实验结果",
            "",
            (root / "results" / "method_summary.md").read_text(encoding="utf-8"),
            "",
            "## 逐次结果",
            "",
            "| method | trial | status | rotation / deg | translation / m | fitness | RMSE / m | success |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        def value(name: str) -> str:
            item = row.get(name)
            return "N/A" if item is None else f"{float(item):.9g}"

        lines.append(
            f"| {row['method']} | {row['trial']} | {row['status']} | "
            f"{value('rotation_error_deg')} | {value('translation_error_m')} | "
            f"{value('fitness')} | {value('inlier_rmse_m')} | {row['success']} |"
        )
    lines.extend(
        [
            "",
            f"- 共记录 {len(rows)} 个方法-trial 结果。",
            f"- 汇总方法数：{len(summaries)}。",
        ]
    )
    (root / "experiment_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    config_path: Path,
    method: str = "all",
    trial: int | None = None,
    prepare_only: bool = False,
    experiment_root: Path | None = None,
) -> list[dict[str, object]]:
    root = Path(experiment_root) if experiment_root else Path(__file__).resolve().parent
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _load_tuned_coarse_configs(config, root)
    logger, log_path = _configure_logging(root)
    logger.info("实验根目录: %s", root)
    data_summaries = _prepare_inputs(config, root, logger)
    if prepare_only:
        logger.info("数据准备完成，按 --prepare-only 停止")
        return []

    left = load_prepared(_resolve(root, config["data"]["left_prepared"]))
    right = load_prepared(_resolve(root, config["data"]["right_prepared"]))
    perturbations = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )
    trial_indices = [trial - 1] if trial is not None else list(range(len(perturbations)))
    if any(index < 0 or index >= len(perturbations) for index in trial_indices):
        raise ValueError(f"trial 必须在 1 到 {len(perturbations)} 之间")
    methods = list(METHOD_DIRS) if method == "all" else [method]
    if any(item not in METHOD_DIRS for item in methods):
        raise ValueError(f"method 必须是 all 或 {list(METHOD_DIRS)}")

    rows: list[dict[str, object]] = []
    trial_one_registered: dict[str, np.ndarray] = {}
    for trial_index in trial_indices:
        trial_number = trial_index + 1
        perturbation: Perturbation = perturbations[trial_index]
        ground_truth = invert_transform(perturbation.matrix)
        perturbed_source = apply_transform(left["xyz_m"], perturbation.matrix)
        for method_name in methods:
            coarse_config = _coarse_config_for_method(config, method_name)
            coarse_output_dir = (
                root
                / "results"
                / "coarse_registration"
                / method_name
                / f"trial_{trial_number:03d}"
            )
            coarse_output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = (
                root
                / "results"
                / METHOD_DIRS[method_name]
                / f"trial_{trial_number:03d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_transform(
                output_dir / "T_perturb.json",
                perturbation.matrix,
                euler_deg=perturbation.euler_deg.tolist(),
                translation_m=perturbation.translation_m.tolist(),
            )
            _save_transform(output_dir / "T_gt.json", ground_truth)
            logger.info(
                "trial=%03d method=%s coarse 开始",
                trial_number,
                method_name,
            )
            coarse_started = time.perf_counter()
            try:
                coarse_result = _run_coarse_registration(
                    method_name,
                    perturbed_source,
                    right["xyz_m"],
                    left["rgb"],
                    right["rgb"],
                    left["spectral6"],
                    right["spectral6"],
                    config,
                    trial_number,
                )
                coarse_runtime = time.perf_counter() - coarse_started
                coarse_registered = apply_transform(
                    perturbed_source,
                    coarse_result.transformation,
                )
                coarse_evaluation = full_metrics(
                    perturbed_source,
                    right["xyz_m"],
                    coarse_result.transformation,
                    ground_truth,
                    float(config["evaluation_max_correspondence_m"]),
                    float(config["success_rotation_deg"]),
                    float(config["success_translation_m"]),
                )
                coarse_metrics: dict[str, object] = {
                    "status": "ok",
                    **coarse_evaluation,
                    "runtime_s": coarse_runtime,
                    "source_downsampled_count": (
                        coarse_result.source_downsampled_count
                    ),
                    "target_downsampled_count": (
                        coarse_result.target_downsampled_count
                    ),
                    "ransac_fitness": coarse_result.fitness,
                    "ransac_inlier_rmse_m": coarse_result.inlier_rmse,
                    "selected_candidate_id": (
                        coarse_result.selected_candidate_id
                    ),
                    "fallback_identity": coarse_result.fallback_identity,
                    "selection_reason": coarse_result.selection_reason,
                    "identity_baseline_score": coarse_result.parameters.get(
                        "identity_baseline_score"
                    ),
                    "selected_score": coarse_result.parameters.get(
                        "selected_score"
                    ),
                    "candidate_count": max(
                        0,
                        len(coarse_result.candidates) - 1,
                    ),
                    "valid_candidate_count": coarse_result.parameters.get(
                        "valid_candidate_count",
                        0,
                    ),
                    **coarse_result.parameters,
                    "error_message": "",
                }
                _save_transform(
                    coarse_output_dir / "T_coarse.json",
                    coarse_result.transformation,
                    **coarse_result.parameters,
                )
                _save_json(
                    coarse_output_dir / "coarse_metrics.json",
                    coarse_metrics,
                )
                _write_coarse_candidates(
                    coarse_output_dir,
                    coarse_result.candidates,
                )
                write_coarse_registration_clouds(
                    coarse_output_dir,
                    coarse_registered,
                    right["xyz_m"],
                    left["rgb"],
                    right["rgb"],
                )
                logger.info(
                    "trial=%03d method=%s coarse 完成 rot=%.4fdeg trans=%.4fm",
                    trial_number,
                    method_name,
                    coarse_evaluation["rotation_error_deg"],
                    coarse_evaluation["translation_error_m"],
                )
            except Exception as exc:
                coarse_runtime = time.perf_counter() - coarse_started
                error_message = f"{type(exc).__name__}: {exc}"
                _save_transform(
                    coarse_output_dir / "T_coarse.json",
                    np.eye(4),
                    status="error",
                    modality=method_name,
                    config_id=coarse_config["config_id"],
                )
                _save_json(
                    coarse_output_dir / "coarse_metrics.json",
                    {
                        "status": "error",
                        "modality": method_name,
                        "config_id": coarse_config["config_id"],
                        "rotation_error_deg": None,
                        "translation_error_m": None,
                        "fitness": None,
                        "inlier_rmse_m": None,
                        "success": False,
                        "runtime_s": coarse_runtime,
                        "enabled": bool(coarse_config["enabled"]),
                        "error_message": error_message,
                    },
                )
                _write_coarse_candidates(coarse_output_dir, [])
                row = {
                    "method": method_name,
                    "trial": trial_number,
                    "status": "error",
                    "rotation_error_deg": None,
                    "translation_error_m": None,
                    "fitness": None,
                    "inlier_rmse_m": None,
                    "success": False,
                    "runtime_s": 0.0,
                    "coarse_method": method_name,
                    "coarse_config_id": coarse_config["config_id"],
                    "coarse_rotation_error_deg": None,
                    "coarse_translation_error_m": None,
                    "coarse_fitness": None,
                    "coarse_inlier_rmse_m": None,
                    "coarse_runtime_s": coarse_runtime,
                    "error_message": (
                        f"coarse registration failed: {error_message}"
                    ),
                }
                _save_transform(
                    output_dir / "T_est.json",
                    np.eye(4),
                    status="error",
                )
                _save_json(output_dir / "metrics.json", row)
                _write_history(output_dir / "convergence.csv", [])
                rows.append(row)
                logger.exception(
                    "trial=%03d method=%s coarse 失败",
                    trial_number,
                    method_name,
                )
                continue

            coarse_fields = {
                "coarse_method": method_name,
                "coarse_config_id": coarse_result.parameters.get(
                    "config_id",
                    coarse_config["config_id"],
                ),
                "coarse_rotation_error_deg": coarse_metrics[
                    "rotation_error_deg"
                ],
                "coarse_translation_error_m": coarse_metrics[
                    "translation_error_m"
                ],
                "coarse_fitness": coarse_metrics["fitness"],
                "coarse_inlier_rmse_m": coarse_metrics["inlier_rmse_m"],
                "coarse_runtime_s": coarse_metrics["runtime_s"],
            }
            logger.info("trial=%03d method=%s 开始", trial_number, method_name)
            started = time.perf_counter()
            try:
                result = _run_method(
                    method_name,
                    perturbed_source,
                    right["xyz_m"],
                    left["rgb"],
                    right["rgb"],
                    left["spectral6"],
                    right["spectral6"],
                    config,
                    init=coarse_result.transformation,
                )
                runtime = time.perf_counter() - started
                estimated = result.transformation
                registered = apply_transform(perturbed_source, estimated)
                metrics = full_metrics(
                    perturbed_source,
                    right["xyz_m"],
                    estimated,
                    ground_truth,
                    float(config["evaluation_max_correspondence_m"]),
                    float(config["success_rotation_deg"]),
                    float(config["success_translation_m"]),
                )
                row: dict[str, object] = {
                    "method": method_name,
                    "trial": trial_number,
                    "status": "ok",
                    **metrics,
                    "runtime_s": runtime,
                    **coarse_fields,
                    "error_message": "",
                }
                _save_transform(output_dir / "T_est.json", estimated)
                _save_json(output_dir / "metrics.json", row)
                _write_history(output_dir / "convergence.csv", result.history)
                write_registration_clouds(
                    output_dir,
                    perturbed_source,
                    registered,
                    right["xyz_m"],
                    left["rgb"],
                    right["rgb"],
                )
                if trial_number == 1:
                    save_before_coarse_after(
                        perturbed_source,
                        right["xyz_m"],
                        coarse_registered,
                        registered,
                        output_dir / "before_after.png",
                        METHOD_LABELS[method_name],
                    )
                    trial_one_registered[method_name] = registered
                logger.info(
                    "trial=%03d method=%s 完成 rot=%.4fdeg trans=%.4fm fitness=%.4f rmse=%.4fm",
                    trial_number,
                    method_name,
                    metrics["rotation_error_deg"],
                    metrics["translation_error_m"],
                    metrics["fitness"],
                    metrics["inlier_rmse_m"],
                )
            except Exception as exc:
                runtime = time.perf_counter() - started
                row = {
                    "method": method_name,
                    "trial": trial_number,
                    "status": "error",
                    "rotation_error_deg": None,
                    "translation_error_m": None,
                    "fitness": None,
                    "inlier_rmse_m": None,
                    "success": False,
                    "runtime_s": runtime,
                    **coarse_fields,
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                _save_transform(output_dir / "T_est.json", np.eye(4), status="error")
                _save_json(output_dir / "metrics.json", row)
                _write_history(output_dir / "convergence.csv", [])
                logger.exception("trial=%03d method=%s 失败", trial_number, method_name)
            rows.append(row)

    summaries = write_result_tables(root, rows)
    write_method_parameter_summary(
        root / "results",
        config,
        left["wavelengths_nm"],
    )
    save_method_comparison(
        trial_one_registered,
        right["xyz_m"],
        root / "results" / "trial_001_method_comparison.png",
    )
    save_error_boxplots(rows, root / "results" / "registration_error_boxplots.png")
    save_coarse_and_fine_error_boxplots(
        rows,
        root / "results" / "coarse_and_fine_error_boxplots.png",
    )
    _write_experiment_log(
        root,
        config,
        data_summaries,
        rows,
        summaries,
        log_path,
        [(index + 1, perturbations[index]) for index in trial_indices],
    )
    logger.info("实验结束，共生成 %d 条结果", len(rows))
    return rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "default.json")
    parser.add_argument(
        "--method",
        choices=("geometry", "rgb", "spectral6", "all"),
        default="all",
    )
    parser.add_argument("--trial", type=int, help="只运行指定的1-based trial")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(
        arguments.config,
        method=arguments.method,
        trial=arguments.trial,
        prepare_only=arguments.prepare_only,
    )
