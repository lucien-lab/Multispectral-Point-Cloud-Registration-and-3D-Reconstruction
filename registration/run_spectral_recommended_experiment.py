#!/usr/bin/env python3
"""运行六通道鲁棒多光谱配准独立实验。"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy

from src.coarse_registration import (
    register_geometry_coarse,
    register_rgb_coarse,
    register_spectral_coarse,
)
from src.coarse_tuning import (
    generate_search_configs,
    select_best_config,
    summarize_search_results,
)
from src.data_prepare import load_prepared
from src.evaluation import full_metrics
from src.icp_spectral import register_spectral_icp
from src.icp_spectral_robust import register_robust_spectral_icp
from src.icp_rgb import register_rgb_icp
from src.spectral_calibration import (
    CalibrationTrial,
    SpectralCalibration,
    apply_source_calibration,
    collect_overlap_correspondences,
    fit_overlap_calibration,
)
from src.spectral_experiment import (
    VARIANTS,
    beats_rgb_reference,
    chamfer_metrics_mm,
    protocol_rounds,
    select_best_schedule,
    trial_split,
)
from src.transform import (
    apply_transform,
    generate_perturbations,
    invert_transform,
)
from src.visualization import (
    save_before_coarse_after,
    write_coarse_registration_clouds,
    write_registration_clouds,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_transform(
    path: Path,
    transformation: np.ndarray,
    **metadata: object,
) -> None:
    _write_json(
        path,
        {
            "transformation": np.asarray(
                transformation,
                dtype=np.float64,
            ),
            **metadata,
        },
    )


def _write_records(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row))
    if not fields:
        fields = ["record_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in fields}
            for row in records
        )


def _write_records_markdown(
    path: Path,
    rows: Iterable[dict[str, object]],
) -> None:
    records = list(rows)
    fields = (
        "round",
        "variant",
        "trial",
        "status",
        "rotation_error_deg",
        "translation_error_m",
        "overlap_100mm_pct",
        "overlap_250mm_pct",
        "global_chamfer_mm",
        "success",
    )
    lines = [
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in records:
        lines.append(
            "| "
            + " | ".join(
                "N/A" if row.get(field) is None else str(row.get(field, ""))
                for field in fields
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coarse_config(config: dict[str, Any]) -> dict[str, object]:
    common = {
        key: value
        for key, value in config["coarse_registration"].items()
        if key != "methods"
    }
    common.update(
        config["coarse_registration"]["methods"]["spectral6"]
    )
    common["modality"] = "spectral6"
    return common


def _geometry_coarse_config(
    config: dict[str, Any],
) -> dict[str, object]:
    common = {
        key: value
        for key, value in config["coarse_registration"].items()
        if key != "methods"
    }
    common.update(
        config["coarse_registration"]["methods"]["geometry"]
    )
    common["modality"] = "geometry"
    return common


def _rgb_coarse_config(config: dict[str, Any]) -> dict[str, object]:
    common = {
        key: value
        for key, value in config["coarse_registration"].items()
        if key != "methods"
    }
    common.update(config["coarse_registration"]["methods"]["rgb"])
    common["modality"] = "rgb"
    return common


def _calibration_payload(
    calibration: SpectralCalibration,
) -> dict[str, object]:
    return {
        "wavelengths_nm": calibration.wavelengths_nm,
        "gain": calibration.gain,
        "offset": calibration.offset,
        "residual_mad": calibration.residual_mad,
        "correlation": calibration.correlation,
        "band_weights": calibration.band_weights,
        "pair_count": calibration.pair_count,
        "model": "target = gain * source + offset",
    }


def _load_round_calibration(
    root: Path,
    config: dict[str, Any],
    round_number: int,
) -> SpectralCalibration:
    path = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "calibration"
        / "calibration.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少冻结的第{round_number}轮校正参数：{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SpectralCalibration(
        wavelengths_nm=np.asarray(
            payload["wavelengths_nm"],
            dtype=np.float64,
        ),
        gain=np.asarray(payload["gain"], dtype=np.float64),
        offset=np.asarray(payload["offset"], dtype=np.float64),
        residual_mad=np.asarray(
            payload["residual_mad"],
            dtype=np.float64,
        ),
        correlation=np.asarray(
            payload["correlation"],
            dtype=np.float64,
        ),
        band_weights=np.asarray(
            payload["band_weights"],
            dtype=np.float64,
        ),
        pair_count=int(payload["pair_count"]),
    )


def _load_round_tuning(
    root: Path,
    config: dict[str, Any],
    round_number: int,
) -> tuple[dict[str, object], tuple[float, ...]]:
    tuning_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "tuning"
    )
    coarse_path = tuning_dir / "selected_coarse_config.json"
    schedule_path = tuning_dir / "selected_schedule.json"
    if not coarse_path.is_file() or not schedule_path.is_file():
        raise FileNotFoundError(
            f"缺少冻结的第{round_number}轮选参结果：{tuning_dir}"
        )
    coarse_payload = json.loads(
        coarse_path.read_text(encoding="utf-8")
    )
    schedule_payload = json.loads(
        schedule_path.read_text(encoding="utf-8")
    )
    return (
        dict(coarse_payload["config"]),
        tuple(
            float(value)
            for value in schedule_payload["lambda_spec_levels"]
        ),
    )


def _save_calibration_outputs(
    output_dir: Path,
    calibration: SpectralCalibration,
    trials: list[CalibrationTrial],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "calibration.json",
        _calibration_payload(calibration),
    )
    pairs = collect_overlap_correspondences(
        trials,
        left["xyz_m"],
        right["xyz_m"],
        config=config["calibration"],
    )
    corrected = apply_source_calibration(
        left["spectral6"],
        calibration,
    )
    rows: list[dict[str, object]] = []
    for source_index, target_index in zip(
        pairs.source_indices,
        pairs.target_indices,
        strict=True,
    ):
        row: dict[str, object] = {
            "source_index": int(source_index),
            "target_index": int(target_index),
        }
        for band, wavelength in enumerate(
            calibration.wavelengths_nm,
        ):
            row[f"source_{int(wavelength)}"] = float(
                left["spectral6"][source_index, band]
            )
            row[f"corrected_{int(wavelength)}"] = float(
                corrected[source_index, band]
            )
            row[f"target_{int(wavelength)}"] = float(
                right["spectral6"][target_index, band]
            )
        rows.append(row)
    _write_records(output_dir / "calibration_pairs.csv", rows)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for band, (axis, wavelength) in enumerate(
        zip(axes.ravel(), calibration.wavelengths_nm, strict=True)
    ):
        target_values = right["spectral6"][
            pairs.target_indices,
            band,
        ]
        axis.scatter(
            left["spectral6"][pairs.source_indices, band],
            target_values,
            s=5,
            alpha=0.35,
            label="before",
        )
        axis.scatter(
            corrected[pairs.source_indices, band],
            target_values,
            s=5,
            alpha=0.35,
            label="after",
        )
        axis.set_title(f"{int(wavelength)} nm")
        axis.set_xlabel("source response")
        axis.set_ylabel("target response")
    axes.ravel()[0].legend()
    fig.savefig(
        output_dir / "spectral_fit_before_after.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)
    lines = [
        "# 六波段重叠区相对校正",
        "",
        f"- 有效去重点对：{calibration.pair_count}",
        "- 模型：`target = gain * source + offset`",
        "- 对应点仅由纯几何粗配准、双向最近邻、距离和法线门限生成。",
        "",
        "| wavelength / nm | gain | offset | residual MAD | correlation | band weight |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for index, wavelength in enumerate(calibration.wavelengths_nm):
        lines.append(
            f"| {int(wavelength)} | {calibration.gain[index]:.8g} | "
            f"{calibration.offset[index]:.8g} | "
            f"{calibration.residual_mad[index]:.8g} | "
            f"{calibration.correlation[index]:.8g} | "
            f"{calibration.band_weights[index]:.8g} |"
        )
    (output_dir / "calibration_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _fit_round_calibration(
    root: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    round_number: int,
) -> SpectralCalibration:
    trial_numbers = (
        range(1, 6)
        if round_number == 1
        else range(1, 11)
    )
    perturbations = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )
    calibration_trials: list[CalibrationTrial] = []
    for trial_number in trial_numbers:
        perturbation = perturbations[trial_number - 1]
        source_perturbed = apply_transform(
            left["xyz_m"],
            perturbation.matrix,
        )
        coarse = register_geometry_coarse(
            source_perturbed,
            right["xyz_m"],
            config=_geometry_coarse_config(config),
            seed=int(config["seed"]) + trial_number,
        )
        calibration_trials.append(
            CalibrationTrial(
                trial=trial_number,
                source_to_target=(
                    coarse.transformation @ perturbation.matrix
                ),
            )
        )
    calibration = fit_overlap_calibration(
        calibration_trials,
        left["xyz_m"],
        right["xyz_m"],
        left["spectral6"],
        right["spectral6"],
        left["wavelengths_nm"],
        config=config["calibration"],
    )
    calibration_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "calibration"
    )
    _save_calibration_outputs(
        calibration_dir,
        calibration,
        calibration_trials,
        left,
        right,
        config,
    )
    return calibration


def _tune_round(
    root: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    calibration: SpectralCalibration,
    *,
    round_number: int,
) -> tuple[dict[str, object], tuple[float, ...]]:
    _, training_trials = trial_split(round_number)
    perturbations = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )
    source_spectral = apply_source_calibration(
        left["spectral6"],
        calibration,
    )
    common = dict(config["coarse_registration"])
    coarse_configs = generate_search_configs("spectral6", common)
    for candidate in coarse_configs:
        candidate["attribute_channel_weights"] = (
            calibration.band_weights.tolist()
        )
    coarse_rows: list[dict[str, object]] = []
    coarse_transforms: dict[tuple[str, int], np.ndarray] = {}
    coarse_errors: dict[tuple[str, int], str] = {}
    for config_index, candidate in enumerate(coarse_configs, 1):
        print(
            f"round{round_number} coarse {config_index}/{len(coarse_configs)} "
            f"{candidate['config_id']}",
            flush=True,
        )
        for trial_number in training_trials:
            perturbation = perturbations[trial_number - 1]
            source_perturbed = apply_transform(
                left["xyz_m"],
                perturbation.matrix,
            )
            ground_truth = invert_transform(perturbation.matrix)
            started = time.perf_counter()
            try:
                result = register_spectral_coarse(
                    source_perturbed,
                    right["xyz_m"],
                    source_spectral,
                    right["spectral6"],
                    config=candidate,
                    seed=int(config["seed"]) + trial_number,
                )
                metrics = full_metrics(
                    source_perturbed,
                    right["xyz_m"],
                    result.transformation,
                    ground_truth,
                    float(config["evaluation_max_correspondence_m"]),
                    float(config["success_rotation_deg"]),
                    float(config["success_translation_m"]),
                )
                coarse_rows.append(
                    {
                        "round": round_number,
                        "config_id": candidate["config_id"],
                        "trial": trial_number,
                        "status": "ok",
                        **metrics,
                        "runtime_s": time.perf_counter() - started,
                    }
                )
                coarse_transforms[
                    (str(candidate["config_id"]), trial_number)
                ] = np.asarray(
                    result.transformation,
                    dtype=np.float64,
                ).copy()
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                coarse_rows.append(
                    {
                        "round": round_number,
                        "config_id": candidate["config_id"],
                        "trial": trial_number,
                        "status": "error",
                        "success": False,
                        "runtime_s": time.perf_counter() - started,
                        "error_message": message,
                    }
                )
                coarse_errors[
                    (str(candidate["config_id"]), trial_number)
                ] = message
    coarse_summaries = summarize_search_results(
        coarse_rows,
        tuning_trials=frozenset(training_trials),
        success_rotation_deg=float(config["success_rotation_deg"]),
        success_translation_m=float(
            config["success_translation_m"]
        ),
    )
    selected_coarse_summary = select_best_config(coarse_summaries)
    selected_coarse = next(
        item
        for item in coarse_configs
        if item["config_id"]
        == selected_coarse_summary["config_id"]
    )
    tuning_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "tuning"
    )
    tuning_dir.mkdir(parents=True, exist_ok=True)
    _write_records(
        tuning_dir / "coarse_search_results.csv",
        coarse_rows,
    )
    _write_json(
        tuning_dir / "selected_coarse_config.json",
        {
            "selection": selected_coarse_summary,
            "config": selected_coarse,
            "training_trials": list(training_trials),
        },
    )

    schedules = [
        tuple(float(value) for value in schedule)
        for schedule in config["robust_registration"][
            "lambda_spec_schedules_round1"
        ]
    ]
    if round_number == 2:
        schedules.extend(
            tuple(float(value) for value in schedule)
            for schedule in config["robust_registration"][
                "lambda_spec_schedules_round2_extra"
            ]
        )
    schedule_rows: list[dict[str, object]] = []
    selected_config_id = str(selected_coarse["config_id"])
    cached_coarse = {
        trial_number: coarse_transforms[
            (selected_config_id, trial_number)
        ]
        for trial_number in training_trials
        if (selected_config_id, trial_number) in coarse_transforms
    }
    robust_config = config["robust_registration"]
    for schedule_index, schedule in enumerate(schedules, 1):
        schedule_id = "_".join(f"{value:.3f}" for value in schedule)
        print(
            f"round{round_number} schedule "
            f"{schedule_index}/{len(schedules)} {schedule_id}",
            flush=True,
        )
        for trial_number in training_trials:
            if trial_number not in cached_coarse:
                schedule_rows.append(
                    {
                        "round": round_number,
                        "schedule_id": schedule_id,
                        "trial": trial_number,
                        "status": "error",
                        "success": False,
                        "runtime_s": 0.0,
                        "error_message": (
                            "coarse initialization unavailable: "
                            + coarse_errors.get(
                                (selected_config_id, trial_number),
                                "unknown coarse failure",
                            )
                        ),
                    }
                )
                continue
            perturbation = perturbations[trial_number - 1]
            source_perturbed = apply_transform(
                left["xyz_m"],
                perturbation.matrix,
            )
            ground_truth = invert_transform(perturbation.matrix)
            started = time.perf_counter()
            try:
                result = register_robust_spectral_icp(
                    source_perturbed,
                    right["xyz_m"],
                    source_spectral,
                    right["spectral6"],
                    list(config["voxel_sizes_m"]),
                    list(config["max_correspondence_distances_m"]),
                    list(config["max_iterations"]),
                    band_weights=calibration.band_weights,
                    lambda_spec_levels=schedule,
                    init=cached_coarse[trial_number],
                    validation_config=config[
                        "registration_validation"
                    ],
                    huber_delta_geo=float(
                        robust_config["huber_delta_geo"]
                    ),
                    huber_delta_spec=float(
                        robust_config["huber_delta_spec"]
                    ),
                    min_gradient_neighbors=int(
                        robust_config["min_gradient_neighbors"]
                    ),
                    max_gradient_condition=float(
                        robust_config["max_gradient_condition"]
                    ),
                    normal_angle_deg=float(
                        robust_config["normal_angle_deg"]
                    ),
                    min_correspondences=int(
                        robust_config["min_correspondences"]
                    ),
                    trim_fraction=float(
                        robust_config["trim_fraction"]
                    ),
                    gradient_max_nn=int(
                        robust_config["gradient_max_nn"]
                    ),
                    gradient_irls_iterations=int(
                        robust_config["gradient_irls_iterations"]
                    ),
                )
                metrics = full_metrics(
                    source_perturbed,
                    right["xyz_m"],
                    result.transformation,
                    ground_truth,
                    float(config["evaluation_max_correspondence_m"]),
                    float(config["success_rotation_deg"]),
                    float(config["success_translation_m"]),
                )
                schedule_rows.append(
                    {
                        "round": round_number,
                        "schedule_id": schedule_id,
                        "trial": trial_number,
                        "status": "ok",
                        **metrics,
                        "runtime_s": time.perf_counter() - started,
                    }
                )
            except Exception as exc:
                schedule_rows.append(
                    {
                        "round": round_number,
                        "schedule_id": schedule_id,
                        "trial": trial_number,
                        "status": "error",
                        "success": False,
                        "runtime_s": time.perf_counter() - started,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
    selected_schedule_summary = select_best_schedule(
        schedule_rows,
        training_trials=training_trials,
    )
    selected_schedule = next(
        schedule
        for schedule in schedules
        if "_".join(f"{value:.3f}" for value in schedule)
        == selected_schedule_summary["schedule_id"]
    )
    _write_records(
        tuning_dir / "schedule_search_results.csv",
        schedule_rows,
    )
    _write_json(
        tuning_dir / "selected_schedule.json",
        {
            "selection": selected_schedule_summary,
            "lambda_spec_levels": selected_schedule,
            "training_trials": list(training_trials),
        },
    )
    _write_records(
        root
        / str(config["output_dir"])
        / f"round{round_number}_tuning.csv",
        [*coarse_rows, *schedule_rows],
    )
    return selected_coarse, selected_schedule


def _run_baseline_trial(
    root: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    trial_number: int,
    round_number: int,
) -> dict[str, object]:
    perturbation = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )[trial_number - 1]
    ground_truth = invert_transform(perturbation.matrix)
    source_perturbed = apply_transform(
        left["xyz_m"],
        perturbation.matrix,
    )
    output_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "baseline"
        / f"trial_{trial_number:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_transform(
        output_dir / "T_perturb.json",
        perturbation.matrix,
        euler_deg=perturbation.euler_deg,
        translation_m=perturbation.translation_m,
    )
    _write_transform(output_dir / "T_gt.json", ground_truth)

    coarse_started = time.perf_counter()
    coarse = register_spectral_coarse(
        source_perturbed,
        right["xyz_m"],
        left["spectral6"],
        right["spectral6"],
        config=_coarse_config(config),
        seed=int(config["seed"]) + trial_number,
    )
    coarse_runtime = time.perf_counter() - coarse_started
    coarse_registered = apply_transform(
        source_perturbed,
        coarse.transformation,
    )
    _write_transform(
        output_dir / "T_coarse.json",
        coarse.transformation,
        **coarse.parameters,
    )
    write_coarse_registration_clouds(
        output_dir,
        coarse_registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )

    started = time.perf_counter()
    result = register_spectral_icp(
        source_perturbed,
        right["xyz_m"],
        left["spectral6"],
        right["spectral6"],
        list(config["voxel_sizes_m"]),
        list(config["max_correspondence_distances_m"]),
        list(config["max_iterations"]),
        lambda_geo=float(config["lambda_geo_spectral"]),
        lambda_spec=float(config["lambda_spec_spectral"]),
        init=coarse.transformation,
        validation_config=config.get("registration_validation"),
    )
    runtime = time.perf_counter() - started
    registered = apply_transform(source_perturbed, result.transformation)
    metrics = full_metrics(
        source_perturbed,
        right["xyz_m"],
        result.transformation,
        ground_truth,
        float(config["evaluation_max_correspondence_m"]),
        float(config["success_rotation_deg"]),
        float(config["success_translation_m"]),
    )
    metrics.update(
        chamfer_metrics_mm(
            registered,
            right["xyz_m"],
            trim_fraction=0.70,
        )
    )
    row: dict[str, object] = {
        "round": round_number,
        "variant": "baseline",
        "trial": trial_number,
        "status": "ok",
        **metrics,
        "runtime_s": runtime,
        "coarse_runtime_s": coarse_runtime,
        "coarse_config_id": coarse.parameters.get(
            "config_id",
            _coarse_config(config).get("config_id", ""),
        ),
        "schedule_id": "legacy_0968_0032",
    }
    _write_transform(output_dir / "T_est.json", result.transformation)
    _write_json(output_dir / "metrics.json", row)
    _write_records(output_dir / "convergence.csv", result.history)
    _write_records(
        output_dir / "correspondence_diagnostics.csv",
        result.history,
    )
    write_registration_clouds(
        output_dir,
        source_perturbed,
        registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )
    if trial_number == 1:
        save_before_coarse_after(
            source_perturbed,
            right["xyz_m"],
            coarse_registered,
            registered,
            output_dir / "before_after.png",
            "Six-band baseline",
        )
    return row


def _run_variant_trial(
    root: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    variant: str,
    trial_number: int,
    round_number: int,
    calibration: SpectralCalibration | None,
    lambda_spec_schedule: tuple[float, ...] | None = None,
    coarse_override: dict[str, object] | None = None,
) -> dict[str, object]:
    if variant == "baseline":
        return _run_baseline_trial(
            root,
            config,
            left,
            right,
            trial_number=trial_number,
            round_number=round_number,
        )
    if calibration is None:
        raise ValueError(f"{variant}需要光谱校正参数")
    perturbation = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )[trial_number - 1]
    ground_truth = invert_transform(perturbation.matrix)
    source_perturbed = apply_transform(
        left["xyz_m"],
        perturbation.matrix,
    )
    source_spectral = apply_source_calibration(
        left["spectral6"],
        calibration,
    )
    output_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / variant
        / f"trial_{trial_number:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_transform(
        output_dir / "T_perturb.json",
        perturbation.matrix,
        euler_deg=perturbation.euler_deg,
        translation_m=perturbation.translation_m,
    )
    _write_transform(output_dir / "T_gt.json", ground_truth)

    coarse_config = (
        dict(coarse_override)
        if coarse_override is not None
        else _coarse_config(config)
    )
    if variant == "robust_full":
        coarse_config["attribute_channel_weights"] = (
            calibration.band_weights.tolist()
        )
    coarse_started = time.perf_counter()
    coarse = register_spectral_coarse(
        source_perturbed,
        right["xyz_m"],
        source_spectral,
        right["spectral6"],
        config=coarse_config,
        seed=int(config["seed"]) + trial_number,
    )
    coarse_runtime = time.perf_counter() - coarse_started
    coarse_registered = apply_transform(
        source_perturbed,
        coarse.transformation,
    )
    _write_transform(
        output_dir / "T_coarse.json",
        coarse.transformation,
        **coarse.parameters,
    )
    write_coarse_registration_clouds(
        output_dir,
        coarse_registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )

    started = time.perf_counter()
    if variant in {"calibrated", "aligned_validation"}:
        result = register_spectral_icp(
            source_perturbed,
            right["xyz_m"],
            source_spectral,
            right["spectral6"],
            list(config["voxel_sizes_m"]),
            list(config["max_correspondence_distances_m"]),
            list(config["max_iterations"]),
            lambda_geo=float(config["lambda_geo_spectral"]),
            lambda_spec=float(config["lambda_spec_spectral"]),
            init=coarse.transformation,
            validation_config=config.get("registration_validation"),
            validation_mode=(
                "tangent"
                if variant == "aligned_validation"
                else "direct"
            ),
        )
        schedule_id = "legacy_0968_0032"
    else:
        robust_config = config["robust_registration"]
        schedule = (
            tuple(lambda_spec_schedule)
            if lambda_spec_schedule is not None
            else tuple(
                robust_config["lambda_spec_schedules_round1"][0]
            )
        )
        result = register_robust_spectral_icp(
            source_perturbed,
            right["xyz_m"],
            source_spectral,
            right["spectral6"],
            list(config["voxel_sizes_m"]),
            list(config["max_correspondence_distances_m"]),
            list(config["max_iterations"]),
            band_weights=calibration.band_weights,
            lambda_spec_levels=schedule,
            init=coarse.transformation,
            validation_config=config.get("registration_validation"),
            huber_delta_geo=float(
                robust_config["huber_delta_geo"]
            ),
            huber_delta_spec=float(
                robust_config["huber_delta_spec"]
            ),
            min_gradient_neighbors=int(
                robust_config["min_gradient_neighbors"]
            ),
            max_gradient_condition=float(
                robust_config["max_gradient_condition"]
            ),
            normal_angle_deg=float(
                robust_config["normal_angle_deg"]
            ),
            min_correspondences=int(
                robust_config["min_correspondences"]
            ),
            trim_fraction=float(
                robust_config["trim_fraction"]
            ),
            gradient_max_nn=int(
                robust_config["gradient_max_nn"]
            ),
            gradient_irls_iterations=int(
                robust_config["gradient_irls_iterations"]
            ),
        )
        schedule_id = "_".join(f"{value:.3f}" for value in schedule)
    runtime = time.perf_counter() - started
    registered = apply_transform(source_perturbed, result.transformation)
    metrics = full_metrics(
        source_perturbed,
        right["xyz_m"],
        result.transformation,
        ground_truth,
        float(config["evaluation_max_correspondence_m"]),
        float(config["success_rotation_deg"]),
        float(config["success_translation_m"]),
    )
    metrics.update(
        chamfer_metrics_mm(
            registered,
            right["xyz_m"],
            trim_fraction=0.70,
        )
    )
    row: dict[str, object] = {
        "round": round_number,
        "variant": variant,
        "trial": trial_number,
        "status": "ok",
        **metrics,
        "runtime_s": runtime,
        "coarse_runtime_s": coarse_runtime,
        "coarse_config_id": coarse.parameters.get(
            "config_id",
            coarse_config.get("config_id", ""),
        ),
        "schedule_id": schedule_id,
    }
    _write_transform(output_dir / "T_est.json", result.transformation)
    _write_json(output_dir / "metrics.json", row)
    _write_records(output_dir / "convergence.csv", result.history)
    _write_records(
        output_dir / "correspondence_diagnostics.csv",
        result.history,
    )
    write_registration_clouds(
        output_dir,
        source_perturbed,
        registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )
    if trial_number == 1:
        save_before_coarse_after(
            source_perturbed,
            right["xyz_m"],
            coarse_registered,
            registered,
            output_dir / "before_after.png",
            variant.replace("_", " ").title(),
        )
    return row


def _run_rgb_reference_trial(
    root: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    trial_number: int,
    round_number: int,
) -> dict[str, object]:
    perturbation = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )[trial_number - 1]
    ground_truth = invert_transform(perturbation.matrix)
    source_perturbed = apply_transform(
        left["xyz_m"],
        perturbation.matrix,
    )
    output_dir = (
        root
        / str(config["output_dir"])
        / f"round{round_number}"
        / "rgb_reference"
        / f"trial_{trial_number:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_transform(
        output_dir / "T_perturb.json",
        perturbation.matrix,
        euler_deg=perturbation.euler_deg,
        translation_m=perturbation.translation_m,
    )
    _write_transform(output_dir / "T_gt.json", ground_truth)
    coarse_started = time.perf_counter()
    coarse = register_rgb_coarse(
        source_perturbed,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
        config=_rgb_coarse_config(config),
        seed=int(config["seed"]) + trial_number,
    )
    coarse_runtime = time.perf_counter() - coarse_started
    coarse_registered = apply_transform(
        source_perturbed,
        coarse.transformation,
    )
    _write_transform(
        output_dir / "T_coarse.json",
        coarse.transformation,
        **coarse.parameters,
    )
    write_coarse_registration_clouds(
        output_dir,
        coarse_registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )
    started = time.perf_counter()
    result = register_rgb_icp(
        source_perturbed,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
        list(config["voxel_sizes_m"]),
        list(config["max_correspondence_distances_m"]),
        list(config["max_iterations"]),
        lambda_geometric=float(config["lambda_geometric_rgb"]),
        init=coarse.transformation,
        validation_config=config["registration_validation"],
    )
    runtime = time.perf_counter() - started
    registered = apply_transform(source_perturbed, result.transformation)
    metrics = full_metrics(
        source_perturbed,
        right["xyz_m"],
        result.transformation,
        ground_truth,
        float(config["evaluation_max_correspondence_m"]),
        float(config["success_rotation_deg"]),
        float(config["success_translation_m"]),
    )
    metrics.update(chamfer_metrics_mm(registered, right["xyz_m"]))
    row: dict[str, object] = {
        "round": round_number,
        "variant": "rgb_reference",
        "trial": trial_number,
        "status": "ok",
        **metrics,
        "runtime_s": runtime,
        "coarse_runtime_s": coarse_runtime,
        "coarse_config_id": coarse.parameters.get(
            "config_id",
            _rgb_coarse_config(config).get("config_id", ""),
        ),
        "schedule_id": "rgb_0968_0032",
    }
    _write_transform(output_dir / "T_est.json", result.transformation)
    _write_json(output_dir / "metrics.json", row)
    _write_records(output_dir / "convergence.csv", result.history)
    _write_records(
        output_dir / "correspondence_diagnostics.csv",
        result.history,
    )
    write_registration_clouds(
        output_dir,
        source_perturbed,
        registered,
        right["xyz_m"],
        left["rgb"],
        right["rgb"],
    )
    return row


def _write_ablation_summary(
    output_root: Path,
    rows: list[dict[str, object]],
) -> None:
    _write_records(output_root / "ablation_results.csv", rows)
    lines = [
        "# 六通道鲁棒配准消融结果",
        "",
        "| round | variant | trial | rotation / deg | translation / m | overlap100 / % | overlap250 / % | success |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    def display(row: dict[str, object], field: str) -> str:
        value = row.get(field)
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric:.6g}" if np.isfinite(numeric) else "N/A"

    for row in rows:
        lines.append(
            f"| {row['round']} | {row['variant']} | {row['trial']} | "
            f"{display(row, 'rotation_error_deg')} | "
            f"{display(row, 'translation_error_m')} | "
            f"{display(row, 'overlap_100mm_pct')} | "
            f"{display(row, 'overlap_250mm_pct')} | "
            f"{row['success']} |"
        )
    (output_root / "ablation_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _error_row(
    round_number: int,
    variant: str,
    trial_number: int,
    exc: Exception,
) -> dict[str, object]:
    return {
        "round": round_number,
        "variant": variant,
        "trial": trial_number,
        "status": "error",
        "rotation_error_deg": None,
        "translation_error_m": None,
        "overlap_100mm_pct": None,
        "overlap_250mm_pct": None,
        "fitness": None,
        "inlier_rmse_m": None,
        "global_chamfer_mm": None,
        "trimmed_chamfer_mm": None,
        "distance_median_mm": None,
        "success": False,
        "runtime_s": 0.0,
        "error_message": f"{type(exc).__name__}: {exc}",
    }


def _persist_trial_error(
    output_root: Path,
    row: dict[str, object],
) -> None:
    output_dir = (
        output_root
        / f"round{int(row['round'])}"
        / str(row["variant"])
        / f"trial_{int(row['trial']):03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", row)
    message = str(row.get("error_message", "unknown error"))
    details = str(row.get("traceback", "")).strip()
    (output_dir / "error.txt").write_text(
        message + ("\n\n" + details if details else "") + "\n",
        encoding="utf-8",
    )
    print(
        f"round={row['round']} variant={row['variant']} "
        f"trial={int(row['trial']):03d} ERROR {message}",
        file=sys.stderr,
        flush=True,
    )


def _plot_protocol_errors(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    variants = [
        name
        for name in (*VARIANTS, "rgb_reference")
        if any(
            row.get("variant") == name
            and row.get("status") == "ok"
            for row in rows
        )
    ]
    if not variants:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = (
        ("rotation_error_deg", "Rotation error / deg"),
        ("translation_error_m", "Translation error / m"),
        ("overlap_100mm_pct", "Overlap 100 mm / %"),
        ("global_chamfer_mm", "Global Chamfer / mm"),
    )
    for axis, (field, label) in zip(axes.ravel(), fields, strict=True):
        values = [
            [
                float(row[field])
                for row in rows
                if row.get("variant") == name
                and row.get("status") == "ok"
                and row.get(field) is not None
            ]
            for name in variants
        ]
        axis.boxplot(values, tick_labels=variants, showmeans=True)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _write_protocol_log(
    root: Path,
    output_root: Path,
    config_path: Path,
    config: dict[str, Any],
    rows: list[dict[str, object]],
    round1_gate: dict[str, object],
    round2_gate: dict[str, object] | None,
) -> None:
    left_path = _resolve(root, config["data"]["left_prepared"])
    right_path = _resolve(root, config["data"]["right_prepared"])
    perturbations = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )
    lines = [
        "# 六通道鲁棒多光谱配准独立实验日志",
        "",
        f"- 配置：`{config_path}`",
        f"- 输入左点云：`{left_path}`",
        f"- 左点云 SHA-256：`{_sha256(left_path)}`",
        f"- 输入右点云：`{right_path}`",
        f"- 右点云 SHA-256：`{_sha256(right_path)}`",
        "- 六波段顺序：495、696、600、803、545、642 nm",
        "- 当前反射率仅作左右相对响应校正，不宣称绝对物理反射率。",
        "- T_gt仅用于选参排序和最终评价。",
        "- round1：trial 1–5调参，trial 6–10独立评价。",
        f"- round1是否超过RGB：{round1_gate['passed']}",
        f"- 是否执行round2：{round2_gate is not None}",
    ]
    if round2_gate is not None:
        lines.extend(
            [
                "- round2：trial 1–10二次调参，trial 11–20最终确认。",
                f"- round2是否超过同期RGB：{round2_gate['passed']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 联合门槛",
            "",
            "要求旋转和平移均低于RGB、成功率不低于RGB，且100/250 mm重叠率下降均不超过1个百分点。",
            "",
            f"- 共记录结果：{len(rows)}",
            f"- 错误结果：{sum(row.get('status') != 'ok' for row in rows)}",
            "",
            "## 实验环境",
            "",
            f"- OS：{platform.platform()}",
            f"- Python：{platform.python_version()} (`{sys.executable}`)",
            f"- Open3D：{o3d.__version__}",
            f"- NumPy：{np.__version__}",
            f"- SciPy：{scipy.__version__}",
            f"- Matplotlib：{matplotlib.__version__}",
            "- 可复现线程设置：`OMP_NUM_THREADS=1`",
            "",
            "## 随机SE(3)扰动",
            "",
            f"- seed：{config['seed']}",
            f"- 每轴旋转范围：±{config['max_rotation_deg']} deg",
            f"- 每轴平移范围：±{config['max_translation_m']} m",
            "",
            "| trial | rx/deg | ry/deg | rz/deg | tx/m | ty/m | tz/m |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for trial_number, perturbation in enumerate(perturbations, 1):
        euler = perturbation.euler_deg
        translation = perturbation.translation_m
        lines.append(
            f"| {trial_number} | {euler[0]:.8g} | {euler[1]:.8g} | "
            f"{euler[2]:.8g} | {translation[0]:.8g} | "
            f"{translation[1]:.8g} | {translation[2]:.8g} |"
        )
    lines.extend(
        [
            "",
            "## 冻结参数",
            "",
            "- 第一轮校正：`round1/calibration/calibration.json`",
            "- 第一轮粗配准：`round1/tuning/selected_coarse_config.json`",
            "- 第一轮精配准：`round1/tuning/selected_schedule.json`",
        ]
    )
    if round2_gate is not None:
        lines.extend(
            [
                "- 第二轮校正：`round2/calibration/calibration.json`",
                "- 第二轮粗配准：`round2/tuning/selected_coarse_config.json`",
                "- 第二轮精配准：`round2/tuning/selected_schedule.json`",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "trial 11–20仍是同一对左右点云的新扰动，只验证扰动泛化，不代表跨场景泛化。",
        ]
    )
    (output_root / "experiment_log.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_automatic_protocol(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    output_root = root / str(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    calibration1 = _fit_round_calibration(
        root,
        config,
        left,
        right,
        round_number=1,
    )
    coarse1, schedule1 = _tune_round(
        root,
        config,
        left,
        right,
        calibration1,
        round_number=1,
    )
    round1_rows: list[dict[str, object]] = []
    for trial_number in range(1, 11):
        for variant_name in VARIANTS:
            started = time.perf_counter()
            try:
                row = _run_variant_trial(
                    root,
                    config,
                    left,
                    right,
                    variant=variant_name,
                    trial_number=trial_number,
                    round_number=1,
                    calibration=calibration1,
                    lambda_spec_schedule=schedule1,
                    coarse_override=(
                        coarse1
                        if variant_name == "robust_full"
                        else None
                    ),
                )
            except Exception as exc:
                row = _error_row(1, variant_name, trial_number, exc)
                row["runtime_s"] = time.perf_counter() - started
                row["traceback"] = traceback.format_exc()
                _persist_trial_error(output_root, row)
            round1_rows.append(row)
    rgb1: list[dict[str, object]] = []
    for trial_number in range(6, 11):
        started = time.perf_counter()
        try:
            row = _run_rgb_reference_trial(
                root,
                config,
                left,
                right,
                trial_number=trial_number,
                round_number=1,
            )
        except Exception as exc:
            row = _error_row(
                1,
                "rgb_reference",
                trial_number,
                exc,
            )
            row["runtime_s"] = time.perf_counter() - started
            row["traceback"] = traceback.format_exc()
            _persist_trial_error(output_root, row)
        rgb1.append(row)
    heldout1 = [
        row
        for row in round1_rows
        if 6 <= int(row["trial"]) <= 10
    ]
    _write_records(
        output_root / "round1_heldout.csv",
        [*heldout1, *rgb1],
    )
    _write_records_markdown(
        output_root / "round1_heldout.md",
        [*heldout1, *rgb1],
    )
    robust_heldout1 = [
        row
        for row in heldout1
        if row["variant"] == "robust_full"
    ]
    gate1 = beats_rgb_reference(
        robust_heldout1,
        rgb1,
        overlap_tolerance_pct=float(
            config["protocol"]["overlap_tolerance_pct"]
        ),
    )
    _write_json(output_root / "round1_rgb_gate.json", gate1)
    all_rows.extend(round1_rows)
    all_rows.extend(rgb1)

    gate2: dict[str, object] | None = None
    if protocol_rounds(round1_passed=bool(gate1["passed"])) == (1, 2):
        calibration2 = _fit_round_calibration(
            root,
            config,
            left,
            right,
            round_number=2,
        )
        coarse2, schedule2 = _tune_round(
            root,
            config,
            left,
            right,
            calibration2,
            round_number=2,
        )
        robust2: list[dict[str, object]] = []
        rgb2: list[dict[str, object]] = []
        for trial_number in range(11, 21):
            started = time.perf_counter()
            try:
                row = _run_variant_trial(
                    root,
                    config,
                    left,
                    right,
                    variant="robust_full",
                    trial_number=trial_number,
                    round_number=2,
                    calibration=calibration2,
                    lambda_spec_schedule=schedule2,
                    coarse_override=coarse2,
                )
            except Exception as exc:
                row = _error_row(
                    2,
                    "robust_full",
                    trial_number,
                    exc,
                )
                row["runtime_s"] = time.perf_counter() - started
                row["traceback"] = traceback.format_exc()
                _persist_trial_error(output_root, row)
            robust2.append(row)
            started = time.perf_counter()
            try:
                rgb_row = _run_rgb_reference_trial(
                    root,
                    config,
                    left,
                    right,
                    trial_number=trial_number,
                    round_number=2,
                )
            except Exception as exc:
                rgb_row = _error_row(
                    2,
                    "rgb_reference",
                    trial_number,
                    exc,
                )
                rgb_row["runtime_s"] = time.perf_counter() - started
                rgb_row["traceback"] = traceback.format_exc()
                _persist_trial_error(output_root, rgb_row)
            rgb2.append(rgb_row)
        _write_records(
            output_root / "round2_confirmation.csv",
            [*robust2, *rgb2],
        )
        _write_records_markdown(
            output_root / "round2_confirmation.md",
            [*robust2, *rgb2],
        )
        gate2 = beats_rgb_reference(
            robust2,
            rgb2,
            overlap_tolerance_pct=float(
                config["protocol"]["overlap_tolerance_pct"]
            ),
        )
        _write_json(output_root / "round2_rgb_gate.json", gate2)
        all_rows.extend(robust2)
        all_rows.extend(rgb2)

    _write_ablation_summary(output_root, all_rows)
    _plot_protocol_errors(
        all_rows,
        output_root / "registration_metric_boxplots.png",
    )
    _write_protocol_log(
        root,
        output_root,
        config_path,
        config,
        all_rows,
        gate1,
        gate2,
    )
    return all_rows


def run_protocol(
    config_path: Path,
    *,
    stage: str = "all",
    variant: str = "all",
    trial: int | None = None,
    experiment_root: Path | None = None,
) -> list[dict[str, object]]:
    root = (
        Path(experiment_root)
        if experiment_root is not None
        else Path(__file__).resolve().parent
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if stage not in {
        "calibrate",
        "tune",
        "evaluate",
        "retune",
        "confirm",
        "all",
    }:
        raise ValueError("未知stage")
    if variant not in {"all", *VARIANTS}:
        raise ValueError("未知variant")
    left = load_prepared(_resolve(root, config["data"]["left_prepared"]))
    right = load_prepared(_resolve(root, config["data"]["right_prepared"]))
    if stage == "all" and variant == "all" and trial is None:
        return _run_automatic_protocol(
            root,
            Path(config_path),
            config,
            left,
            right,
        )
    round_number = 2 if stage in {"retune", "confirm"} else 1
    if stage == "calibrate":
        _fit_round_calibration(
            root,
            config,
            left,
            right,
            round_number=1,
        )
        return []
    if stage in {"tune", "retune"}:
        calibration = _fit_round_calibration(
            root,
            config,
            left,
            right,
            round_number=round_number,
        )
        _tune_round(
            root,
            config,
            left,
            right,
            calibration,
            round_number=round_number,
        )
        return []
    trial_numbers = (
        [trial]
        if trial is not None
        else (
            list(range(11, 21))
            if stage == "confirm"
            else (
                list(range(6, 11))
                if stage == "evaluate"
                else list(range(1, int(config["trial_count"]) + 1))
            )
        )
    )
    if any(
        item is None
        or item < 1
        or item > int(config["trial_count"])
        for item in trial_numbers
    ):
        raise ValueError("trial超出配置范围")
    variants = list(VARIANTS) if variant == "all" else [variant]
    needs_calibration = any(item != "baseline" for item in variants)
    use_frozen = (
        needs_calibration
        and (
            stage == "confirm"
            or (stage == "evaluate" and trial is None)
        )
    )
    if use_frozen:
        calibration = _load_round_calibration(root, config, round_number)
        coarse_override, schedule = _load_round_tuning(
            root,
            config,
            round_number,
        )
    else:
        calibration = (
            _fit_round_calibration(
                root,
                config,
                left,
                right,
                round_number=round_number,
            )
            if needs_calibration
            else None
        )
        coarse_override = None
        schedule = None
    rows: list[dict[str, object]] = []
    for trial_number in trial_numbers:
        for variant_name in variants:
            rows.append(
                _run_variant_trial(
                    root,
                    config,
                    left,
                    right,
                    variant=variant_name,
                    trial_number=int(trial_number),
                    round_number=round_number,
                    calibration=calibration,
                    lambda_spec_schedule=(
                        schedule
                        if variant_name == "robust_full"
                        else None
                    ),
                    coarse_override=(
                        coarse_override
                        if variant_name == "robust_full"
                        else None
                    ),
                )
            )
    output_root = root / str(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_ablation_summary(output_root, rows)
    return rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "spectral6_recommended.json",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "calibrate",
            "tune",
            "evaluate",
            "retune",
            "confirm",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "all"),
        default="all",
    )
    parser.add_argument("--trial", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    log_dir = root / str(config["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{args.stage}_{timestamp}.log"

    class Tee:
        def __init__(self, *streams: object) -> None:
            self.streams = streams

        def write(self, text: str) -> int:
            for stream in self.streams:
                stream.write(text)
                stream.flush()
            return len(text)

        def flush(self) -> None:
            for stream in self.streams:
                stream.flush()

    with log_path.open("w", encoding="utf-8") as log_handle:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log_handle)),
            contextlib.redirect_stderr(Tee(sys.stderr, log_handle)),
        ):
            print(f"log={log_path}", flush=True)
            run_protocol(
                args.config,
                stage=args.stage,
                variant=args.variant,
                trial=args.trial,
            )


if __name__ == "__main__":
    main()
