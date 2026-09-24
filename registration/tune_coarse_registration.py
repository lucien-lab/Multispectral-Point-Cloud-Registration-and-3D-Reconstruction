#!/usr/bin/env python3
"""分别调优几何、RGB和六波段粗配准参数。"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.coarse_registration import (
    register_geometry_coarse,
    register_rgb_coarse,
    register_spectral_coarse,
)
from src.coarse_tuning import (
    TUNING_TRIALS,
    generate_search_configs,
    write_tuning_outputs,
)
from src.data_prepare import load_prepared, prepare_snapshot
from src.evaluation import full_metrics
from src.transform import (
    apply_transform,
    generate_perturbations,
    invert_transform,
)


METHODS = ("geometry", "rgb", "spectral6")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _logger(root: Path) -> logging.Logger:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger = logging.getLogger(f"coarse_tuning_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(
            log_dir / f"coarse_tuning_{timestamp}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _register(
    method: str,
    source: dict[str, object],
    target: dict[str, object],
    perturbed_xyz,
    config: dict[str, object],
    seed: int,
):
    if method == "geometry":
        return register_geometry_coarse(
            perturbed_xyz,
            target["xyz_m"],
            config=config,
            seed=seed,
        )
    if method == "rgb":
        return register_rgb_coarse(
            perturbed_xyz,
            target["xyz_m"],
            source["rgb"],
            target["rgb"],
            config=config,
            seed=seed,
        )
    if method == "spectral6":
        return register_spectral_coarse(
            perturbed_xyz,
            target["xyz_m"],
            source["spectral6"],
            target["spectral6"],
            config=config,
            seed=seed,
        )
    raise ValueError(f"未知调参方法: {method}")


def _write_combined_summary(
    root: Path,
    selected: dict[str, dict[str, object]],
) -> None:
    results_dir = root / "results"
    rows = [
        {"method": method, **summary}
        for method, summary in selected.items()
    ]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (results_dir / "coarse_tuning_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in fields}
            for row in rows
        )
    lines = [
        "# Coarse tuning summary",
        "",
        "| method | selected config | success / 5 | normalized error | median translation / m | runtime / s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['config_id']} | "
            f"{row['success_count']}/{row['trial_count']} | "
            f"{float(row['normalized_error_mean']):.6g} | "
            f"{float(row['translation_error_m_median']):.6g} | "
            f"{float(row['runtime_s_mean']):.6g} |"
        )
    (results_dir / "coarse_tuning_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_tuning(
    config_path: Path,
    *,
    methods: Iterable[str] = METHODS,
    experiment_root: Path | None = None,
    search_spaces: dict[str, list[dict[str, object]]] | None = None,
    trial_numbers: Iterable[int] = tuple(sorted(TUNING_TRIALS)),
) -> dict[str, dict[str, object]]:
    root = (
        Path(experiment_root)
        if experiment_root is not None
        else Path(__file__).resolve().parent
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    logger = _logger(root)
    data_config = config["data"]
    prepare_snapshot(
        _resolve(root, data_config["left_input"]),
        _resolve(root, data_config["left_prepared"]),
        "left",
    )
    prepare_snapshot(
        _resolve(root, data_config["right_input"]),
        _resolve(root, data_config["right_prepared"]),
        "right",
    )
    left = load_prepared(_resolve(root, data_config["left_prepared"]))
    right = load_prepared(_resolve(root, data_config["right_prepared"]))
    perturbations = generate_perturbations(
        int(config["trial_count"]),
        int(config["seed"]),
        float(config["max_rotation_deg"]),
        float(config["max_translation_m"]),
    )
    selected_by_method: dict[str, dict[str, object]] = {}
    common = config["coarse_registration"]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"未知调参方法: {method}")
        configs = (
            search_spaces[method]
            if search_spaces is not None
            else generate_search_configs(method, common)
        )
        rows: list[dict[str, object]] = []
        logger.info("%s 开始，共 %d 组配置", method, len(configs))
        for config_index, coarse_config in enumerate(configs, 1):
            for trial_number in trial_numbers:
                if trial_number not in TUNING_TRIALS:
                    raise ValueError("调参运行只允许trial 1–5")
                perturbation = perturbations[trial_number - 1]
                perturbed = apply_transform(
                    left["xyz_m"],
                    perturbation.matrix,
                )
                ground_truth = invert_transform(perturbation.matrix)
                started = time.perf_counter()
                try:
                    result = _register(
                        method,
                        left,
                        right,
                        perturbed,
                        coarse_config,
                        int(config["seed"]) + trial_number,
                    )
                    runtime = time.perf_counter() - started
                    metrics = full_metrics(
                        perturbed,
                        right["xyz_m"],
                        result.transformation,
                        ground_truth,
                        float(config["evaluation_max_correspondence_m"]),
                        float(config["success_rotation_deg"]),
                        float(config["success_translation_m"]),
                    )
                    row = {
                        "method": method,
                        "config_id": coarse_config["config_id"],
                        "trial": trial_number,
                        "status": "ok",
                        **metrics,
                        "runtime_s": runtime,
                        "selected_candidate_id": result.selected_candidate_id,
                        "fallback_identity": result.fallback_identity,
                        "error_message": "",
                    }
                except Exception as exc:
                    runtime = time.perf_counter() - started
                    row = {
                        "method": method,
                        "config_id": coarse_config["config_id"],
                        "trial": trial_number,
                        "status": "error",
                        "rotation_error_deg": None,
                        "translation_error_m": None,
                        "fitness": None,
                        "inlier_rmse_m": None,
                        "success": False,
                        "runtime_s": runtime,
                        "selected_candidate_id": "",
                        "fallback_identity": False,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                    logger.exception(
                        "%s config=%s trial=%d 失败",
                        method,
                        coarse_config["config_id"],
                        trial_number,
                    )
                rows.append(row)
            logger.info(
                "%s 配置进度 %d/%d: %s",
                method,
                config_index,
                len(configs),
                coarse_config["config_id"],
            )
        selected = write_tuning_outputs(
            root / "results" / "coarse_tuning" / method,
            method,
            configs,
            rows,
        )
        selected_by_method[method] = selected
        logger.info("%s 最优配置: %s", method, selected["config_id"])
    _write_combined_summary(root, selected_by_method)
    return selected_by_method


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "default.json",
    )
    parser.add_argument(
        "--method",
        choices=(*METHODS, "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = METHODS if args.method == "all" else (args.method,)
    run_tuning(args.config, methods=methods)


if __name__ == "__main__":
    main()
