#!/usr/bin/env python3
"""用已保存变换补算联合评价与初始点云输出，不重新运行ICP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_experiment import (
    METHOD_DIRS,
    _load_tuned_coarse_configs,
    write_result_tables,
)
from src.data_prepare import load_prepared
from src.evaluation import full_metrics
from src.result_reporting import write_method_parameter_summary
from src.transform import apply_transform
from src.visualization import write_initial_registration_cloud


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_transform(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("matrix") if isinstance(payload, dict) else payload
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{path} 不包含有限的4x4变换矩阵")
    return result


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _trial_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("trial_"))
    except ValueError as exc:
        raise ValueError(f"无法从目录名解析trial编号: {path.name}") from exc


def backfill_outputs(
    config_path: Path,
    *,
    experiment_root: Path | None = None,
) -> list[dict[str, object]]:
    """补齐已有成功结果的评价字段、初始PLY和汇总表。"""
    root = (
        Path(experiment_root).resolve()
        if experiment_root is not None
        else Path(__file__).resolve().parent
    )
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    _load_tuned_coarse_configs(config, root)

    left_path = _resolve(root, config["data"]["left_prepared"])
    right_path = _resolve(root, config["data"]["right_prepared"])
    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少处理后点云: {path}")
    left = load_prepared(left_path)
    right = load_prepared(right_path)

    rows: list[dict[str, object]] = []
    for method, folder in METHOD_DIRS.items():
        method_dir = root / "results" / folder
        if not method_dir.is_dir():
            continue
        for trial_dir in sorted(method_dir.glob("trial_*")):
            if not trial_dir.is_dir():
                continue
            trial = _trial_number(trial_dir)
            metrics_path = trial_dir / "metrics.json"
            required = [
                metrics_path,
                trial_dir / "T_perturb.json",
                trial_dir / "T_gt.json",
                trial_dir / "T_est.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "trial结果不完整，缺少: " + ", ".join(missing)
                )

            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            perturbation = _load_transform(trial_dir / "T_perturb.json")
            perturbed_source = apply_transform(left["xyz_m"], perturbation)
            write_initial_registration_cloud(
                trial_dir,
                perturbed_source,
                right["xyz_m"],
            )

            if existing.get("status", "ok") == "ok":
                estimated = _load_transform(trial_dir / "T_est.json")
                ground_truth = _load_transform(trial_dir / "T_gt.json")
                existing.update(
                    full_metrics(
                        perturbed_source,
                        right["xyz_m"],
                        estimated,
                        ground_truth,
                        float(config["evaluation_max_correspondence_m"]),
                        float(config["success_rotation_deg"]),
                        float(config["success_translation_m"]),
                    )
                )
            existing["method"] = method
            existing["trial"] = trial
            existing.setdefault("status", "ok")
            existing.setdefault("runtime_s", 0.0)
            existing.setdefault("error_message", "")
            _save_json(metrics_path, existing)
            rows.append(existing)

    rows.sort(key=lambda row: (int(row["trial"]), str(row["method"])))
    write_result_tables(root, rows)
    write_method_parameter_summary(
        root / "results",
        config,
        left["wavelengths_nm"],
    )
    return rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "default.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = backfill_outputs(args.config)
    print(
        json.dumps(
            {
                "updated_result_count": len(rows),
                "initial_ply_count": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
