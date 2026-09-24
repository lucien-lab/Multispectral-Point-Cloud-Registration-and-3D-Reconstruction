#!/usr/bin/env python3
"""仅重跑六波段细配准，并与原六波段及RGB结果做权重消融比较。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_experiment import run_pipeline, write_result_tables
from src.visualization import (
    save_coarse_and_fine_error_boxplots,
    save_error_boxplots,
)


BASELINE_DIR = Path("results/weight_ablation/spectral_050_050_baseline")
NEW_DIR = Path("results/weight_ablation/spectral_0968_0032")
EXPECTED_COARSE_CONFIG = "spectral6_v0.10_f2.0_d0.30"
RESULT_FILES = (
    "results_summary.csv",
    "method_summary.csv",
    "method_summary.md",
    "all_trials_method_summary.csv",
    "all_trials_method_summary.md",
    "heldout_method_summary.csv",
    "heldout_method_summary.md",
    "registration_error_boxplots.png",
    "coarse_and_fine_error_boxplots.png",
    "coarse_and_fine_error_boxplots.pdf",
    "coarse_and_fine_error_boxplots.svg",
    "coarse_and_fine_error_boxplots.tiff",
    "trial_001_method_comparison.png",
)
COMPARISON_FIELDS = (
    "variant",
    "method",
    "trial",
    "lambda_geo",
    "lambda_attribute",
    "status",
    "rotation_error_deg",
    "translation_error_m",
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
    "accepted_levels",
    "rollback_levels",
    "error_message",
)
SUMMARY_FIELDS = (
    "variant",
    "split",
    "trial_count",
    "success_count",
    "success_rate",
    "rotation_error_deg_mean",
    "rotation_error_deg_median",
    "translation_error_m_mean",
    "translation_error_m_median",
    "fitness_mean",
    "inlier_rmse_m_mean",
    "accepted_levels",
    "rollback_levels",
    "rotation_improvement_fraction_vs_spectral_baseline",
    "translation_improvement_fraction_vs_spectral_baseline",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fields: Iterable[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fieldnames}
            for row in rows
        )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: object) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"非有限数值: {value}")
    return number


def _trial_directories(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("trial_*") if item.is_dir())


def validate_baseline(path: Path) -> None:
    summary_path = path / "results_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"基线缺少 {summary_path}")
    rows = _read_csv(summary_path)
    counts = Counter(row["method"] for row in rows)
    if len(rows) != 30 or counts != {
        "geometry": 10,
        "rgb": 10,
        "spectral6": 10,
    }:
        raise ValueError(f"基线总表必须是三种方法各10行，实际为 {counts}")
    spectral_rows = [row for row in rows if row["method"] == "spectral6"]
    if {
        row["coarse_config_id"] for row in spectral_rows
    } != {EXPECTED_COARSE_CONFIG}:
        raise ValueError("基线六波段粗配准配置ID不一致")
    for relative in ("icp_spectral6", "coarse_registration_spectral6"):
        trials = _trial_directories(path / relative)
        if len(trials) != 10:
            raise ValueError(f"{relative} 必须包含10个trial，实际为{len(trials)}")


def archive_baseline(root: Path) -> Path:
    """首次运行时归档正式结果；已有有效快照时绝不覆盖。"""
    root = Path(root)
    destination = root / BASELINE_DIR
    if destination.exists():
        validate_baseline(destination)
        return destination

    results = root / "results"
    destination.mkdir(parents=True)
    shutil.copytree(
        results / "icp_spectral6",
        destination / "icp_spectral6",
    )
    shutil.copytree(
        results / "coarse_registration" / "spectral6",
        destination / "coarse_registration_spectral6",
    )
    for name in RESULT_FILES:
        source = results / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    shutil.copy2(root / "experiment_log.md", destination / "experiment_log.md")
    validate_baseline(destination)
    (destination / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "fine_weights": {
                    "lambda_geo": 0.5,
                    "lambda_spec": 0.5,
                },
                "coarse_config_id": EXPECTED_COARSE_CONFIG,
                "rows": 30,
                "policy": "immutable_after_first_valid_snapshot",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def build_comparison_rows(
    baseline_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """按trial对齐旧RGB、旧六波段和新六波段结果。"""
    variants = (
        (
            "rgb_baseline_0968_0032",
            "rgb",
            baseline_rows,
            0.968,
            0.032,
        ),
        (
            "spectral6_baseline_050_050",
            "spectral6",
            baseline_rows,
            0.5,
            0.5,
        ),
        (
            "spectral6_new_0968_0032",
            "spectral6",
            new_rows,
            0.968,
            0.032,
        ),
    )
    output: list[dict[str, object]] = []
    for variant, method, source_rows, lambda_geo, lambda_attribute in variants:
        selected = [row for row in source_rows if row.get("method") == method]
        for row in selected:
            output.append(
                {
                    **row,
                    "variant": variant,
                    "lambda_geo": lambda_geo,
                    "lambda_attribute": lambda_attribute,
                }
            )
    output.sort(key=lambda row: (int(row["trial"]), str(row["variant"])))
    return output


def _convergence_counts(path: Path) -> tuple[int, int]:
    rows = _read_csv(path)
    levels = [row for row in rows if row.get("record_type") == "level_validation"]
    if len(levels) != 3:
        raise ValueError(f"{path} 应包含3个level_validation，实际为{len(levels)}")
    accepted = sum(_as_bool(row.get("accepted", False)) for row in levels)
    return accepted, len(levels) - accepted


def _attach_level_counts(
    rows: list[dict[str, object]],
    root: Path,
    baseline_dir: Path,
) -> None:
    directories = {
        "rgb_baseline_0968_0032": root / "results" / "icp_rgb",
        "spectral6_baseline_050_050": baseline_dir / "icp_spectral6",
        "spectral6_new_0968_0032": root / "results" / "icp_spectral6",
    }
    for row in rows:
        path = (
            directories[str(row["variant"])]
            / f"trial_{int(row['trial']):03d}"
            / "convergence.csv"
        )
        accepted, rollback = _convergence_counts(path)
        row["accepted_levels"] = accepted
        row["rollback_levels"] = rollback


def summarize_comparison(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """分别汇总训练集、留出集和全部trial。"""
    splits = {
        "train_1_5": lambda trial: 1 <= trial <= 5,
        "heldout_6_10": lambda trial: 6 <= trial <= 10,
        "all_1_10": lambda trial: 1 <= trial <= 10,
    }
    variants = sorted({str(row["variant"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for variant in variants:
        for split, predicate in splits.items():
            selected = [
                row
                for row in rows
                if row["variant"] == variant
                and predicate(int(row["trial"]))
                and row.get("status") == "ok"
            ]
            if not selected:
                continue

            def values(key: str) -> np.ndarray:
                return np.array(
                    [_as_float(row[key]) for row in selected],
                    dtype=float,
                )

            rotation = values("rotation_error_deg")
            translation = values("translation_error_m")
            success_count = sum(_as_bool(row["success"]) for row in selected)
            summaries.append(
                {
                    "variant": variant,
                    "split": split,
                    "trial_count": len(selected),
                    "success_count": success_count,
                    "success_rate": success_count / len(selected),
                    "rotation_error_deg_mean": float(rotation.mean()),
                    "rotation_error_deg_median": float(np.median(rotation)),
                    "translation_error_m_mean": float(translation.mean()),
                    "translation_error_m_median": float(
                        np.median(translation)
                    ),
                    "fitness_mean": float(values("fitness").mean()),
                    "inlier_rmse_m_mean": float(
                        values("inlier_rmse_m").mean()
                    ),
                    "accepted_levels": sum(
                        int(row.get("accepted_levels", 0))
                        for row in selected
                    ),
                    "rollback_levels": sum(
                        int(row.get("rollback_levels", 0))
                        for row in selected
                    ),
                    "rotation_improvement_fraction_vs_spectral_baseline": "",
                    "translation_improvement_fraction_vs_spectral_baseline": "",
                }
            )

    lookup = {
        (str(row["variant"]), str(row["split"])): row
        for row in summaries
    }
    for split in splits:
        baseline = lookup.get(("spectral6_baseline_050_050", split))
        current = lookup.get(("spectral6_new_0968_0032", split))
        if baseline is None or current is None:
            continue
        for metric, output_key in (
            (
                "rotation_error_deg_mean",
                "rotation_improvement_fraction_vs_spectral_baseline",
            ),
            (
                "translation_error_m_mean",
                "translation_improvement_fraction_vs_spectral_baseline",
            ),
        ):
            old = float(baseline[metric])
            current[output_key] = (old - float(current[metric])) / old
    return summaries


def _write_comparison_markdown(
    path: Path,
    summaries: list[dict[str, object]],
) -> None:
    lines = [
        "# 六波段 ICP 细配准权重消融",
        "",
        "粗配准配置保持 `spectral6_v0.10_f2.0_d0.30` 不变；",
        "本表比较旧六波段0.5/0.5、新六波段0.968/0.032和既有RGB 0.968/0.032。",
        "",
        "| variant | split | rot mean / deg | rot median / deg | trans mean / m | trans median / m | fitness | RMSE / m | success | accepted / rollback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"heldout_6_10": 0, "train_1_5": 1, "all_1_10": 2}
    for row in sorted(
        summaries,
        key=lambda item: (order[str(item["split"])], str(item["variant"])),
    ):
        lines.append(
            f"| {row['variant']} | {row['split']} | "
            f"{float(row['rotation_error_deg_mean']):.6f} | "
            f"{float(row['rotation_error_deg_median']):.6f} | "
            f"{float(row['translation_error_m_mean']):.6f} | "
            f"{float(row['translation_error_m_median']):.6f} | "
            f"{float(row['fitness_mean']):.6f} | "
            f"{float(row['inlier_rmse_m_mean']):.6f} | "
            f"{100 * float(row['success_rate']):.1f}% | "
            f"{row['accepted_levels']} / {row['rollback_levels']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_new_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not np.isclose(float(config["lambda_geo_spectral"]), 0.968):
        raise ValueError("lambda_geo_spectral 必须为0.968")
    if not np.isclose(float(config["lambda_spec_spectral"]), 0.032):
        raise ValueError("lambda_spec_spectral 必须为0.032")
    coarse = config["coarse_registration"]["methods"]["spectral6"]
    if coarse["config_id"] != EXPECTED_COARSE_CONFIG:
        raise ValueError("六波段粗配准配置发生变化")
    return config


def _write_ablation_outputs(
    root: Path,
    baseline_dir: Path,
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output_dir = root / NEW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "new_spectral_results_summary.csv",
        new_rows,
        new_rows[0].keys(),
    )
    baseline_rows = _read_csv(baseline_dir / "results_summary.csv")
    comparison = build_comparison_rows(baseline_rows, new_rows)
    _attach_level_counts(comparison, root, baseline_dir)
    summaries = summarize_comparison(comparison)
    _write_csv(
        output_dir / "comparison_trials.csv",
        comparison,
        COMPARISON_FIELDS,
    )
    _write_csv(
        output_dir / "comparison_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    _write_comparison_markdown(
        output_dir / "comparison_summary.md",
        summaries,
    )
    if (root / "experiment_log.md").is_file():
        shutil.copy2(
            root / "experiment_log.md",
            output_dir / "spectral_only_experiment_log.md",
        )
    return comparison


def run_ablation(
    root: Path,
    config_path: Path,
) -> list[dict[str, object]]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    _validate_new_config(config_path)
    baseline_dir = archive_baseline(root)
    baseline_rows = _read_csv(baseline_dir / "results_summary.csv")

    new_rows = run_pipeline(
        config_path,
        method="spectral6",
        experiment_root=root,
    )
    if len(new_rows) != 10 or {
        str(row["method"]) for row in new_rows
    } != {"spectral6"}:
        raise ValueError("新实验必须恰好生成10条spectral6结果")

    comparison = _write_ablation_outputs(
        root,
        baseline_dir,
        new_rows,
    )
    combined_rows: list[dict[str, object]] = [
        row
        for row in baseline_rows
        if row["method"] in {"geometry", "rgb"}
    ] + new_rows
    combined_rows.sort(key=lambda row: (int(row["trial"]), str(row["method"])))
    write_result_tables(root, combined_rows)
    save_error_boxplots(
        combined_rows,
        root / "results" / "registration_error_boxplots.png",
    )
    save_coarse_and_fine_error_boxplots(
        combined_rows,
        root / "results" / "coarse_and_fine_error_boxplots.png",
    )
    return comparison


def verify_outputs(root: Path, config_path: Path) -> dict[str, object]:
    root = Path(root).resolve()
    _validate_new_config(Path(config_path).resolve())
    baseline_dir = root / BASELINE_DIR
    validate_baseline(baseline_dir)
    baseline_rows = _read_csv(baseline_dir / "results_summary.csv")
    output_dir = root / NEW_DIR
    new_rows = _read_csv(output_dir / "new_spectral_results_summary.csv")
    combined_rows = _read_csv(root / "results" / "results_summary.csv")
    comparison_rows = _read_csv(output_dir / "comparison_trials.csv")
    if len(new_rows) != 10:
        raise ValueError("新六波段结果不是10行")
    if len(combined_rows) != 30:
        raise ValueError("标准结果总表不是30行")
    if len(comparison_rows) != 30:
        raise ValueError("逐trial消融对比不是30行")
    if {
        row["coarse_config_id"] for row in new_rows
    } != {EXPECTED_COARSE_CONFIG}:
        raise ValueError("新结果的粗配准配置ID发生变化")
    if {
        (row["lambda_geo"], row["lambda_attribute"])
        for row in comparison_rows
        if row["variant"] == "spectral6_new_0968_0032"
    } != {("0.968", "0.032")}:
        raise ValueError("新六波段结果权重记录不正确")
    return {
        "baseline_rows": len(baseline_rows),
        "new_spectral_rows": len(new_rows),
        "combined_rows": len(combined_rows),
        "comparison_rows": len(comparison_rows),
        "coarse_config_id": EXPECTED_COARSE_CONFIG,
        "weights": "0.968/0.032",
        "verification": "passed",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "default.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    root = Path(__file__).resolve().parent
    if arguments.verify_only:
        report = verify_outputs(root, arguments.config)
        for key, value in report.items():
            print(f"{key}={value}")
        return
    run_ablation(root, arguments.config)
    report = verify_outputs(root, arguments.config)
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
