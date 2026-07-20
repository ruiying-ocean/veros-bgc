#!/usr/bin/env python
"""Assemble monthly diagnostics across a fresh run and a restart segment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

MODEL_MONTH_DAYS = 30
TIME_FIELDS = (
    "day",
    "iteration",
    "phytoplankton_mean",
    "zooplankton_mean",
    "detritus_mean",
    "po4_mean",
    "no3_mean",
    "dop_mean",
    "don_mean",
    "diazotrophs_mean",
    "oxygen_mean",
    "dic_mean",
    "alkalinity_mean",
    "npp_pg_c_per_year",
    "airsea_co2_pg_c_per_year",
    "airsea_co2_mmol_c_per_second",
    "nitrogen_fixation_mmol_n_per_second",
    "water_column_denitrification_mmol_n_per_second",
    "benthic_denitrification_mmol_n_per_second",
    "phosphorus_inventory_mmol",
    "nitrogen_inventory_mmol",
    "carbon_inventory_mmol",
    "cumulative_airsea_co2_mmol",
    "cumulative_nitrogen_source_mmol",
    "phosphorus_drift_ppm",
    "nitrogen_budget_residual_ppm",
    "carbon_budget_residual_ppm",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-csv",
        type=Path,
        default=Path("results/bgc_10yr_conservative/daily_metrics.csv"),
    )
    parser.add_argument(
        "--tail-dir",
        type=Path,
        default=Path("results/bgc_10yr_surface_tail"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bgc_10yr_monthly"),
    )
    parser.add_argument("--split-day", type=int, default=9 * 360)
    return parser.parse_args()


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _to_number(key, value):
    if key == "iteration":
        return int(float(value))
    return float(value)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = _read_csv(args.initial_csv)
    tail_rows = _read_csv(args.tail_dir / "daily_metrics.csv")
    time_fields = tuple(
        key for key in TIME_FIELDS if key in initial_rows[0] and key in tail_rows[0]
    )
    initial_phosphorus = float(initial_rows[0]["phosphorus_inventory_mmol"])
    initial_nitrogen = float(initial_rows[0].get("nitrogen_inventory_mmol", 0.0))
    initial_carbon = float(initial_rows[0]["carbon_inventory_mmol"])

    prefix = [
        row
        for row in initial_rows
        if float(row["day"]) <= args.split_day
        and float(row["day"]) % MODEL_MONTH_DAYS == 0.0
    ]
    split_row = next(row for row in prefix if float(row["day"]) == args.split_day)
    cumulative_at_split = float(split_row["cumulative_airsea_co2_mmol"])
    cumulative_nitrogen_at_split = float(
        split_row.get("cumulative_nitrogen_source_mmol", 0.0)
    )
    suffix = [row for row in tail_rows if float(row["day"]) > args.split_day]

    monthly = []
    for source, restarted in ((prefix, False), (suffix, True)):
        for row in source:
            values = {key: _to_number(key, row[key]) for key in time_fields}
            values["iteration"] = int(round(values["day"]))
            if restarted:
                values["cumulative_airsea_co2_mmol"] += cumulative_at_split
                if "cumulative_nitrogen_source_mmol" in values:
                    values["cumulative_nitrogen_source_mmol"] += (
                        cumulative_nitrogen_at_split
                    )
            values["phosphorus_drift_ppm"] = (
                (values["phosphorus_inventory_mmol"] - initial_phosphorus)
                / initial_phosphorus
                * 1e6
            )
            values["carbon_budget_residual_ppm"] = (
                (
                    values["carbon_inventory_mmol"]
                    - initial_carbon
                    - values["cumulative_airsea_co2_mmol"]
                )
                / initial_carbon
                * 1e6
            )
            if "nitrogen_budget_residual_ppm" in values and initial_nitrogen > 0.0:
                values["nitrogen_budget_residual_ppm"] = (
                    (
                        values["nitrogen_inventory_mmol"]
                        - initial_nitrogen
                        - values["cumulative_nitrogen_source_mmol"]
                    )
                    / initial_nitrogen
                    * 1e6
                )
            monthly.append(values)

    expected_days = np.arange(0, 10 * 360 + 1, MODEL_MONTH_DAYS, dtype=float)
    actual_days = np.asarray([row["day"] for row in monthly])
    np.testing.assert_array_equal(actual_days, expected_days)

    csv_path = args.output_dir / "monthly_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=time_fields)
        writer.writeheader()
        writer.writerows(monthly)

    with np.load(args.tail_dir / "bgc_1yr_diagnostics.npz") as tail_data:
        payload = {
            key: np.asarray([row[key] for row in monthly]) for key in time_fields
        }
        payload.update(
            {
                key: np.asarray(tail_data[key])
                for key in tail_data.files
                if key not in time_fields
            }
        )
        mean_field_names = sorted(
            key
            for key in tail_data.files
            if key.startswith("mean_") and key != "mean_surface_sample_count"
        )
    np.savez_compressed(
        args.output_dir / "bgc_10yr_monthly_diagnostics.npz",
        **payload,
    )
    shutil.copy2(
        args.tail_dir / "checkpoint.restart.h5",
        args.output_dir / "checkpoint.restart.h5",
    )

    tail_metadata = json.loads((args.tail_dir / "metadata.json").read_text())
    metadata = {
        "veros_version": tail_metadata["veros_version"],
        "backend": tail_metadata["backend"],
        "float_type": tail_metadata["float_type"],
        "start_day": 0.0,
        "end_day": float(monthly[-1]["day"]),
        "model_years": 10,
        "model_month_days": MODEL_MONTH_DAYS,
        "samples": len(monthly),
        "final_phosphorus_drift_ppm": monthly[-1]["phosphorus_drift_ppm"],
        "final_nitrogen_budget_residual_ppm": monthly[-1].get(
            "nitrogen_budget_residual_ppm"
        ),
        "final_carbon_budget_residual_ppm": monthly[-1]["carbon_budget_residual_ppm"],
        "enable_bgc_superbee_advection": tail_metadata["enable_bgc_superbee_advection"],
        "enable_bgc_conservative_clipping": tail_metadata[
            "enable_bgc_conservative_clipping"
        ],
        "enable_nitrogen": tail_metadata.get("enable_nitrogen", False),
        "biology_parameters": tail_metadata.get("biology_parameters"),
        "restart_split_day": args.split_day,
    }
    if tail_metadata.get("mean_surface_sample_count", 0):
        mean_sample_count = tail_metadata["mean_surface_sample_count"]
        mean_sample_interval = tail_metadata.get(
            "mean_surface_sample_interval_days",
            tail_metadata["days"] / mean_sample_count,
        )
        metadata.update(
            {
                "mean_surface_sample_count": mean_sample_count,
                "mean_surface_sample_interval_days": mean_sample_interval,
                "mean_surface_first_sample_day": tail_metadata.get(
                    "mean_surface_first_sample_day",
                    tail_metadata["start_day"] + mean_sample_interval,
                ),
                "mean_surface_last_sample_day": tail_metadata.get(
                    "mean_surface_last_sample_day",
                    tail_metadata["end_day"],
                ),
                "mean_surface_fields": mean_field_names,
            }
        )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
