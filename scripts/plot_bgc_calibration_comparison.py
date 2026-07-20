#!/usr/bin/env python
"""Compare the Fortran-rate and legacy-calibrated P-only BGC integrations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_YEAR_DAYS = 360
FORTRAN_COLOR = "#4C78A8"
CALIBRATED_COLOR = "#E07A2D"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fortran-rate",
        type=Path,
        default=Path(
            "results/bgc_10yr_monthly/bgc_10yr_monthly_diagnostics.npz"
        ),
    )
    parser.add_argument(
        "--calibrated",
        type=Path,
        default=Path(
            "results/bgc_10yr_calibrated/bgc_10yr_monthly_diagnostics.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/bgc_10yr_calibration_comparison"),
        help="Output path without a suffix",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("results/bgc_10yr_calibration_comparison.csv"),
    )
    return parser.parse_args()


def _load(path):
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _annual_mean(arrays, key):
    day = arrays["day"]
    return np.asarray(
        [
            np.nanmean(
                arrays[key][
                    (day > (year - 1) * MODEL_YEAR_DAYS)
                    & (day <= year * MODEL_YEAR_DAYS)
                ]
            )
            for year in range(1, 11)
        ]
    )


def _year_end(arrays, key):
    day = arrays["day"]
    return np.asarray(
        [
            arrays[key][np.flatnonzero(day == year * MODEL_YEAR_DAYS)[0]]
            for year in range(1, 11)
        ]
    )


def _panel_label(ax, label):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _write_source_data(path, years, series):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["year", *series]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, year in enumerate(years):
            writer.writerow(
                {
                    "year": int(year),
                    **{key: values[index] for key, values in series.items()},
                }
            )


def main():
    args = parse_args()
    fortran = _load(args.fortran_rate)
    calibrated = _load(args.calibrated)
    np.testing.assert_array_equal(fortran["day"], calibrated["day"])

    years = np.arange(1, 11)
    fortran_npp = _annual_mean(fortran, "npp_pg_c_per_year")
    calibrated_npp = _annual_mean(calibrated, "npp_pg_c_per_year")
    fortran_phyto = _annual_mean(fortran, "phytoplankton_mean")
    calibrated_phyto = _annual_mean(calibrated, "phytoplankton_mean")
    fortran_p = _year_end(fortran, "phosphorus_drift_ppm")
    calibrated_p = _year_end(calibrated, "phosphorus_drift_ppm")

    source_series = {
        "fortran_rate_npp_pg_c_per_year": fortran_npp,
        "calibrated_npp_pg_c_per_year": calibrated_npp,
        "fortran_rate_phytoplankton_mmol_n_per_m3": fortran_phyto,
        "calibrated_phytoplankton_mmol_n_per_m3": calibrated_phyto,
        "fortran_rate_phosphorus_drift_ppm": fortran_p,
        "calibrated_phosphorus_drift_ppm": calibrated_p,
    }
    _write_source_data(args.source_data, years, source_series)

    fig = plt.figure(figsize=(7.2, 5.3))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.35, 1.0),
        hspace=0.48,
        wspace=0.38,
        left=0.10,
        right=0.97,
        bottom=0.10,
        top=0.97,
    )

    ax_a = fig.add_subplot(grid[0, :])
    for values, label, color, marker in (
        (fortran_npp, "Fortran-rate P-only", FORTRAN_COLOR, "o"),
        (calibrated_npp, "Legacy global-4° calibration", CALIBRATED_COLOR, "s"),
    ):
        ax_a.plot(
            years,
            values,
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=1.7,
            label=label,
        )
    reduction = 100.0 * (1.0 - calibrated_npp[-1] / fortran_npp[-1])
    ax_a.text(
        0.985,
        0.52,
        f"Year 10: {fortran_npp[-1]:.1f} → {calibrated_npp[-1]:.1f}\n"
        f"{reduction:.1f}% lower",
        transform=ax_a.transAxes,
        ha="right",
        va="center",
        fontsize=8,
    )
    ax_a.set(
        xlabel="Model year",
        ylabel=r"Annual-mean NPP (Pg C yr$^{-1}$)",
        xlim=(0.8, 10.2),
    )
    ax_a.set_xticks(years)
    ax_a.legend(loc="upper right", fontsize=7.5)
    _panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1, 0])
    ax_b.plot(
        years,
        fortran_phyto,
        color=FORTRAN_COLOR,
        marker="o",
        markersize=3.2,
        linewidth=1.5,
    )
    ax_b.plot(
        years,
        calibrated_phyto,
        color=CALIBRATED_COLOR,
        marker="s",
        markersize=3.2,
        linewidth=1.5,
    )
    phyto_ratio = calibrated_phyto[-1] / fortran_phyto[-1]
    ax_b.text(
        0.96,
        0.76,
        f"Year 10: {phyto_ratio:.2f}× higher",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    ax_b.set(
        xlabel="Model year",
        ylabel=r"Phytoplankton (mmol N m$^{-3}$)",
        xlim=(0.8, 10.2),
    )
    ax_b.set_xticks((2, 4, 6, 8, 10))
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 1])
    ax_c.plot(
        years,
        np.abs(fortran_p),
        color=FORTRAN_COLOR,
        marker="o",
        markersize=3.2,
        linewidth=1.5,
    )
    ax_c.plot(
        years,
        np.abs(calibrated_p),
        color=CALIBRATED_COLOR,
        linestyle="--",
        marker="s",
        markersize=3.2,
        linewidth=1.5,
    )
    ax_c.set_yscale("log")
    ax_c.set(
        xlabel="Model year",
        ylabel="Absolute P inventory drift (ppm)",
        xlim=(0.8, 10.2),
    )
    ax_c.set_xticks((2, 4, 6, 8, 10))
    _panel_label(ax_c, "c")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(
            args.output.with_suffix(f".{suffix}"),
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
