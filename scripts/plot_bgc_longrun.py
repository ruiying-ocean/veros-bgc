#!/usr/bin/env python
"""Plot ten-year equilibration and nitrogen-cycle diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SECONDS_PER_DAY = 86400.0
MODEL_YEAR_DAYS = 360
REDFIELD_CN = 7.1
MMOL_N_S_TO_TG_N_YR = 14.007e-15 * MODEL_YEAR_DAYS * SECONDS_PER_DAY
MMOL_N_M2_S_TO_G_C_M2_YR = REDFIELD_CN * 12.011e-3 * MODEL_YEAR_DAYS * SECONDS_PER_DAY

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

COLORS = {
    "npp": "#238A62",
    "fixation": "#D68C2C",
    "benthic": "#7B5AA6",
    "water_column": "#B64B4B",
    "no3": "#245C8A",
    "don": "#3C8D88",
    "dop": "#8B6BB1",
    "diazotrophs": "#D28A35",
    "oxygen": "#777777",
    "phosphorus": "#3775BA",
    "nitrogen": "#8156A3",
    "carbon": "#D07B29",
    "neutral": "#777777",
}

ANNUAL_FIELDS = (
    "phytoplankton_mean",
    "zooplankton_mean",
    "detritus_mean",
    "po4_mean",
    "no3_mean",
    "dop_mean",
    "don_mean",
    "diazotrophs_mean",
    "oxygen_mean",
    "npp_pg_c_per_year",
    "airsea_co2_pg_c_per_year",
    "nitrogen_fixation_mmol_n_per_second",
    "water_column_denitrification_mmol_n_per_second",
    "benthic_denitrification_mmol_n_per_second",
    "nitrogen_inventory_mmol",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/bgc_10yr_nitrogen/bgc_10yr_monthly_diagnostics.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/bgc_10yr_nitrogen_summary"),
        help="Output path without a suffix",
    )
    parser.add_argument(
        "--annual-output",
        type=Path,
        default=Path("results/bgc_10yr_nitrogen/annual_summary.csv"),
    )
    return parser.parse_args()


def _panel_label(ax, label):
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9.5,
        ha="left",
        va="bottom",
    )


def _smooth(values, window):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    kernel = np.ones(window)
    numerator = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    denominator = np.convolve(valid.astype(float), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def _annual_means(arrays):
    day = np.asarray(arrays["day"], dtype=float)
    complete_years = int(np.floor(np.nanmax(day) / MODEL_YEAR_DAYS))
    years = np.arange(1, complete_years + 1)
    annual = {"year": years}
    for key in ANNUAL_FIELDS:
        if key not in arrays:
            continue
        values = np.asarray(arrays[key], dtype=float)
        annual[key] = np.asarray(
            [
                np.nanmean(
                    values[
                        (day > (year - 1) * MODEL_YEAR_DAYS)
                        & (day <= year * MODEL_YEAR_DAYS)
                    ]
                )
                for year in years
            ]
        )

    for source, target in (
        ("nitrogen_fixation_mmol_n_per_second", "nitrogen_fixation_tg_n_per_year"),
        (
            "water_column_denitrification_mmol_n_per_second",
            "water_column_denitrification_tg_n_per_year",
        ),
        (
            "benthic_denitrification_mmol_n_per_second",
            "benthic_denitrification_tg_n_per_year",
        ),
    ):
        annual[target] = annual[source] * MMOL_N_S_TO_TG_N_YR
    return annual


def _write_annual_summary(annual, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(annual)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(annual["year"])):
            writer.writerow({key: annual[key][index] for key in fieldnames})


def _last_years_cycle(day, values, year_count, count=3):
    selected = []
    cycle_day = None
    for year in range(max(1, year_count - count + 1), year_count + 1):
        mask = (day > (year - 1) * MODEL_YEAR_DAYS) & (day <= year * MODEL_YEAR_DAYS)
        cycle = np.asarray(values[mask], dtype=float)
        current_cycle_day = day[mask] - (year - 1) * MODEL_YEAR_DAYS
        if cycle_day is None:
            cycle_day = current_cycle_day
        if cycle.size == cycle_day.size:
            np.testing.assert_array_equal(current_cycle_day, cycle_day)
            selected.append(cycle)
    stack = np.stack(selected)
    return (
        cycle_day,
        np.nanmean(stack, axis=0),
        np.nanmin(stack, axis=0),
        np.nanmax(stack, axis=0),
    )


def _relative_change(values):
    values = np.asarray(values, dtype=float)
    baseline = values[np.flatnonzero(np.isfinite(values))[0]]
    return 100.0 * (values / baseline - 1.0)


def _map_limits(field, lower=2.0, upper=98.0):
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        raise ValueError("map field contains no finite ocean values")
    return np.nanpercentile(finite, (lower, upper))


def main():
    args = parse_args()
    with np.load(args.input) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    required = {
        "day",
        "npp_pg_c_per_year",
        "nitrogen_fixation_mmol_n_per_second",
        "water_column_denitrification_mmol_n_per_second",
        "benthic_denitrification_mmol_n_per_second",
        "nitrogen_budget_residual_ppm",
        "mean_surface_no3",
        "mean_surface_po4",
        "mean_vertically_integrated_npp",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"input is missing required nitrogen diagnostics: {missing}")

    day = np.asarray(arrays["day"], dtype=float)
    model_year = day / MODEL_YEAR_DAYS
    sample_days = float(np.nanmedian(np.diff(day)))
    smooth_window = max(1, int(round(90.0 / sample_days)))
    annual = _annual_means(arrays)
    _write_annual_summary(annual, args.annual_output)
    year_count = len(annual["year"])
    if year_count < 2:
        raise ValueError("at least two complete model years are required")

    fig = plt.figure(figsize=(7.2, 12.0), constrained_layout=False)
    grid = fig.add_gridspec(
        6,
        2,
        height_ratios=(1.12, 1.0, 1.0, 0.62, 1.02, 1.02),
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.96,
        bottom=0.045,
        top=0.975,
        hspace=0.68,
        wspace=0.40,
    )

    ax_a = fig.add_subplot(grid[0, :])
    npp = np.asarray(arrays["npp_pg_c_per_year"], dtype=float)
    ax_a.axvspan(0.0, 1.0, color="#EFEFEF", zorder=0)
    ax_a.text(
        0.5,
        0.94,
        "initial adjustment",
        color=COLORS["neutral"],
        fontsize=6.8,
        ha="center",
        va="top",
        transform=ax_a.get_xaxis_transform(),
    )
    ax_a.plot(
        model_year,
        _smooth(npp, smooth_window),
        color=COLORS["npp"],
        linewidth=1.55,
        label="90-day mean",
    )
    ax_a.plot(
        annual["year"] - 0.5,
        annual["npp_pg_c_per_year"],
        color="#155C42",
        marker="o",
        markersize=3.2,
        linewidth=1.0,
        label="annual mean",
    )
    ax_a.text(
        0.985,
        0.93,
        (
            f"Year 1: {annual['npp_pg_c_per_year'][0]:.1f}\n"
            f"Year {year_count}: {annual['npp_pg_c_per_year'][-1]:.1f} Pg C yr$^{{-1}}$"
        ),
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        color="#155C42",
        fontsize=7.2,
    )
    ax_a.set(
        xlabel="Model year",
        ylabel=r"Global NPP (Pg C yr$^{-1}$)",
        xlim=(0.0, year_count),
        ylim=(0.0, None),
    )
    ax_a.legend(fontsize=6.8, loc="upper center", ncol=2)
    _panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1, 0])
    for key, label, color, marker in (
        ("nitrogen_fixation_tg_n_per_year", "N fixation", COLORS["fixation"], "o"),
        (
            "benthic_denitrification_tg_n_per_year",
            "Benthic denitrification",
            COLORS["benthic"],
            "s",
        ),
        (
            "water_column_denitrification_tg_n_per_year",
            "Water-column denitrification",
            COLORS["water_column"],
            "^",
        ),
    ):
        ax_b.plot(
            annual["year"],
            annual[key],
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.25,
            label=label,
        )
    ax_b.set(
        xlabel="Model year",
        ylabel=r"Nitrogen flux (Tg N yr$^{-1}$)",
        xlim=(0.8, year_count + 0.2),
        ylim=(0.0, None),
    )
    ax_b.legend(fontsize=6.2, loc="best")
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 1])
    for key, label, color in (
        ("no3_mean", r"NO$_3$", COLORS["no3"]),
        ("don_mean", "DON", COLORS["don"]),
        ("dop_mean", "DOP", COLORS["dop"]),
        ("diazotrophs_mean", "Diazotrophs", COLORS["diazotrophs"]),
        ("oxygen_mean", r"O$_2$", COLORS["oxygen"]),
    ):
        ax_c.plot(
            model_year,
            _smooth(_relative_change(arrays[key]), smooth_window),
            color=color,
            linewidth=1.15,
            label=label,
        )
    ax_c.axhline(0.0, color="#C8C8C8", linewidth=0.8, zorder=0)
    ax_c.set(
        xlabel="Model year",
        ylabel="Change from initialization (%)",
        xlim=(0.0, year_count),
    )
    ax_c.legend(fontsize=6.0, ncol=2, loc="best")
    _panel_label(ax_c, "c")

    ax_d = fig.add_subplot(grid[2, 0])
    cycle_day, npp_mean, npp_low, npp_high = _last_years_cycle(day, npp, year_count)
    ax_d.fill_between(
        cycle_day,
        npp_low,
        npp_high,
        color=COLORS["npp"],
        alpha=0.18,
        linewidth=0.0,
        label=f"range, years {year_count - 2}–{year_count}",
    )
    ax_d.plot(
        cycle_day,
        npp_mean,
        color=COLORS["npp"],
        linewidth=1.5,
        marker="o",
        markersize=2.6,
        label="three-year mean",
    )
    ax_d.set(
        xlabel="Day of model year",
        ylabel=r"Global NPP (Pg C yr$^{-1}$)",
        xlim=(1.0, MODEL_YEAR_DAYS),
        ylim=(0.0, None),
    )
    ax_d.legend(fontsize=6.2, loc="best")
    _panel_label(ax_d, "d")

    ax_e = fig.add_subplot(grid[2, 1])
    for key, label, color, marker in (
        ("npp_pg_c_per_year", "NPP", COLORS["npp"], "o"),
        ("no3_mean", r"NO$_3$", COLORS["no3"], "s"),
        ("don_mean", "DON", COLORS["don"], "^"),
    ):
        values = annual[key]
        change = np.abs(np.diff(values) / values[:-1]) * 100.0
        ax_e.plot(
            annual["year"][1:],
            np.maximum(change, 1e-8),
            color=color,
            marker=marker,
            markersize=2.8,
            linewidth=1.2,
            label=label,
        )
    ax_e.set_yscale("log")
    ax_e.set(
        xlabel="Model year",
        ylabel="Year-to-year change (%)",
        xlim=(1.8, year_count + 0.2),
    )
    ax_e.legend(fontsize=6.4, loc="best")
    _panel_label(ax_e, "e")

    ax_f = fig.add_subplot(grid[3, :])
    for key, label, color in (
        ("phosphorus_drift_ppm", "Phosphorus", COLORS["phosphorus"]),
        ("nitrogen_budget_residual_ppm", "Nitrogen", COLORS["nitrogen"]),
        ("carbon_budget_residual_ppm", "Carbon", COLORS["carbon"]),
    ):
        ax_f.plot(
            model_year,
            arrays[key],
            color=color,
            linewidth=1.25,
            label=label,
        )
    ax_f.axhline(0.0, color="#C8C8C8", linewidth=0.8, zorder=0)
    ax_f.set(
        xlabel="Model year",
        ylabel="Budget residual (ppm)",
        xlim=(0.0, year_count),
    )
    ax_f.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax_f.legend(fontsize=6.5, loc="best", ncol=3)
    _panel_label(ax_f, "f")

    surface_mask = arrays["surface_mask"].astype(bool)
    ax_g = fig.add_subplot(grid[4, :])
    annual_npp_map = np.where(
        surface_mask,
        arrays["mean_vertically_integrated_npp"] * MMOL_N_M2_S_TO_G_C_M2_YR,
        np.nan,
    ).T
    vmin, vmax = _map_limits(annual_npp_map)
    image = ax_g.pcolormesh(
        arrays["xt"],
        arrays["yt"],
        annual_npp_map,
        shading="auto",
        cmap="YlGn",
        vmin=max(0.0, vmin),
        vmax=vmax,
        rasterized=True,
    )
    ax_g.set(
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
        title=f"Year {year_count} mean vertically integrated NPP",
    )
    ax_g.title.set_fontsize(7.5)
    ax_g.set_xticks((60, 180, 300))
    ax_g.set_yticks((-60, 0, 60))
    colorbar = fig.colorbar(image, ax=ax_g, pad=0.02, fraction=0.048, extend="both")
    colorbar.set_label(r"NPP (g C m$^{-2}$ yr$^{-1}$)", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.5)
    _panel_label(ax_g, "g")

    ax_h = fig.add_subplot(grid[5, 0])
    surface_no3 = np.where(surface_mask, arrays["mean_surface_no3"], np.nan).T
    vmin, vmax = _map_limits(surface_no3)
    image = ax_h.pcolormesh(
        arrays["xt"],
        arrays["yt"],
        surface_no3,
        shading="auto",
        cmap="cividis",
        vmin=max(0.0, vmin),
        vmax=vmax,
        rasterized=True,
    )
    ax_h.set(
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
        title=rf"Year {year_count} mean surface NO$_3$",
    )
    ax_h.title.set_fontsize(7.5)
    ax_h.set_xticks((60, 180, 300))
    ax_h.set_yticks((-60, 0, 60))
    colorbar = fig.colorbar(image, ax=ax_h, pad=0.02, fraction=0.048, extend="both")
    colorbar.set_label(r"NO$_3$ (mmol N m$^{-3}$)", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.5)
    _panel_label(ax_h, "h")

    ax_i = fig.add_subplot(grid[5, 1])
    surface_po4 = np.where(surface_mask, arrays["mean_surface_po4"], np.nan).T
    vmin, vmax = _map_limits(surface_po4)
    image = ax_i.pcolormesh(
        arrays["xt"],
        arrays["yt"],
        surface_po4,
        shading="auto",
        cmap="cividis",
        vmin=max(0.0, vmin),
        vmax=vmax,
        rasterized=True,
    )
    ax_i.set(
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
        title=rf"Year {year_count} mean surface PO$_4$",
    )
    ax_i.title.set_fontsize(7.5)
    ax_i.set_xticks((60, 180, 300))
    ax_i.set_yticks((-60, 0, 60))
    colorbar = fig.colorbar(image, ax=ax_i, pad=0.02, fraction=0.048, extend="both")
    colorbar.set_label(r"PO$_4$ (mmol P m$^{-3}$)", fontsize=6.8)
    colorbar.ax.tick_params(labelsize=6.5)
    _panel_label(ax_i, "i")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
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
