#!/usr/bin/env python
"""Plot the compact diagnostics from ``run_bgc_1yr.py``."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# Mandatory publication/export settings: sans-serif type and editable SVG text.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


COLORS = {
    "phyto": "#0F4D92",
    "zoo": "#B64342",
    "detritus": "#8A6D3B",
    "npp": "#2E8B57",
    "co2": "#9A4D8E",
    "phosphorus": "#3775BA",
    "carbon": "#E28E2C",
    "neutral": "#767676",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/bgc_1yr/bgc_1yr_diagnostics.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/bgc_1yr_summary"),
        help="Output path without a suffix",
    )
    return parser.parse_args()


def _panel_label(ax, label):
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        ha="left",
        va="bottom",
    )


def _smooth(values, window=15):
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return values
    valid = np.isfinite(values)
    numerator = np.convolve(np.where(valid, values, 0.0), np.ones(window), mode="same")
    denominator = np.convolve(valid.astype(float), np.ones(window), mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def _map_panel(ax, xt, yt, values, *, cmap, label, robust=False, norm=None):
    field = np.asarray(values, dtype=float).T
    if robust:
        finite = field[np.isfinite(field)]
        vmin, vmax = np.nanpercentile(finite, (2.0, 98.0))
    else:
        vmin = vmax = None
    image = ax.pcolormesh(
        xt,
        yt,
        field,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        rasterized=True,
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xticks((60, 180, 300))
    ax.set_yticks((-60, 0, 60))
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.048)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    return image


def main():
    args = parse_args()
    with np.load(args.input) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    day = arrays["day"]
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.12))
    fig.subplots_adjust(
        left=0.09, right=0.96, bottom=0.07, top=0.97, hspace=0.52, wspace=0.52
    )

    ax_a = fig.add_subplot(grid[0, 0])
    for key, label, color in (
        ("phytoplankton_mean", "Phytoplankton", COLORS["phyto"]),
        ("zooplankton_mean", "Zooplankton", COLORS["zoo"]),
        ("detritus_mean", "Detritus", COLORS["detritus"]),
    ):
        ax_a.plot(day, arrays[key], color=color, linewidth=1.6, label=label)
    ax_a.set_yscale("log")
    ax_a.set(
        xlabel="Model day",
        ylabel=r"Global volume-weighted mean (mmol N m$^{-3}$)",
    )
    ax_a.set_xlim(day.min(), day.max())
    ax_a.legend(ncol=1, loc="best", fontsize=7)
    _panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[0, 1])
    npp = arrays["npp_pg_c_per_year"]
    cflux = arrays["airsea_co2_pg_c_per_year"]
    ax_b.plot(day, npp, color=COLORS["npp"], alpha=0.22, linewidth=0.7)
    npp_line = ax_b.plot(
        day,
        _smooth(npp),
        color=COLORS["npp"],
        linewidth=1.8,
        label="NPP",
    )[0]
    ax_b.set(xlabel="Model day", ylabel=r"NPP (Pg C yr$^{-1}$)")
    ax_b.set_xlim(day.min(), day.max())
    ax_b.tick_params(axis="y", colors=COLORS["npp"])
    ax_b.spines["left"].set_color(COLORS["npp"])
    ax_b2 = ax_b.twinx()
    ax_b2.spines["top"].set_visible(False)
    ax_b2.plot(day, cflux, color=COLORS["co2"], alpha=0.20, linewidth=0.7)
    flux_line = ax_b2.plot(
        day,
        _smooth(cflux),
        color=COLORS["co2"],
        linewidth=1.6,
        label=r"Air–sea CO$_2$",
    )[0]
    ax_b2.axhline(0.0, color="#C8C8C8", linewidth=0.8, zorder=0)
    ax_b2.set_ylabel(r"CO$_2$ flux (Pg C yr$^{-1}$)", color=COLORS["co2"])
    ax_b2.tick_params(axis="y", colors=COLORS["co2"])
    ax_b.legend(
        [npp_line, flux_line],
        ["NPP", r"Air–sea CO$_2$"],
        loc="best",
        fontsize=7,
    )
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 0])
    ax_c.plot(
        day,
        arrays["phosphorus_drift_ppm"],
        color=COLORS["phosphorus"],
        linewidth=1.5,
        label="Phosphorus",
    )
    ax_c.plot(
        day,
        arrays["carbon_budget_residual_ppm"],
        color=COLORS["carbon"],
        linewidth=1.5,
        label="Carbon residual",
    )
    ax_c.axhline(0.0, color="#C8C8C8", linewidth=0.8, zorder=0)
    ax_c.set(xlabel="Model day", ylabel="Budget residual (ppm)")
    ax_c.set_xlim(day.min(), day.max())
    ax_c.legend(fontsize=7)
    _panel_label(ax_c, "c")

    ax_d = fig.add_subplot(grid[1, 1])
    surface_mask = arrays["surface_mask"].astype(bool)
    surface_phyto = np.where(surface_mask, arrays["surface_phytoplankton"], np.nan)
    _map_panel(
        ax_d,
        arrays["xt"],
        arrays["yt"],
        surface_phyto,
        cmap="YlGnBu",
        label=r"Phytoplankton (mmol N m$^{-3}$)",
        robust=True,
    )
    ax_d.set_title("End-of-year surface phytoplankton", fontsize=8, pad=3)
    _panel_label(ax_d, "d")

    ax_e = fig.add_subplot(grid[2, 0])
    surface_pco2 = np.where(surface_mask, arrays["surface_pco2"], np.nan)
    finite_pco2 = surface_pco2[np.isfinite(surface_pco2)]
    pco2_low, pco2_high = np.nanpercentile(finite_pco2, (2.0, 98.0))
    pco2_norm = TwoSlopeNorm(vmin=pco2_low, vcenter=280.0, vmax=max(pco2_high, 280.1))
    _map_panel(
        ax_e,
        arrays["xt"],
        arrays["yt"],
        surface_pco2,
        cmap="RdBu_r",
        label=r"pCO$_2$ ($\mu$atm)",
        norm=pco2_norm,
    )
    ax_e.set_title(r"End-of-year surface pCO$_2$", fontsize=8, pad=3)
    _panel_label(ax_e, "e")

    ax_f = fig.add_subplot(grid[2, 1])
    zonal_po4 = arrays["zonal_po4"].T
    image = ax_f.pcolormesh(
        arrays["yt"],
        -arrays["zt"],
        zonal_po4,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    ax_f.set(xlabel="Latitude (°N)", ylabel="Depth (m)")
    ax_f.set_yscale("symlog", linthresh=100.0)
    ax_f.invert_yaxis()
    ax_f.set_yticks((0, 100, 500, 1000, 3000, 5000))
    ax_f.set_yticklabels(("0", "100", "500", "1000", "3000", "5000"))
    colorbar = fig.colorbar(image, ax=ax_f, pad=0.02, fraction=0.048)
    colorbar.set_label(r"PO$_4$ (mmol P m$^{-3}$)", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)
    ax_f.set_title("End-of-year zonal-mean phosphate", fontsize=8, pad=3)
    _panel_label(ax_f, "f")

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
