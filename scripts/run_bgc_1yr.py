#!/usr/bin/env python
"""Run a compact, restartable one-year Veros--MOBI experiment.

The script disables Veros' gridded diagnostics and records daily global
biogeochemical metrics itself.  A single rolling restart file protects a long
run without creating a stack of large snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86400.0
MODEL_YEAR_DAYS = 360
MMOL_C_TO_PG_C = 12.011e-18


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=MODEL_YEAR_DAYS)
    parser.add_argument("--sample-days", type=int, default=1)
    parser.add_argument("--checkpoint-days", type=int, default=30)
    parser.add_argument(
        "--restart-input",
        type=Path,
        help="Continue from a Veros restart; --days is the additional duration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bgc_1yr"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing result directory",
    )
    parser.add_argument(
        "--save-mean-surface-fields",
        action="store_true",
        help=(
            "Save timestep-mean surface nutrients, plankton, and vertically "
            "integrated NPP over this segment"
        ),
    )
    return parser.parse_args()


def _weighted_mean(field, weights):
    return float(np.sum(field * weights) / np.sum(weights))


def _collect_metrics(
    state,
    cumulative_airsea_mmol,
    cumulative_nitrogen_source_mmol,
    rates_valid=True,
):
    vs = state.variables
    settings = state.settings
    tau = int(vs.tau)

    area = np.asarray(vs.area_t[2:-2, 2:-2])
    mask = np.asarray(vs.maskT[2:-2, 2:-2, :])
    volume = (
        area[:, :, np.newaxis] * np.asarray(vs.dzt)[np.newaxis, np.newaxis, :] * mask
    )
    surface_area = area * mask[:, :, -1]

    phyto = np.asarray(vs.phytoplankton[2:-2, 2:-2, :, tau])
    zoo = np.asarray(vs.zooplankton[2:-2, 2:-2, :, tau])
    detritus = np.asarray(vs.detritus[2:-2, 2:-2, :, tau])
    po4 = np.asarray(vs.po4[2:-2, 2:-2, :, tau])
    dic = np.asarray(vs.dic[2:-2, 2:-2, :, tau])
    alkalinity = np.asarray(vs.alkalinity[2:-2, 2:-2, :, tau])
    redfield_organic_nitrogen = phyto + zoo + detritus

    if settings.enable_nitrogen:
        no3 = np.asarray(vs.no3[2:-2, 2:-2, :, tau])
        dop = np.asarray(vs.dop[2:-2, 2:-2, :, tau])
        don = np.asarray(vs.don[2:-2, 2:-2, :, tau])
        diazotrophs = np.asarray(vs.diazotrophs[2:-2, 2:-2, :, tau])
        oxygen = np.asarray(vs.oxygen[2:-2, 2:-2, :, tau])
        phosphorus = (
            po4
            + dop
            + settings.redfield_ratio_PN * redfield_organic_nitrogen
            + diazotrophs / settings.diazotroph_NP_ratio
        )
        nitrogen = no3 + don + redfield_organic_nitrogen + diazotrophs
        carbon_organic_nitrogen = redfield_organic_nitrogen + don + diazotrophs
    else:
        no3 = dop = don = diazotrophs = oxygen = np.zeros_like(po4)
        phosphorus = po4 + settings.redfield_ratio_PN * redfield_organic_nitrogen
        nitrogen = np.zeros_like(po4)
        carbon_organic_nitrogen = redfield_organic_nitrogen

    phosphorus_inventory = float(np.sum(phosphorus * volume))
    nitrogen_inventory = float(np.sum(nitrogen * volume))
    carbon_inventory = float(
        np.sum((dic + settings.redfield_ratio_CN * carbon_organic_nitrogen) * volume)
    )

    if rates_valid:
        npp = np.asarray(vs.net_primary_production[2:-2, 2:-2, :])
        cflux = np.asarray(vs.cflux[2:-2, 2:-2])
        npp_mmol_c_per_second = float(np.sum(npp * volume) * settings.redfield_ratio_CN)
        airsea_mmol_c_per_second = float(np.sum(cflux * surface_area))
        annualization = MODEL_YEAR_DAYS * SECONDS_PER_DAY * MMOL_C_TO_PG_C
        npp_pg_c_per_year = npp_mmol_c_per_second * annualization
        airsea_pg_c_per_year = airsea_mmol_c_per_second * annualization
        if settings.enable_nitrogen:
            nitrogen_fixation_mmol_n_per_second = float(
                np.sum(np.asarray(vs.nitrogen_fixation[2:-2, 2:-2, :]) * volume)
            )
            water_column_denitrification_mmol_n_per_second = float(
                np.sum(
                    np.asarray(vs.water_column_denitrification[2:-2, 2:-2, :]) * volume
                )
            )
            benthic_denitrification_mmol_n_per_second = float(
                np.sum(np.asarray(vs.benthic_denitrification[2:-2, 2:-2, :]) * volume)
            )
        else:
            nitrogen_fixation_mmol_n_per_second = 0.0
            water_column_denitrification_mmol_n_per_second = 0.0
            benthic_denitrification_mmol_n_per_second = 0.0
    else:
        airsea_mmol_c_per_second = 0.0
        npp_pg_c_per_year = np.nan
        airsea_pg_c_per_year = np.nan
        nitrogen_fixation_mmol_n_per_second = np.nan
        water_column_denitrification_mmol_n_per_second = np.nan
        benthic_denitrification_mmol_n_per_second = np.nan

    values = {
        "day": float(vs.time / SECONDS_PER_DAY),
        "iteration": int(vs.itt),
        "phytoplankton_mean": _weighted_mean(phyto, volume),
        "zooplankton_mean": _weighted_mean(zoo, volume),
        "detritus_mean": _weighted_mean(detritus, volume),
        "po4_mean": _weighted_mean(po4, volume),
        "no3_mean": _weighted_mean(no3, volume),
        "dop_mean": _weighted_mean(dop, volume),
        "don_mean": _weighted_mean(don, volume),
        "diazotrophs_mean": _weighted_mean(diazotrophs, volume),
        "oxygen_mean": _weighted_mean(oxygen, volume),
        "dic_mean": _weighted_mean(dic, volume),
        "alkalinity_mean": _weighted_mean(alkalinity, volume),
        "npp_pg_c_per_year": npp_pg_c_per_year,
        "airsea_co2_pg_c_per_year": airsea_pg_c_per_year,
        "airsea_co2_mmol_c_per_second": airsea_mmol_c_per_second,
        "nitrogen_fixation_mmol_n_per_second": (nitrogen_fixation_mmol_n_per_second),
        "water_column_denitrification_mmol_n_per_second": (
            water_column_denitrification_mmol_n_per_second
        ),
        "benthic_denitrification_mmol_n_per_second": (
            benthic_denitrification_mmol_n_per_second
        ),
        "phosphorus_inventory_mmol": phosphorus_inventory,
        "nitrogen_inventory_mmol": nitrogen_inventory,
        "carbon_inventory_mmol": carbon_inventory,
        "cumulative_airsea_co2_mmol": float(cumulative_airsea_mmol),
        "cumulative_nitrogen_source_mmol": float(cumulative_nitrogen_source_mmol),
    }

    for name in (
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
        "phosphorus_inventory_mmol",
        "nitrogen_inventory_mmol",
        "carbon_inventory_mmol",
    ):
        if not np.isfinite(values[name]):
            raise RuntimeError(f"non-finite diagnostic {name} at day {values['day']}")

    return values


def _zonal_mean(field, area, mask):
    weights = area[:, :, np.newaxis] * mask
    numerator = np.sum(field * weights, axis=0)
    denominator = np.sum(weights, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _save_final_fields(state, metrics, output_file, mean_surface_fields=None):
    vs = state.variables
    tau = int(vs.tau)
    area = np.asarray(vs.area_t[2:-2, 2:-2])
    mask = np.asarray(vs.maskT[2:-2, 2:-2, :], dtype=bool)

    phyto = np.asarray(vs.phytoplankton[2:-2, 2:-2, :, tau])
    po4 = np.asarray(vs.po4[2:-2, 2:-2, :, tau])
    dic = np.asarray(vs.dic[2:-2, 2:-2, :, tau])
    no3 = (
        np.asarray(vs.no3[2:-2, 2:-2, :, tau])
        if state.settings.enable_nitrogen
        else np.zeros_like(po4)
    )

    payload = {key: np.asarray([row[key] for row in metrics]) for key in metrics[0]}
    payload.update(
        xt=np.asarray(vs.xt[2:-2]),
        yt=np.asarray(vs.yt[2:-2]),
        zt=np.asarray(vs.zt),
        surface_mask=mask[:, :, -1],
        surface_phytoplankton=phyto[:, :, -1],
        surface_po4=po4[:, :, -1],
        surface_no3=no3[:, :, -1],
        surface_pco2=np.asarray(vs.pCO2[2:-2, 2:-2]),
        surface_cflux=np.asarray(vs.cflux[2:-2, 2:-2]),
        surface_temperature=np.asarray(vs.temp[2:-2, 2:-2, -1, tau]),
        zonal_phytoplankton=_zonal_mean(phyto, area, mask),
        zonal_po4=_zonal_mean(po4, area, mask),
        zonal_no3=_zonal_mean(no3, area, mask),
        zonal_dic=_zonal_mean(dic, area, mask),
    )
    if mean_surface_fields is not None:
        payload.update(mean_surface_fields)
    np.savez_compressed(output_file, **payload)


def main():
    args = parse_args()
    if args.days < 1 or args.sample_days < 1 or args.checkpoint_days < 0:
        raise SystemExit(
            "days and sample-days must be positive; checkpoint-days cannot be negative"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"result directory is not empty: {output_dir} (use --overwrite)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Veros only after command-line parsing so VEROS_BACKEND and related
    # environment variables can select the runtime before its settings lock.
    import veros
    from veros import restart, runtime_settings, veros_routine

    from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import (
        GlobalFourDegreeBGC,
    )

    class OneYearBGC(GlobalFourDegreeBGC):
        def __init__(self):
            self._run_days = args.days
            self._checkpoint_days = args.checkpoint_days
            self._result_dir = output_dir
            super().__init__()

        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            settings = state.settings
            settings.identifier = "bgc_global_4deg_1yr"
            settings.runlen = self._run_days * SECONDS_PER_DAY
            settings.restart_frequency = self._checkpoint_days * SECONDS_PER_DAY
            settings.restart_output_filename = str(
                self._result_dir / "checkpoint.restart.h5"
            )
            if args.restart_input is not None:
                settings.restart_input_filename = str(args.restart_input.resolve())

        @veros_routine
        def set_diagnostics(self, state):
            super().set_diagnostics(state)
            for diagnostic in state.diagnostics.values():
                diagnostic.output_frequency = None
                diagnostic.sampling_frequency = None

    simulation = OneYearBGC()
    setup_start = time.perf_counter()
    simulation.setup()
    setup_seconds = time.perf_counter() - setup_start

    state = simulation.state
    dt = float(state.settings.dt_tracer)
    start_day = float(state.variables.time / SECONDS_PER_DAY)
    target_day = start_day + args.days
    total_steps = int(round(args.days * SECONDS_PER_DAY / dt))
    sample_steps = max(1, int(round(args.sample_days * SECONDS_PER_DAY / dt)))

    metrics_file = output_dir / "daily_metrics.csv"
    metrics = []
    cumulative_airsea_mmol = 0.0
    cumulative_nitrogen_source_mmol = 0.0
    mean_surface_sums = None
    mean_surface_sample_count = 0
    if args.save_mean_surface_fields:
        tau = int(state.variables.tau)
        surface_shape = state.variables.po4[2:-2, 2:-2, -1, tau].shape
        mean_surface_sums = {
            "mean_surface_po4": np.zeros(surface_shape, dtype=float),
            "mean_surface_phytoplankton": np.zeros(surface_shape, dtype=float),
            "mean_surface_no3": np.zeros(surface_shape, dtype=float),
            "mean_surface_diazotrophs": np.zeros(surface_shape, dtype=float),
            "mean_vertically_integrated_npp": np.zeros(surface_shape, dtype=float),
        }
    initial = _collect_metrics(
        state,
        cumulative_airsea_mmol,
        cumulative_nitrogen_source_mmol,
        rates_valid=False,
    )
    initial_phosphorus = initial["phosphorus_inventory_mmol"]
    initial_nitrogen = initial["nitrogen_inventory_mmol"]
    initial_carbon = initial["carbon_inventory_mmol"]

    fieldnames = list(initial)
    fieldnames.extend(
        (
            "phosphorus_drift_ppm",
            "nitrogen_budget_residual_ppm",
            "carbon_budget_residual_ppm",
        )
    )
    initial["phosphorus_drift_ppm"] = 0.0
    initial["nitrogen_budget_residual_ppm"] = 0.0
    initial["carbon_budget_residual_ppm"] = 0.0
    metrics.append(initial)

    run_start = time.perf_counter()
    with metrics_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(initial)
        stream.flush()

        for step in range(1, total_steps + 1):
            simulation.step(state)

            if mean_surface_sums is not None:
                tau = int(state.variables.tau)
                mean_surface_sums["mean_surface_po4"] += np.asarray(
                    state.variables.po4[2:-2, 2:-2, -1, tau]
                )
                mean_surface_sums["mean_surface_phytoplankton"] += np.asarray(
                    state.variables.phytoplankton[2:-2, 2:-2, -1, tau]
                )
                if state.settings.enable_nitrogen:
                    mean_surface_sums["mean_surface_no3"] += np.asarray(
                        state.variables.no3[2:-2, 2:-2, -1, tau]
                    )
                    mean_surface_sums["mean_surface_diazotrophs"] += np.asarray(
                        state.variables.diazotrophs[2:-2, 2:-2, -1, tau]
                    )
                mean_surface_sums["mean_vertically_integrated_npp"] += np.sum(
                    np.asarray(state.variables.net_primary_production[2:-2, 2:-2, :])
                    * np.asarray(state.variables.dzt)[np.newaxis, np.newaxis, :],
                    axis=2,
                )
                mean_surface_sample_count += 1

            current_flux = float(
                np.sum(
                    np.asarray(state.variables.cflux[2:-2, 2:-2])
                    * np.asarray(state.variables.area_t[2:-2, 2:-2])
                    * np.asarray(state.variables.maskT[2:-2, 2:-2, -1])
                )
            )
            cumulative_airsea_mmol += current_flux * dt
            if state.settings.enable_nitrogen:
                nitrogen_source = (
                    np.asarray(state.variables.nitrogen_fixation[2:-2, 2:-2, :])
                    - np.asarray(
                        state.variables.water_column_denitrification[2:-2, 2:-2, :]
                    )
                    - np.asarray(state.variables.benthic_denitrification[2:-2, 2:-2, :])
                )
                area = np.asarray(state.variables.area_t[2:-2, 2:-2])
                mask = np.asarray(state.variables.maskT[2:-2, 2:-2, :])
                volume = (
                    area[:, :, np.newaxis]
                    * np.asarray(state.variables.dzt)[np.newaxis, np.newaxis, :]
                    * mask
                )
                cumulative_nitrogen_source_mmol += (
                    float(np.sum(nitrogen_source * volume)) * dt
                )

            if step % sample_steps == 0 or step == total_steps:
                row = _collect_metrics(
                    state,
                    cumulative_airsea_mmol,
                    cumulative_nitrogen_source_mmol,
                )
                row["phosphorus_drift_ppm"] = (
                    (row["phosphorus_inventory_mmol"] - initial_phosphorus)
                    / initial_phosphorus
                    * 1e6
                )
                row["carbon_budget_residual_ppm"] = (
                    (
                        row["carbon_inventory_mmol"]
                        - initial_carbon
                        - cumulative_airsea_mmol
                    )
                    / initial_carbon
                    * 1e6
                )
                if state.settings.enable_nitrogen:
                    row["nitrogen_budget_residual_ppm"] = (
                        (
                            row["nitrogen_inventory_mmol"]
                            - initial_nitrogen
                            - cumulative_nitrogen_source_mmol
                        )
                        / initial_nitrogen
                        * 1e6
                    )
                else:
                    row["nitrogen_budget_residual_ppm"] = 0.0
                metrics.append(row)
                writer.writerow(row)
                stream.flush()

            if step == 1 or step % 10 == 0 or step == total_steps:
                elapsed = time.perf_counter() - run_start
                rate = step / max(elapsed, 1e-12)
                remaining_minutes = (total_steps - step) / max(rate, 1e-12) / 60.0
                print(
                    f"day {state.variables.time / SECONDS_PER_DAY:6.1f}/{target_day:g} "
                    f"({rate:.2f} steps/s, ETA {remaining_minutes:.1f} min)",
                    flush=True,
                )

    run_seconds = time.perf_counter() - run_start
    restart.write_restart(state, force=True)
    if mean_surface_sums is None:
        mean_surface_fields = None
    else:
        mean_surface_fields = {
            key: values / mean_surface_sample_count
            for key, values in mean_surface_sums.items()
        }
        mean_surface_fields["mean_surface_sample_count"] = np.asarray(
            mean_surface_sample_count
        )
    _save_final_fields(
        state,
        metrics,
        output_dir / "bgc_1yr_diagnostics.npz",
        mean_surface_fields,
    )

    metadata = {
        "veros_version": veros.__version__,
        "backend": runtime_settings.backend,
        "float_type": runtime_settings.float_type,
        "days": args.days,
        "start_day": start_day,
        "end_day": float(state.variables.time / SECONDS_PER_DAY),
        "steps": total_steps,
        "sample_days": args.sample_days,
        "setup_seconds": setup_seconds,
        "run_seconds": run_seconds,
        "final_phosphorus_drift_ppm": metrics[-1]["phosphorus_drift_ppm"],
        "final_nitrogen_budget_residual_ppm": metrics[-1][
            "nitrogen_budget_residual_ppm"
        ],
        "final_carbon_budget_residual_ppm": metrics[-1]["carbon_budget_residual_ppm"],
        "enable_bgc_superbee_advection": state.settings.enable_bgc_superbee_advection,
        "enable_bgc_conservative_clipping": (
            state.settings.enable_bgc_conservative_clipping
        ),
        "enable_nitrogen": state.settings.enable_nitrogen,
        "biology_parameters": {
            "remineralization_rate_detritus_per_day": (
                state.settings.remineralization_rate_detritus * SECONDS_PER_DAY
            ),
            "bbio": state.settings.bbio,
            "cbio": state.settings.cbio,
            "maximum_growth_rate_phyto_per_day": (
                state.settings.maximum_growth_rate_phyto * SECONDS_PER_DAY
            ),
            "maximum_grazing_rate_per_day": (
                state.settings.maximum_grazing_rate * SECONDS_PER_DAY
            ),
            "fast_recycling_rate_phytoplankton_per_day": (
                state.settings.fast_recycling_rate_phytoplankton * SECONDS_PER_DAY
            ),
            "specific_mortality_phytoplankton_per_day": (
                state.settings.specific_mortality_phytoplankton * SECONDS_PER_DAY
            ),
            "quadric_mortality_zooplankton_per_day": (
                state.settings.quadric_mortality_zooplankton * SECONDS_PER_DAY
            ),
            "zooplankton_growth_efficiency": (
                state.settings.zooplankton_growth_efficiency
            ),
            "assimilation_efficiency": state.settings.assimilation_efficiency,
            "wd0_m_per_day": state.settings.wd0 * SECONDS_PER_DAY,
            "mwz_m": state.settings.mwz,
            "mw_per_day": state.settings.mw * SECONDS_PER_DAY,
            "dcaco3_m": state.settings.dcaco3,
            "fast_recycling_rate_diazotrophs_per_day": (
                state.settings.fast_recycling_rate_diazotrophs * SECONDS_PER_DAY
            ),
            "quadric_mortality_diazotrophs_per_day": (
                state.settings.quadric_mortality_diazotrophs * SECONDS_PER_DAY
            ),
            "remineralization_rate_don_per_day": (
                state.settings.remineralization_rate_don * SECONDS_PER_DAY
            ),
            "remineralization_rate_dop_per_day": (
                state.settings.remineralization_rate_dop * SECONDS_PER_DAY
            ),
            "diazotroph_growth_rate_factor": (
                state.settings.diazotroph_growth_rate_factor
            ),
            "diazotroph_NP_ratio": state.settings.diazotroph_NP_ratio,
        },
        "restart_input": (
            str(args.restart_input.resolve())
            if args.restart_input is not None
            else None
        ),
        "mean_surface_sample_count": mean_surface_sample_count,
        "mean_surface_sample_interval_days": (
            dt / SECONDS_PER_DAY if mean_surface_sample_count else None
        ),
        "mean_surface_first_sample_day": (
            start_day + dt / SECONDS_PER_DAY if mean_surface_sample_count else None
        ),
        "mean_surface_last_sample_day": (
            float(state.variables.time / SECONDS_PER_DAY)
            if mean_surface_sample_count
            else None
        ),
        "environment_backend": os.environ.get("VEROS_BACKEND", "numpy"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
