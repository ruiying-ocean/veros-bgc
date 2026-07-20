#!/usr/bin/env python

"""Global four-degree Veros setup with the MOBI NPZD plugin."""

import os

import h5netcdf
import veros.tools
from veros import veros_routine
from veros.core.operators import at, update
from veros.core.operators import numpy as npx
from veros.setups.global_4deg import GlobalFourDegreeSetup
from veros.variables import Variable

import veros_bgc

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BGC_DATA_FILES = veros.tools.get_assets(
    "global_4deg",
    os.path.join(BASE_PATH, "assets.json"),
)


class GlobalFourDegreeBGC(GlobalFourDegreeSetup):
    """Global 4-degree model with 15 vertical levels and MOBI NPZD.

    The physical configuration is inherited from Veros' maintained
    :class:`GlobalFourDegreeSetup`; this class only adds shortwave forcing,
    biogeochemical initial conditions, and diagnostics.
    """

    __veros_plugins__ = (veros_bgc,)

    @veros_routine
    def set_parameter(self, state):
        super().set_parameter(state)

        settings = state.settings
        settings.identifier = "bgc_global_4deg"
        settings.description = "Global four-degree Veros setup with MOBI NPZD"
        settings.enable_npzd = True
        settings.enable_carbon = True
        settings.enable_nitrogen = True
        settings.dt_bio = settings.dt_tracer / 4.0

        # ``mobi_init`` selects these preferences for the nitrogen branch
        # without prognostic coccolithophores. All kinetic and stoichiometric
        # parameters otherwise retain the Fortran defaults from settings.py.
        settings.zprefP = 0.30
        settings.zprefZ = 0.30
        settings.zprefDet = 0.30
        settings.zprefDiaz = 0.10

        # The parent setup creates the nmonths dimension.
        state.var_meta.update(
            swr_clim=Variable(
                "Monthly shortwave radiation",
                ("xt", "yt", "nmonths"),
                "W / m2",
                "Monthly climatology of downward shortwave radiation",
                time_dependent=False,
            )
        )

    @veros_routine(
        dist_safe=False,
        local_variables=[
            # Variables accessed by GlobalFourDegreeSetup.set_initial_conditions
            "taux",
            "tauy",
            "qnec",
            "qnet",
            "sss_clim",
            "sst_clim",
            "temp",
            "salt",
            "area_t",
            "maskT",
            "zw",
            "zt",
            "forc_iw_bottom",
            "forc_iw_surface",
            # BGC additions
            "swr_clim",
            "phytoplankton",
            "zooplankton",
            "detritus",
            "po4",
            "no3",
            "dop",
            "don",
            "diazotrophs",
            "oxygen",
            "dic",
            "alkalinity",
            "atmospheric_co2",
            "hSWS",
        ],
    )
    def set_initial_conditions(self, state):
        super().set_initial_conditions(state)

        vs = state.variables
        settings = state.settings

        with h5netcdf.File(BGC_DATA_FILES["corev2"], "r") as infile:
            shortwave = npx.asarray(infile.variables["SWDN_MOD"]).T
        vs.swr_clim = update(vs.swr_clim, at[2:-2, 2:-2, :], shortwave)

        phyto = 0.14 * npx.exp(vs.zw / 100.0) * vs.maskT
        zoo = 0.014 * npx.exp(vs.zw / 100.0) * vs.maskT
        vs.phytoplankton = phyto[..., npx.newaxis] * npx.ones_like(vs.phytoplankton)
        vs.zooplankton = zoo[..., npx.newaxis] * npx.ones_like(vs.zooplankton)
        vs.detritus = 1e-4 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.detritus)
        vs.po4 = 2.2 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.po4)

        if settings.enable_nitrogen:
            depth = npx.maximum(-vs.zt, 0.0)
            no3_profile = 0.1 + 30.0 * (1.0 - npx.exp(-depth / 500.0))
            oxygen_profile = 180.0 + 70.0 * npx.exp(-depth / 500.0)
            vs.no3 = (
                no3_profile[npx.newaxis, npx.newaxis, :, npx.newaxis]
                * vs.maskT[..., npx.newaxis]
                * npx.ones_like(vs.no3)
            )
            vs.dop = 0.1 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.dop)
            vs.don = 5.0 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.don)
            diazotrophs = 1e-4 * npx.exp(vs.zw / 100.0) * vs.maskT
            vs.diazotrophs = diazotrophs[..., npx.newaxis] * npx.ones_like(
                vs.diazotrophs
            )
            vs.oxygen = (
                oxygen_profile[npx.newaxis, npx.newaxis, :, npx.newaxis]
                * vs.maskT[..., npx.newaxis]
                * npx.ones_like(vs.oxygen)
            )

        if settings.enable_carbon:
            vs.dic = 2300.0 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.dic)
            vs.alkalinity = (
                2400.0 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.alkalinity)
            )
            vs.atmospheric_co2 = 280.0 * npx.ones_like(vs.atmospheric_co2)
            vs.hSWS = 5e-7 * npx.ones_like(vs.hSWS)

    @veros_routine
    def set_forcing(self, state):
        super().set_forcing(state)

        vs = state.variables
        year_in_seconds = 360.0 * 86400.0
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(
            vs.time,
            year_in_seconds,
            year_in_seconds / 12.0,
            12,
        )
        vs.swr = (f1 * vs.swr_clim[:, :, n1] + f2 * vs.swr_clim[:, :, n2]) * vs.maskT[
            :, :, -1
        ]

    @veros_routine
    def set_diagnostics(self, state):
        super().set_diagnostics(state)

        settings = state.settings
        diagnostics = state.diagnostics
        bgc_variables = [
            "phytoplankton",
            "zooplankton",
            "detritus",
            "po4",
            "net_primary_production",
            "detritus_remineralization",
            "detritus_export",
        ]
        if settings.enable_nitrogen:
            bgc_variables.extend(
                [
                    "no3",
                    "dop",
                    "don",
                    "diazotrophs",
                    "oxygen",
                    "diazotroph_primary_production",
                    "nitrogen_fixation",
                    "water_column_denitrification",
                    "benthic_denitrification",
                ]
            )
        if settings.enable_carbon:
            bgc_variables.extend(
                [
                    "dic",
                    "alkalinity",
                    "cflux",
                    "pCO2",
                    "dpCO2",
                    "atmospheric_co2",
                ]
            )

        diagnostics["averages"].output_variables.extend(bgc_variables)
        diagnostics["npzd"].output_frequency = 10.0 * 86400.0


def run(**kwargs):
    """Run the setup directly (``veros run`` is preferred for CLI use)."""

    simulation = GlobalFourDegreeBGC(**kwargs)
    simulation.setup()
    simulation.run()


if __name__ == "__main__":
    run()
