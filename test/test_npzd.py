import unittest

import numpy as np
from veros.core import numerics
from veros.core.operators import at, update
from veros.plugins import load_plugin
from veros.state import get_default_state

import veros_bgc
from veros_bgc.core.atmospherefluxes import carbon_flux, carbonate_system
from veros_bgc.core.npzd import (
    enforce_conservative_tracer_floor,
    integrate_npzd,
    mobi_biology,
    setup_npzd,
)
from veros_bgc.setup.bgc_global_4deg.bgc_global_four_degree import (
    GlobalFourDegreeBGC,
)


def make_state(*, carbon=True, nitrogen=False, implicit_calcite=True):
    state = get_default_state(load_plugin(veros_bgc))
    with state.settings.unlock():
        settings = state.settings
        settings.nx = 4
        settings.ny = 4
        settings.nz = 3
        settings.dt_mom = 1800.0
        settings.dt_tracer = 86400.0
        settings.enable_npzd = True
        settings.enable_carbon = carbon
        settings.enable_nitrogen = nitrogen
        settings.enable_implicit_calcite = implicit_calcite
        settings.enable_streamfunction = False
        settings.coord_degree = True
        settings.y_origin = -6.0
        if nitrogen:
            settings.zprefP = 0.30
            settings.zprefZ = 0.30
            settings.zprefDet = 0.30
            settings.zprefDiaz = 0.10

    state.initialize_variables()
    vs = state.variables
    with vs.unlock():
        vs.dxt = np.full_like(vs.dxt, 4.0)
        vs.dyt = np.full_like(vs.dyt, 4.0)
        vs.dzt = np.array([500.0, 200.0, 50.0])

    numerics.calc_grid(state)
    with vs.unlock():
        vs.kbot = np.zeros_like(vs.kbot)
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], 1)
    numerics.calc_topo(state)

    with vs.unlock():
        tracer_mask = vs.maskT[..., np.newaxis]
        vs.temp = 10.0 * tracer_mask * np.ones_like(vs.temp)
        vs.salt = 35.0 * tracer_mask * np.ones_like(vs.salt)
        vs.phytoplankton = 0.14 * tracer_mask * np.ones_like(vs.phytoplankton)
        vs.zooplankton = 0.014 * tracer_mask * np.ones_like(vs.zooplankton)
        vs.detritus = 0.01 * tracer_mask * np.ones_like(vs.detritus)
        vs.po4 = 2.2 * tracer_mask * np.ones_like(vs.po4)
        vs.swr = 150.0 * vs.maskT[..., -1]

        if nitrogen:
            vs.no3 = 12.0 * tracer_mask * np.ones_like(vs.no3)
            vs.dop = 0.1 * tracer_mask * np.ones_like(vs.dop)
            vs.don = 5.0 * tracer_mask * np.ones_like(vs.don)
            vs.diazotrophs = 0.01 * tracer_mask * np.ones_like(vs.diazotrophs)
            vs.oxygen = 200.0 * tracer_mask * np.ones_like(vs.oxygen)

        if carbon:
            vs.dic = 2300.0 * tracer_mask * np.ones_like(vs.dic)
            vs.alkalinity = 2400.0 * tracer_mask * np.ones_like(vs.alkalinity)
            vs.atmospheric_co2 = 280.0 * np.ones_like(vs.atmospheric_co2)
            vs.hSWS = 5e-7 * np.ones_like(vs.hSWS)

    setup_npzd(state)
    return state


class NPZDTest(unittest.TestCase):
    def test_plugin_contract_and_conditional_variables(self):
        plugin = load_plugin(veros_bgc)
        self.assertEqual(plugin.name, "biogeochemistry")
        self.assertIn("dt_bio", plugin.settings)
        self.assertIn("phytoplankton", plugin.variables)

        state = make_state(carbon=False)
        self.assertNotIn("dic", state.variables.fields())
        self.assertNotIn("no3", state.variables.fields())
        self.assertIn("phytoplankton", state.variables.fields())

        nitrogen_state = make_state(carbon=False, nitrogen=True)
        self.assertIn("no3", nitrogen_state.variables.fields())
        self.assertIn("oxygen", nitrogen_state.variables.fields())

    def test_defaults_match_fortran_mobi_init(self):
        plugin = load_plugin(veros_bgc)
        seconds_per_day = 86400.0

        expected_rates = {
            "maximum_growth_rate_phyto": 0.6,
            "maximum_grazing_rate": 0.38,
            "remineralization_rate_detritus": 0.07,
            "fast_recycling_rate_phytoplankton": 0.015,
            "specific_mortality_phytoplankton": 0.03,
            "quadric_mortality_zooplankton": 0.06,
            "fast_recycling_rate_diazotrophs": 0.001,
            "quadric_mortality_diazotrophs": 0.0001,
            "remineralization_rate_don": 2.33e-5,
            "remineralization_rate_dop": 7.0e-5,
        }
        for name, expected_per_day in expected_rates.items():
            self.assertAlmostEqual(
                plugin.settings[name].default * seconds_per_day,
                expected_per_day,
            )

        self.assertEqual(plugin.settings["bbio"].default, 1.066)
        self.assertEqual(plugin.settings["assimilation_efficiency"].default, 0.70)
        self.assertEqual(plugin.settings["diazotroph_NP_ratio"].default, 28.0)
        self.assertEqual(plugin.settings["diazotroph_growth_rate_factor"].default, 0.08)
        self.assertEqual(plugin.settings["wd0"].default * seconds_per_day, 16.0)
        self.assertEqual(plugin.settings["dcaco3"].default, 6500.0)

    def test_global_setup_uses_nitrogen_and_fortran_defaults(self):
        simulation = GlobalFourDegreeBGC()
        state = simulation.state
        with state.settings.unlock():
            simulation.set_parameter(state)

        self.assertTrue(state.settings.enable_nitrogen)
        self.assertAlmostEqual(state.settings.maximum_growth_rate_phyto * 86400.0, 0.6)
        self.assertEqual(state.settings.bbio, 1.066)
        self.assertAlmostEqual(state.settings.maximum_grazing_rate * 86400.0, 0.38)
        self.assertAlmostEqual(state.settings.wd0 * 86400.0, 16.0)

    def test_bottom_mask_and_calcite_profile(self):
        state = make_state(carbon=True)
        vs = state.variables
        self.assertEqual(int(np.sum(vs.bottom_mask[2:-2, 2:-2])), 16)

        profile_integral = np.sum(
            vs.rcak[2:-2, 2:-2] * vs.dzt[np.newaxis, np.newaxis, :], axis=2
        )
        np.testing.assert_allclose(profile_integral, 1.0, rtol=0.0, atol=1e-14)

    def test_biology_conserves_phosphorus(self):
        state = make_state(carbon=False)
        vs = state.variables
        settings = state.settings
        zeros = np.zeros_like(vs.swr)
        result = mobi_biology(state, zeros)

        initial = vs.po4[..., vs.tau] + settings.redfield_ratio_PN * (
            vs.phytoplankton[..., vs.tau]
            + vs.zooplankton[..., vs.tau]
            + vs.detritus[..., vs.tau]
        )
        final = (
            vs.po4[..., vs.tau]
            + result.po4_change
            + settings.redfield_ratio_PN
            * (
                vs.phytoplankton[..., vs.tau]
                + result.phytoplankton_change
                + vs.zooplankton[..., vs.tau]
                + result.zooplankton_change
                + vs.detritus[..., vs.tau]
                + result.detritus_change
            )
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT
        np.testing.assert_allclose(
            np.sum(final[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            np.sum(initial[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            rtol=2e-14,
            atol=1e-10,
        )

    def test_nitrogen_branch_conserves_phosphorus(self):
        state = make_state(carbon=False, nitrogen=True)
        vs = state.variables
        settings = state.settings
        result = mobi_biology(state, np.zeros_like(vs.swr))

        def phosphorus(po4, dop, phyto, zoo, detritus, diazotrophs):
            return (
                po4
                + dop
                + settings.redfield_ratio_PN * (phyto + zoo + detritus)
                + diazotrophs / settings.diazotroph_NP_ratio
            )

        initial = phosphorus(
            vs.po4[..., vs.tau],
            vs.dop[..., vs.tau],
            vs.phytoplankton[..., vs.tau],
            vs.zooplankton[..., vs.tau],
            vs.detritus[..., vs.tau],
            vs.diazotrophs[..., vs.tau],
        )
        final = phosphorus(
            vs.po4[..., vs.tau] + result.po4_change,
            vs.dop[..., vs.tau] + result.dop_change,
            vs.phytoplankton[..., vs.tau] + result.phytoplankton_change,
            vs.zooplankton[..., vs.tau] + result.zooplankton_change,
            vs.detritus[..., vs.tau] + result.detritus_change,
            vs.diazotrophs[..., vs.tau] + result.diazotrophs_change,
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT
        np.testing.assert_allclose(
            np.sum(final[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            np.sum(initial[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            rtol=2e-14,
            atol=1e-10,
        )

    def test_nitrogen_budget_closes_with_fixation_and_denitrification(self):
        state = make_state(carbon=False, nitrogen=True)
        vs = state.variables
        result = mobi_biology(state, np.zeros_like(vs.swr))

        initial = (
            vs.no3[..., vs.tau]
            + vs.don[..., vs.tau]
            + vs.phytoplankton[..., vs.tau]
            + vs.zooplankton[..., vs.tau]
            + vs.detritus[..., vs.tau]
            + vs.diazotrophs[..., vs.tau]
        )
        final = (
            vs.no3[..., vs.tau]
            + result.no3_change
            + vs.don[..., vs.tau]
            + result.don_change
            + vs.phytoplankton[..., vs.tau]
            + result.phytoplankton_change
            + vs.zooplankton[..., vs.tau]
            + result.zooplankton_change
            + vs.detritus[..., vs.tau]
            + result.detritus_change
            + vs.diazotrophs[..., vs.tau]
            + result.diazotrophs_change
        )
        expected_change = state.settings.dt_tracer * (
            result.nitrogen_fixation
            - result.water_column_denitrification
            - result.benthic_denitrification
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT
        np.testing.assert_allclose(
            np.sum((final - initial)[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            np.sum(expected_change[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            rtol=2e-12,
            atol=1e-9,
        )

    def test_nitrate_limitation_reduces_primary_production(self):
        replete = make_state(carbon=False, nitrogen=True)
        limited = make_state(carbon=False, nitrogen=True)
        with limited.variables.unlock():
            limited.variables.no3 = (
                0.01
                * limited.variables.maskT[..., np.newaxis]
                * np.ones_like(limited.variables.no3)
            )

        replete_result = mobi_biology(replete, np.zeros_like(replete.variables.swr))
        limited_result = mobi_biology(limited, np.zeros_like(limited.variables.swr))
        wet = limited.variables.maskT.astype(bool)
        self.assertLess(
            float(np.sum(limited_result.net_primary_production[wet])),
            float(np.sum(replete_result.net_primary_production[wet])),
        )

    def test_warm_low_nitrate_water_supports_nitrogen_fixation(self):
        state = make_state(carbon=False, nitrogen=True)
        vs = state.variables
        with vs.unlock():
            tracer_mask = vs.maskT[..., np.newaxis]
            vs.temp = 25.0 * tracer_mask * np.ones_like(vs.temp)
            vs.no3 = 0.01 * tracer_mask * np.ones_like(vs.no3)

        result = mobi_biology(state, np.zeros_like(vs.swr))
        wet = vs.maskT.astype(bool)
        self.assertGreater(float(np.sum(result.nitrogen_fixation[wet])), 0.0)
        self.assertGreater(
            float(np.sum(result.diazotroph_primary_production[wet])), 0.0
        )

    def test_low_oxygen_remineralization_triggers_denitrification(self):
        state = make_state(carbon=False, nitrogen=True)
        vs = state.variables
        with vs.unlock():
            tracer_mask = vs.maskT[..., np.newaxis]
            vs.oxygen = 1.0 * tracer_mask * np.ones_like(vs.oxygen)
            vs.no3 = 20.0 * tracer_mask * np.ones_like(vs.no3)
            vs.detritus = 1.0 * tracer_mask * np.ones_like(vs.detritus)
            vs.swr = np.zeros_like(vs.swr)

        result = mobi_biology(state, np.zeros_like(vs.swr))
        wet = vs.maskT.astype(bool)
        self.assertGreater(float(np.sum(result.water_column_denitrification[wet])), 0.0)
        self.assertGreater(float(np.sum(result.benthic_denitrification[wet])), 0.0)

    def test_tracer_floor_preserves_each_column_inventory(self):
        state = make_state(carbon=False)
        vs = state.variables
        raw = np.array(vs.phytoplankton[..., vs.tau], copy=True)
        raw[3, 3, 0] = -0.02
        raw[3, 3, 1] = 0.08
        raw[4, 4, 2] = -1e-6

        corrected = np.asarray(enforce_conservative_tracer_floor(state, raw))
        thickness = vs.dzt[np.newaxis, np.newaxis, :]
        raw_inventory = np.sum(raw * vs.maskT * thickness, axis=2)
        corrected_inventory = np.sum(corrected * thickness, axis=2)

        np.testing.assert_allclose(
            corrected_inventory[2:-2, 2:-2],
            raw_inventory[2:-2, 2:-2],
            rtol=2e-14,
            atol=1e-12,
        )
        self.assertTrue(
            np.all(corrected[vs.maskT.astype(bool)] >= state.settings.trcmin)
        )
        self.assertTrue(np.all(corrected[~vs.maskT.astype(bool)] == 0.0))

    def test_biology_conserves_ocean_carbon_without_air_exchange(self):
        state = make_state(carbon=True, implicit_calcite=False)
        vs = state.variables
        settings = state.settings
        result = mobi_biology(state, np.zeros_like(vs.swr))

        initial = vs.dic[..., vs.tau] + settings.redfield_ratio_CN * (
            vs.phytoplankton[..., vs.tau]
            + vs.zooplankton[..., vs.tau]
            + vs.detritus[..., vs.tau]
        )
        final = (
            vs.dic[..., vs.tau]
            + result.dic_change
            + settings.redfield_ratio_CN
            * (
                vs.phytoplankton[..., vs.tau]
                + result.phytoplankton_change
                + vs.zooplankton[..., vs.tau]
                + result.zooplankton_change
                + vs.detritus[..., vs.tau]
                + result.detritus_change
            )
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT
        np.testing.assert_allclose(
            np.sum(final[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            np.sum(initial[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            rtol=2e-14,
            atol=1e-8,
        )

    def test_nitrogen_branch_conserves_carbon_without_air_exchange(self):
        state = make_state(carbon=True, nitrogen=True, implicit_calcite=False)
        vs = state.variables
        settings = state.settings
        result = mobi_biology(state, np.zeros_like(vs.swr))

        initial_organic_nitrogen = (
            vs.phytoplankton[..., vs.tau]
            + vs.zooplankton[..., vs.tau]
            + vs.detritus[..., vs.tau]
            + vs.diazotrophs[..., vs.tau]
            + vs.don[..., vs.tau]
        )
        final_organic_nitrogen = (
            vs.phytoplankton[..., vs.tau]
            + result.phytoplankton_change
            + vs.zooplankton[..., vs.tau]
            + result.zooplankton_change
            + vs.detritus[..., vs.tau]
            + result.detritus_change
            + vs.diazotrophs[..., vs.tau]
            + result.diazotrophs_change
            + vs.don[..., vs.tau]
            + result.don_change
        )
        initial = (
            vs.dic[..., vs.tau] + settings.redfield_ratio_CN * initial_organic_nitrogen
        )
        final = (
            vs.dic[..., vs.tau]
            + result.dic_change
            + settings.redfield_ratio_CN * final_organic_nitrogen
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT
        np.testing.assert_allclose(
            np.sum(final[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            np.sum(initial[2:-2, 2:-2] * weights[2:-2, 2:-2]),
            rtol=2e-14,
            atol=1e-8,
        )

    def test_nitrogen_full_step_is_finite_and_positive(self):
        state = make_state(carbon=True, nitrogen=True)
        vs = state.variables
        result = integrate_npzd(state)

        for name in (
            "no3",
            "dop",
            "don",
            "diazotrophs",
            "oxygen",
        ):
            values = np.asarray(getattr(result, name))[..., vs.taup1]
            self.assertTrue(np.all(np.isfinite(values)), name)
            self.assertTrue(np.all(values >= 0.0), name)

    def test_implicit_calcite_redistribution_is_conservative(self):
        state = make_state(carbon=True, implicit_calcite=True)
        vs = state.variables
        settings = state.settings
        result = mobi_biology(state, np.zeros_like(vs.swr))

        initial_organic_nitrogen = (
            vs.phytoplankton[..., vs.tau]
            + vs.zooplankton[..., vs.tau]
            + vs.detritus[..., vs.tau]
        )
        initial_carbon = (
            vs.dic[..., vs.tau] + settings.redfield_ratio_CN * initial_organic_nitrogen
        )
        final_organic_nitrogen = (
            vs.phytoplankton[..., vs.tau]
            + result.phytoplankton_change
            + vs.zooplankton[..., vs.tau]
            + result.zooplankton_change
            + vs.detritus[..., vs.tau]
            + result.detritus_change
        )
        final_carbon = (
            vs.dic[..., vs.tau]
            + result.dic_change
            + settings.redfield_ratio_CN * final_organic_nitrogen
        )
        # Organic nitrogen uptake raises alkalinity by one equivalent per N;
        # calcite production and redistribution must conserve the combination.
        initial_alkalinity = vs.alkalinity[..., vs.tau] - initial_organic_nitrogen
        final_alkalinity = (
            vs.alkalinity[..., vs.tau]
            + result.alkalinity_change
            - final_organic_nitrogen
        )
        weights = vs.dzt[np.newaxis, np.newaxis, :] * vs.maskT

        for initial, final in (
            (initial_carbon, final_carbon),
            (initial_alkalinity, final_alkalinity),
        ):
            np.testing.assert_allclose(
                np.sum(final[2:-2, 2:-2] * weights[2:-2, 2:-2]),
                np.sum(initial[2:-2, 2:-2] * weights[2:-2, 2:-2]),
                rtol=2e-14,
                atol=1e-8,
            )

    def test_zeroed_cooling_forcing_still_masks_ice(self):
        state = make_state(carbon=True)
        vs = state.variables
        with vs.unlock():
            vs.temp = update(
                vs.temp,
                at[:, :, -1, vs.tau],
                -2.0 * vs.maskT[:, :, -1],
            )
            vs.forc_temp_surface = np.zeros_like(vs.forc_temp_surface)
            vs.surface_taux = 0.1 * np.ones_like(vs.surface_taux)

        result = carbon_flux(state)
        np.testing.assert_allclose(result.cflux, 0.0, rtol=0.0, atol=0.0)

    def test_carbonate_system_and_full_step_are_finite(self):
        state = make_state(carbon=True)
        vs = state.variables
        chemistry = carbonate_system(
            vs.temp[:, :, -1, vs.tau],
            vs.salt[:, :, -1, vs.tau],
            vs.dic[:, :, -1, vs.tau] * 1e-3,
            vs.alkalinity[:, :, -1, vs.tau] * 1e-3,
            vs.atmospheric_co2,
            state.settings.atmospheric_pressure,
            vs.hSWS,
            vs.maskT[:, :, -1],
        )
        self.assertTrue(np.all(np.isfinite(chemistry.pCO2)))
        self.assertTrue(np.all(chemistry.hSWS > 0.0))

        result = integrate_npzd(state)
        for name in (
            "phytoplankton",
            "zooplankton",
            "detritus",
            "po4",
            "dic",
            "alkalinity",
        ):
            values = np.asarray(getattr(result, name))[..., vs.taup1]
            self.assertTrue(np.all(np.isfinite(values)), name)
            self.assertTrue(np.all(values >= 0.0), name)


if __name__ == "__main__":
    unittest.main()
