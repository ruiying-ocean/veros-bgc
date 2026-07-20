"""Settings exposed by the Veros-BGC plugin.

The defaults in this module follow the ``O_npzd`` branch of MOBI's
``mobi_init`` routine.  Rates are converted from the Fortran model's per-day
units to SI units at definition time.
"""

from veros.settings import Setting

SECONDS_PER_DAY = 86400.0


SETTINGS = {
    "enable_npzd": Setting(False, bool, "Enable the MOBI NPZD plugin"),
    "enable_carbon": Setting(
        False, bool, "Enable DIC, alkalinity, and air-sea CO2 exchange"
    ),
    "enable_nitrogen": Setting(
        False,
        bool,
        "Enable MOBI nitrate, dissolved organic nutrients, diazotrophs, oxygen, "
        "nitrogen fixation, and denitrification",
    ),
    "enable_implicit_calcite": Setting(
        True,
        bool,
        "Represent calcite production and dissolution with the implicit MOBI profile",
    ),
    "enable_bgc_superbee_advection": Setting(
        True,
        bool,
        "Use flux-limited Superbee advection for biogeochemical tracers",
    ),
    "enable_bgc_conservative_clipping": Setting(
        True,
        bool,
        "Enforce the MOBI tracer floor without changing column inventories",
    ),
    "dt_bio": Setting(
        0.0,
        float,
        "Biogeochemical time step in seconds (0 selects one quarter of dt_tracer)",
    ),
    "light_attenuation_phytoplankton": Setting(
        0.047,
        float,
        "Light attenuation by phytoplankton [m2 / mmol N]",
    ),
    "light_attenuation_water": Setting(
        0.04, float, "Light attenuation by water [1 / m]"
    ),
    "light_attenuation_ice": Setting(
        5.0, float, "Optical thickness of sea ice and snow"
    ),
    "photosynthesis_initial_slope": Setting(
        0.16 / SECONDS_PER_DAY,
        float,
        "Initial slope of the phytoplankton P-I curve [1 / (W m-2 s)]",
    ),
    "photosynthetically_active_radiation_fraction": Setting(
        0.43,
        float,
        "Fraction of incoming shortwave radiation available for photosynthesis",
    ),
    "remineralization_rate_detritus": Setting(
        0.07 / SECONDS_PER_DAY,
        float,
        "Detritus remineralization rate at 0 degrees C [1 / s]",
    ),
    "bbio": Setting(1.066, float, "Base of the MOBI temperature scaling factor"),
    "cbio": Setting(
        1.0,
        float,
        "Exponent coefficient of the MOBI temperature scaling factor [1 / deg C]",
    ),
    "maximum_growth_rate_phyto": Setting(
        0.6 / SECONDS_PER_DAY,
        float,
        "Maximum phytoplankton growth rate at 0 degrees C [1 / s]",
    ),
    "maximum_grazing_rate": Setting(
        0.38 / SECONDS_PER_DAY,
        float,
        "Maximum zooplankton grazing rate at 0 degrees C [1 / s]",
    ),
    "fast_recycling_rate_phytoplankton": Setting(
        0.015 / SECONDS_PER_DAY,
        float,
        "Temperature-dependent fast phytoplankton recycling rate [1 / s]",
    ),
    "fast_recycling_rate_diazotrophs": Setting(
        0.001 / SECONDS_PER_DAY,
        float,
        "Temperature-dependent fast diazotroph recycling rate [1 / s]",
    ),
    "saturation_constant_N": Setting(
        0.7, float, "Half-saturation constant for N uptake [mmol N / m3]"
    ),
    "saturation_constant_Z_grazing": Setting(
        0.15,
        float,
        "Half-saturation constant for zooplankton grazing [mmol N / m3]",
    ),
    "specific_mortality_phytoplankton": Setting(
        0.03 / SECONDS_PER_DAY,
        float,
        "Specific phytoplankton mortality rate [1 / s]",
    ),
    "quadric_mortality_zooplankton": Setting(
        0.06 / SECONDS_PER_DAY,
        float,
        "Quadratic zooplankton mortality rate [m3 / (mmol N s)]",
    ),
    "quadric_mortality_diazotrophs": Setting(
        0.0001 / SECONDS_PER_DAY,
        float,
        "Quadratic diazotroph mortality rate [m3 / (mmol N s)]",
    ),
    "assimilation_efficiency": Setting(
        0.70, float, "Fraction of grazed material that is digested"
    ),
    "zooplankton_growth_efficiency": Setting(
        0.60,
        float,
        "Fraction of digested material converted to zooplankton biomass",
    ),
    "wd0": Setting(
        16.0 / SECONDS_PER_DAY, float, "Detritus sinking speed at the surface [m / s]"
    ),
    "mwz": Setting(
        1000.0, float, "Depth below which detritus sinking speed is constant [m]"
    ),
    "mw": Setting(
        0.02 / SECONDS_PER_DAY,
        float,
        "Linear increase of detritus sinking speed with depth [1 / s]",
    ),
    "zprefP": Setting(0.35, float, "Zooplankton preference for phytoplankton"),
    "zprefZ": Setting(0.35, float, "Zooplankton preference for zooplankton"),
    "zprefDet": Setting(0.30, float, "Zooplankton preference for detritus"),
    "zprefDiaz": Setting(0.10, float, "Zooplankton preference for diazotrophs"),
    "diazotroph_growth_rate_factor": Setting(
        0.08,
        float,
        "MOBI jdiar factor reducing diazotroph maximum growth",
    ),
    "diazotroph_NP_ratio": Setting(
        28.0, float, "Diazotroph nitrogen-to-phosphorus ratio"
    ),
    "refractory_fraction_phytoplankton_mortality": Setting(
        0.08,
        float,
        "Fraction of phytoplankton mortality routed to dissolved organic matter",
    ),
    "refractory_fraction_phytoplankton_recycling": Setting(
        0.01,
        float,
        "Fraction of fast phytoplankton recycling routed to dissolved organic matter",
    ),
    "remineralization_rate_don": Setting(
        2.33e-5 / SECONDS_PER_DAY,
        float,
        "Dissolved organic nitrogen remineralization rate at 0 degrees C [1 / s]",
    ),
    "remineralization_rate_dop": Setting(
        7.0e-5 / SECONDS_PER_DAY,
        float,
        "Dissolved organic phosphorus remineralization rate at 0 degrees C [1 / s]",
    ),
    "dop_uptake_efficiency": Setting(
        0.4,
        float,
        "MOBI handicap applied when phytoplankton use dissolved organic phosphorus",
    ),
    "redfield_ratio_PN": Setting(
        1.0 / 16.0, float, "MOBI phosphorus-to-nitrogen ratio"
    ),
    "redfield_ratio_CP": Setting(7.1 * 16.0, float, "MOBI carbon-to-phosphorus ratio"),
    "redfield_ratio_ON": Setting(10.6, float, "MOBI oxygen-to-nitrogen ratio"),
    "redfield_ratio_CN": Setting(7.1, float, "MOBI carbon-to-nitrogen ratio"),
    "trcmin": Setting(1e-13, float, "Minimum concentration used by MOBI [mmol / m3]"),
    "u1_min": Setting(
        1e-6, float, "Lower bound in the Evans-Parslow light-growth approximation"
    ),
    "zooplankton_max_growth_temp": Setting(
        20.0,
        float,
        "Temperature above which zooplankton growth no longer accelerates [deg C]",
    ),
    "capr": Setting(0.022, float, "Carbonate-to-organic-carbon production ratio"),
    "dcaco3": Setting(6500.0, float, "Calcite remineralization e-folding depth [m]"),
    "atmospheric_pressure": Setting(
        1.0, float, "Atmospheric pressure used by carbonate chemistry [atm]"
    ),
    "co2_transfer_coefficient": Setting(
        0.337 * 0.75 / 3.6e5,
        float,
        "Coefficient in the quadratic air-sea CO2 piston velocity [s / m]",
    ),
    "wind_speed_from_stress_factor": Setting(
        500.0,
        float,
        "Legacy MOBI factor used to diagnose wind speed from Veros wind stress",
    ),
}
