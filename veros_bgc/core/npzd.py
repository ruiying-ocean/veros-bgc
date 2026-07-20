"""MOBI ecosystem dynamics and coupling to the current Veros tracer API.

This module implements MOBI's ``O_npzd`` core, the ``O_npzd_nitrogen`` and
required ``O_npzd_o2`` branches, and the optional ``O_carbon`` /
``O_npzd_alk`` and implicit-calcite branches. The kernel is array based and
supports both Veros' NumPy and JAX backends.
"""

from veros import KernelOutput, veros_kernel, veros_routine
from veros.core import advection, diffusion, utilities
from veros.core.isoneutral.diffusion import isoneutral_diffusion_tracer
from veros.core.operators import at, update, update_add
from veros.core.operators import numpy as npx
from veros.variables import allocate

from . import atmospherefluxes

TRACER_FIELDS = (
    ("phytoplankton", "dphytoplankton"),
    ("zooplankton", "dzooplankton"),
    ("detritus", "ddetritus"),
    ("po4", "dpo4"),
)
NITROGEN_FIELDS = (
    ("no3", "dno3"),
    ("dop", "ddop"),
    ("don", "ddon"),
    ("diazotrophs", "ddiazotrophs"),
    ("oxygen", "doxygen"),
)
CARBON_FIELDS = (("dic", "ddic"), ("alkalinity", "dalkalinity"))


def _bio_timestep(settings):
    dt_bio = settings.dt_bio if settings.dt_bio > 0.0 else settings.dt_tracer / 4.0
    n_substeps = int(round(settings.dt_tracer / dt_bio))
    return dt_bio, n_substeps


@veros_routine
def setup_npzd(state):
    """Validate MOBI settings and initialize geometry-dependent fields."""

    settings = state.settings
    if not settings.enable_npzd:
        return

    dt_bio, n_substeps = _bio_timestep(settings)
    if dt_bio <= 0.0 or n_substeps < 1:
        raise ValueError("dt_bio must be positive and no larger than dt_tracer")
    if abs(n_substeps * dt_bio - settings.dt_tracer) > 1e-10 * settings.dt_tracer:
        raise ValueError("dt_tracer must be an integer multiple of dt_bio")
    if (
        settings.enable_carbon
        and settings.enable_implicit_calcite
        and settings.dcaco3 <= 0.0
    ):
        raise ValueError("dcaco3 must be positive when the carbon cycle is enabled")

    preference_sum = settings.zprefP + settings.zprefZ + settings.zprefDet
    if settings.enable_nitrogen:
        preference_sum += settings.zprefDiaz
    if preference_sum <= 0.0:
        raise ValueError("at least one zooplankton grazing preference must be positive")
    if settings.enable_nitrogen:
        if settings.diazotroph_NP_ratio <= 0.0:
            raise ValueError("diazotroph_NP_ratio must be positive")
        for name in (
            "refractory_fraction_phytoplankton_mortality",
            "refractory_fraction_phytoplankton_recycling",
            "dop_uptake_efficiency",
        ):
            value = getattr(settings, name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between zero and one")

    state.variables.update(setup_npzd_kernel(state))


@veros_kernel
def setup_npzd_kernel(state):
    vs = state.variables
    settings = state.settings

    vertical_index = npx.arange(settings.nz)[npx.newaxis, npx.newaxis, :]
    bottom_mask = npx.logical_and(
        vs.maskT,
        vertical_index == (vs.kbot[:, :, npx.newaxis] - 1),
    )

    out = {"bottom_mask": bottom_mask}

    if settings.enable_carbon:
        # ``zw`` is the upper face of each T cell and is zero at the sea
        # surface.  Calcite follows an exponential dissolution profile; any
        # material left at the seabed dissolves in the deepest wet cell.
        top_depth = npx.maximum(-vs.zw, 0.0)
        bottom_depth = top_depth + vs.dzt
        profile_1d = (
            npx.exp(-top_depth / settings.dcaco3)
            - npx.exp(-bottom_depth / settings.dcaco3)
        ) / vs.dzt
        seabed_1d = npx.exp(-top_depth / settings.dcaco3) / vs.dzt
        profile = npx.broadcast_to(
            profile_1d[npx.newaxis, npx.newaxis, :], vs.maskT.shape
        )
        seabed = npx.broadcast_to(
            seabed_1d[npx.newaxis, npx.newaxis, :], vs.maskT.shape
        )
        out["rcak"] = npx.where(bottom_mask, seabed, profile) * vs.maskT

    return KernelOutput(**out)


@veros_routine
def npzd(state):
    """Advance MOBI tracers by one Veros tracer time step."""

    if not state.settings.enable_npzd:
        return

    state.variables.update(integrate_npzd(state))


@veros_kernel
def integrate_npzd(state):
    """Couple MOBI source terms to Veros tracer transport."""

    vs = state.variables
    settings = state.settings
    out = {}

    if settings.enable_carbon:
        carbon = atmospherefluxes.carbon_flux(state)
        surface_carbon_flux = carbon.cflux
        out.update(
            cflux=carbon.cflux,
            wind_speed=carbon.wind_speed,
            hSWS=carbon.hSWS,
            pCO2=carbon.pCO2,
            dpCO2=carbon.dpCO2,
            co2star=carbon.co2star,
            dco2star=carbon.dco2star,
        )
    else:
        surface_carbon_flux = npx.zeros_like(vs.swr)

    biology = mobi_biology(state, surface_carbon_flux)

    for tracer_name, tendency_name in TRACER_FIELDS:
        tracer, tendency = transport_tracer(
            state,
            getattr(vs, tracer_name),
            getattr(vs, tendency_name),
            getattr(biology, f"{tracer_name}_change"),
        )
        out[tracer_name] = tracer
        out[tendency_name] = tendency

    if settings.enable_nitrogen:
        for tracer_name, tendency_name in NITROGEN_FIELDS:
            tracer, tendency = transport_tracer(
                state,
                getattr(vs, tracer_name),
                getattr(vs, tendency_name),
                getattr(biology, f"{tracer_name}_change"),
            )
            out[tracer_name] = tracer
            out[tendency_name] = tendency

    if settings.enable_carbon:
        for tracer_name, tendency_name in CARBON_FIELDS:
            tracer, tendency = transport_tracer(
                state,
                getattr(vs, tracer_name),
                getattr(vs, tendency_name),
                getattr(biology, f"{tracer_name}_change"),
            )
            out[tracer_name] = tracer
            out[tendency_name] = tendency

    out.update(
        rctheta=biology.rctheta,
        dayfrac=biology.dayfrac,
        excretion_total=biology.excretion_total,
        net_primary_production=biology.net_primary_production,
        detritus_remineralization=biology.detritus_remineralization,
        detritus_export=biology.detritus_export,
    )
    if settings.enable_nitrogen:
        out.update(
            diazotroph_primary_production=biology.diazotroph_primary_production,
            nitrogen_fixation=biology.nitrogen_fixation,
            water_column_denitrification=biology.water_column_denitrification,
            benthic_denitrification=biology.benthic_denitrification,
        )

    return KernelOutput(**out)


@veros_kernel
def mobi_biology(state, surface_carbon_flux):
    """Integrate MOBI source/sink terms over one Veros tracer step.

    The implementation follows the ``O_npzd`` source terms and, when enabled,
    the ``O_npzd_nitrogen`` / ``O_npzd_o2`` branches in
    ``mobi_src/updates/npzd_src.F``. Returned ``*_change`` fields are finite
    concentration increments, not tendencies.
    """

    vs = state.variables
    settings = state.settings
    dt_bio, n_substeps = _bio_timestep(settings)
    mask = vs.maskT

    phyto_initial = vs.phytoplankton[..., vs.tau]
    zoo_initial = vs.zooplankton[..., vs.tau]
    detritus_initial = vs.detritus[..., vs.tau]
    po4_initial = vs.po4[..., vs.tau]

    phyto = npx.maximum(phyto_initial, settings.trcmin) * mask
    zoo = npx.maximum(zoo_initial, settings.trcmin) * mask
    detritus = npx.maximum(detritus_initial, settings.trcmin) * mask
    po4 = npx.maximum(po4_initial, settings.trcmin) * mask

    if settings.enable_nitrogen:
        no3_initial = vs.no3[..., vs.tau]
        dop_initial = vs.dop[..., vs.tau]
        don_initial = vs.don[..., vs.tau]
        diazotrophs_initial = vs.diazotrophs[..., vs.tau]
        oxygen_initial = vs.oxygen[..., vs.tau]
        no3 = npx.maximum(no3_initial, settings.trcmin) * mask
        dop = npx.maximum(dop_initial, settings.trcmin) * mask
        don = npx.maximum(don_initial, settings.trcmin) * mask
        diazotrophs = npx.maximum(diazotrophs_initial, settings.trcmin) * mask
        oxygen = npx.maximum(oxygen_initial, settings.trcmin) * mask
    else:
        no3 = npx.zeros_like(phyto)
        dop = npx.zeros_like(phyto)
        don = npx.zeros_like(phyto)
        diazotrophs = npx.zeros_like(phyto)
        oxygen = npx.zeros_like(phyto)

    if settings.enable_carbon:
        dic_initial = vs.dic[..., vs.tau]
        alkalinity_initial = vs.alkalinity[..., vs.tau]
        dic = npx.maximum(dic_initial, settings.trcmin) * mask
        alkalinity = npx.maximum(alkalinity_initial, settings.trcmin) * mask
        dic = update_add(
            dic,
            at[:, :, -1],
            surface_carbon_flux * settings.dt_tracer / vs.dzt[-1],
        )
    else:
        dic = npx.zeros_like(phyto)
        alkalinity = npx.zeros_like(phyto)

    # Seasonal declination, refracted path length, and daylight fraction.
    year_fraction = npx.mod(vs.time / (360.0 * 86400.0), 1.0)
    declination = npx.sin((year_fraction - 0.72) * 2.0 * settings.pi) * 0.4
    radians = settings.pi / 180.0
    incidence = npx.clip(vs.yt * radians - declination, -1.5, 1.5)
    rctheta = settings.light_attenuation_water / npx.sqrt(
        1.0 - (1.0 - npx.cos(incidence) ** 2) / 1.33**2
    )
    day_argument = npx.clip(-npx.tan(vs.yt * radians) * npx.tan(declination), -1.0, 1.0)
    dayfrac = npx.maximum(1e-12, npx.arccos(day_argument) / settings.pi)

    # Light at the top of every cell.  Veros orders z from seafloor to
    # surface, hence the reverse cumulative sum.
    light_blockers = phyto + diazotrophs
    plankton_inventory = light_blockers * vs.dzt[npx.newaxis, npx.newaxis, :]
    integrated_above = (
        npx.cumsum(plankton_inventory[:, :, ::-1], axis=2)[:, :, ::-1]
        - plankton_inventory
    )
    top_depth = npx.maximum(-vs.zw, 0.0)
    light = (
        2.0
        * settings.photosynthesis_initial_slope
        * settings.photosynthetically_active_radiation_fraction
        * vs.swr[:, :, npx.newaxis]
        * npx.exp(-settings.light_attenuation_phytoplankton * integrated_above)
        * npx.exp(
            -top_depth[npx.newaxis, npx.newaxis, :]
            * rctheta[npx.newaxis, :, npx.newaxis]
        )
    )
    ice_mask = npx.logical_and(
        vs.temp[:, :, -1, vs.tau] * mask[:, :, -1] < -1.8,
        vs.forc_temp_surface <= 0.0,
    )
    light = light * npx.exp(
        -settings.light_attenuation_ice * ice_mask[:, :, npx.newaxis]
    )

    bct = settings.bbio ** (settings.cbio * vs.temp[..., vs.tau])
    jmax = settings.maximum_growth_rate_phyto * bct
    gd = npx.maximum(jmax * dayfrac[npx.newaxis, :, npx.newaxis], 1e-30)
    attenuation = (
        settings.light_attenuation_water
        + settings.light_attenuation_phytoplankton * light_blockers
    ) * vs.dzt[npx.newaxis, npx.newaxis, :]
    attenuation = npx.maximum(attenuation, 1e-30)
    f1 = npx.exp(-attenuation)
    u1 = npx.maximum(light / gd, settings.u1_min)
    u2 = npx.maximum(u1 * f1, settings.u1_min * f1)
    phi1 = _evans_parslow_phi(u1)
    phi2 = _evans_parslow_phi(u2)
    average_light_growth = gd * (phi1 - phi2) / attenuation * mask

    capped_temperature_factor = settings.bbio ** (
        settings.cbio
        * npx.minimum(settings.zooplankton_max_growth_temp, vs.temp[..., vs.tau])
    )
    if settings.enable_nitrogen:
        oxygen_grazing_factor = 0.5 * (npx.tanh(oxygen - 8.0) + 1.0)
        gmax = (
            settings.maximum_grazing_rate
            * oxygen_grazing_factor
            * capped_temperature_factor
        )
        detritus_remineralization_rate = (
            settings.remineralization_rate_detritus
            * (0.65 + 0.35 * npx.tanh(oxygen - 3.0))
            * bct
        )
        jmax_diazotrophs = (
            npx.maximum(0.0, settings.maximum_growth_rate_phyto * (bct - 2.6))
            * settings.diazotroph_growth_rate_factor
        )
        gd_diazotrophs = npx.maximum(
            jmax_diazotrophs * dayfrac[npx.newaxis, :, npx.newaxis], 1e-14
        )
        u1_diazotrophs = npx.maximum(light / gd_diazotrophs, settings.u1_min)
        u2_diazotrophs = npx.maximum(u1_diazotrophs * f1, settings.u1_min * f1)
        average_light_growth_diazotrophs = (
            gd_diazotrophs
            * (_evans_parslow_phi(u1_diazotrophs) - _evans_parslow_phi(u2_diazotrophs))
            / attenuation
            * mask
        )
    else:
        gmax = settings.maximum_grazing_rate * capped_temperature_factor
        detritus_remineralization_rate = settings.remineralization_rate_detritus * bct
        jmax_diazotrophs = npx.zeros_like(phyto)
        average_light_growth_diazotrophs = npx.zeros_like(phyto)

    fast_recycling_rate = settings.fast_recycling_rate_phytoplankton * bct
    fast_recycling_rate_diazotrophs = settings.fast_recycling_rate_diazotrophs * bct
    cell_depth = npx.maximum(-vs.zt, 0.0)
    sinking_speed = settings.wd0 + settings.mw * npx.minimum(cell_depth, settings.mwz)
    sinking_rate = (
        sinking_speed[npx.newaxis, npx.newaxis, :] / vs.dzt[npx.newaxis, npx.newaxis, :]
    )

    phyto_flag = npx.logical_and(mask, phyto_initial > settings.trcmin)
    zoo_flag = npx.logical_and(mask, zoo_initial > settings.trcmin)
    detritus_flag = npx.logical_and(mask, detritus_initial > settings.trcmin)
    po4_flag = npx.logical_and(mask, po4_initial > settings.trcmin)

    if settings.enable_nitrogen:
        no3_flag = npx.logical_and(mask, no3_initial > settings.trcmin)
        dop_flag = npx.logical_and(mask, dop_initial > settings.trcmin)
        don_flag = npx.logical_and(mask, don_initial > settings.trcmin)
        diazotrophs_flag = npx.logical_and(mask, diazotrophs_initial > settings.trcmin)
    else:
        diazotrophs_flag = npx.zeros_like(mask, dtype="bool")

    preference_sum = settings.zprefP + settings.zprefZ + settings.zprefDet
    if settings.enable_nitrogen:
        preference_sum += settings.zprefDiaz
    zpref_p = settings.zprefP / preference_sum
    zpref_z = settings.zprefZ / preference_sum
    zpref_det = settings.zprefDet / preference_sum
    zpref_diaz = settings.zprefDiaz / preference_sum

    calcite_production = npx.zeros_like(phyto)
    npp_sum = npx.zeros_like(phyto)
    diazotroph_npp_sum = npx.zeros_like(phyto)
    excretion_sum = npx.zeros_like(phyto)
    remineralization_sum = npx.zeros_like(phyto)
    export_sum = npx.zeros_like(phyto)
    organic_dic_rate_sum = npx.zeros_like(phyto)
    nitrogen_fixation_sum = npx.zeros_like(phyto)

    for _ in range(n_substeps):
        if settings.enable_nitrogen:
            phosphorus_half_saturation = (
                settings.saturation_constant_N * settings.redfield_ratio_PN
            )
            po4_limitation = po4 / (phosphorus_half_saturation + po4)
            dop_limitation = (
                settings.dop_uptake_efficiency
                * dop
                / (phosphorus_half_saturation + dop)
            )
            use_dop = npx.greater_equal(dop_limitation, po4_limitation)
            phosphorus_limitation = npx.where(use_dop, dop_limitation, po4_limitation)
            phosphorus_pool_flag = npx.where(use_dop, dop_flag, po4_flag)
            no3_limitation = no3 / (settings.saturation_constant_N + no3)
            growth_rate = npx.minimum(
                average_light_growth,
                npx.minimum(jmax * phosphorus_limitation, jmax * no3_limitation),
            )
            npp = growth_rate * phyto * phyto_flag * no3_flag * phosphorus_pool_flag
            dop_uptake = npp * use_dop

            growth_rate_diazotrophs = npx.minimum(
                average_light_growth_diazotrophs,
                jmax_diazotrophs * phosphorus_limitation,
            )
            diazotroph_npp = (
                npx.maximum(0.0, growth_rate_diazotrophs * diazotrophs)
                * diazotrophs_flag
                * phosphorus_pool_flag
            )
            diazotroph_dop_uptake = diazotroph_npp * use_dop
            diazotroph_no3_uptake = (
                (0.5 + 0.5 * npx.tanh(no3 - 5.0)) * diazotroph_npp * no3_flag
            )
            nitrogen_fixation = diazotroph_npp - diazotroph_no3_uptake
        else:
            limitation = po4 / (
                settings.saturation_constant_N * settings.redfield_ratio_PN + po4
            )
            growth_rate = npx.minimum(average_light_growth, jmax * limitation)
            npp = growth_rate * phyto * phyto_flag * po4_flag
            dop_uptake = npx.zeros_like(phyto)
            diazotroph_npp = npx.zeros_like(phyto)
            diazotroph_dop_uptake = npx.zeros_like(phyto)
            diazotroph_no3_uptake = npx.zeros_like(phyto)
            nitrogen_fixation = npx.zeros_like(phyto)

        theta_z = zpref_p * phyto + zpref_det * detritus + zpref_z * zoo
        if settings.enable_nitrogen:
            theta_z = (
                theta_z
                + zpref_diaz * diazotrophs
                + settings.saturation_constant_Z_grazing
            )
        else:
            theta_z = (
                theta_z
                + settings.saturation_constant_Z_grazing * settings.redfield_ratio_PN
            )
        theta_z = npx.maximum(theta_z, settings.trcmin)

        grazing_phyto = gmax * zpref_p / theta_z * phyto * zoo * phyto_flag * zoo_flag
        grazing_zoo = gmax * zpref_z / theta_z * zoo * zoo * zoo_flag
        grazing_detritus = (
            gmax * zpref_det / theta_z * detritus * zoo * detritus_flag * zoo_flag
        )
        if settings.enable_nitrogen:
            grazing_diazotrophs = (
                gmax
                * zpref_diaz
                / theta_z
                * diazotrophs
                * zoo
                * diazotrophs_flag
                * zoo_flag
            )
            diazotroph_redfield_fraction = (
                1.0 / settings.redfield_ratio_PN / settings.diazotroph_NP_ratio
            )
        else:
            grazing_diazotrophs = npx.zeros_like(phyto)
            diazotroph_redfield_fraction = 0.0

        total_grazing = grazing_phyto + grazing_zoo + grazing_detritus
        redfield_equivalent_grazing = (
            total_grazing + grazing_diazotrophs * diazotroph_redfield_fraction
        )
        digestion = settings.assimilation_efficiency * redfield_equivalent_grazing
        excretion = (1.0 - settings.zooplankton_growth_efficiency) * digestion
        sloppy_feeding = (
            1.0 - settings.assimilation_efficiency
        ) * redfield_equivalent_grazing
        non_redfield_diazotroph_excretion = grazing_diazotrophs * (
            1.0 - diazotroph_redfield_fraction
        )

        phytoplankton_mortality = (
            settings.specific_mortality_phytoplankton * phyto * phyto_flag
        )
        fast_recycling = fast_recycling_rate * phyto * phyto_flag
        diazotroph_fast_recycling = (
            fast_recycling_rate_diazotrophs * diazotrophs * diazotrophs_flag
        )
        diazotroph_mortality = (
            settings.quadric_mortality_diazotrophs * diazotrophs**2 * diazotrophs_flag
        )
        zooplankton_mortality = (
            settings.quadric_mortality_zooplankton * zoo**2 * zoo_flag
        )
        remineralization = detritus_remineralization_rate * detritus * detritus_flag
        if settings.enable_nitrogen:
            don_remineralization = (
                settings.remineralization_rate_don * bct * don * don_flag
            )
            dop_remineralization = (
                settings.remineralization_rate_dop * bct * dop * dop_flag
            )
        else:
            don_remineralization = npx.zeros_like(phyto)
            dop_remineralization = npx.zeros_like(phyto)

        detritus_export = sinking_rate * detritus * detritus_flag
        detritus_import = npx.zeros_like(detritus_export)
        detritus_import = update(
            detritus_import,
            at[:, :, :-1],
            detritus_export[:, :, 1:]
            * vs.dzt[npx.newaxis, npx.newaxis, 1:]
            / vs.dzt[npx.newaxis, npx.newaxis, :-1],
        )
        detritus_import = detritus_import * mask

        phyto = phyto + dt_bio * (
            npp - phytoplankton_mortality - grazing_phyto - fast_recycling
        )
        zoo = zoo + dt_bio * (
            digestion - zooplankton_mortality - grazing_zoo - excretion
        )
        if settings.enable_nitrogen:
            refractory_mortality = settings.refractory_fraction_phytoplankton_mortality
            refractory_recycling = settings.refractory_fraction_phytoplankton_recycling
            diazotroph_PN_ratio = 1.0 / settings.diazotroph_NP_ratio

            detritus = detritus + dt_bio * (
                (1.0 - refractory_mortality) * phytoplankton_mortality
                + sloppy_feeding
                + zooplankton_mortality
                - remineralization
                - grazing_detritus
                - detritus_export
                + detritus_import
                + diazotroph_mortality * diazotroph_redfield_fraction
            )
            po4 = po4 + dt_bio * (
                settings.redfield_ratio_PN
                * (
                    excretion
                    + remineralization
                    + (1.0 - refractory_recycling) * fast_recycling
                    - (npp - dop_uptake)
                )
                + diazotroph_PN_ratio
                * (diazotroph_fast_recycling - (diazotroph_npp - diazotroph_dop_uptake))
                + dop_remineralization
            )
            dop = dop + dt_bio * (
                settings.redfield_ratio_PN
                * (
                    refractory_mortality * phytoplankton_mortality
                    + refractory_recycling * fast_recycling
                    - dop_uptake
                )
                - diazotroph_PN_ratio * diazotroph_dop_uptake
                - dop_remineralization
            )
            no3 = no3 + dt_bio * (
                excretion
                + remineralization
                + (1.0 - refractory_recycling) * fast_recycling
                - npp
                + diazotroph_fast_recycling
                - diazotroph_no3_uptake
                + don_remineralization
                + non_redfield_diazotroph_excretion
                + diazotroph_mortality * (1.0 - diazotroph_redfield_fraction)
            )
            don = don + dt_bio * (
                refractory_mortality * phytoplankton_mortality
                + refractory_recycling * fast_recycling
                - don_remineralization
            )
            diazotrophs = diazotrophs + dt_bio * (
                diazotroph_npp
                - diazotroph_mortality
                - diazotroph_fast_recycling
                - grazing_diazotrophs
            )

            organic_dic_rate = settings.redfield_ratio_CN * (
                excretion
                + remineralization
                + (1.0 - refractory_recycling) * fast_recycling
                - npp
                + diazotroph_fast_recycling
                - diazotroph_npp
                + don_remineralization
                + non_redfield_diazotroph_excretion
                + diazotroph_mortality * (1.0 - diazotroph_redfield_fraction)
            )
        else:
            detritus = detritus + dt_bio * (
                phytoplankton_mortality
                + sloppy_feeding
                + zooplankton_mortality
                - remineralization
                - grazing_detritus
                - detritus_export
                + detritus_import
            )
            po4 = po4 + dt_bio * settings.redfield_ratio_PN * (
                remineralization + excretion - npp + fast_recycling
            )
            organic_dic_rate = settings.redfield_ratio_CN * (
                fast_recycling + excretion + remineralization - npp
            )

        organic_dic_rate_sum = organic_dic_rate_sum + organic_dic_rate

        if settings.enable_carbon:
            dic = dic + dt_bio * organic_dic_rate
            # MOBI O_npzd_alk: organic carbon drawdown raises alkalinity.
            alkalinity = (
                alkalinity - dt_bio * organic_dic_rate / settings.redfield_ratio_CN
            )
            if settings.enable_implicit_calcite:
                calpro = (
                    (
                        phytoplankton_mortality
                        + zooplankton_mortality
                        + (grazing_phyto + grazing_zoo)
                        * (1.0 - settings.assimilation_efficiency)
                    )
                    * settings.capr
                    * settings.redfield_ratio_CN
                )
                calcite_production = calcite_production + dt_bio * calpro
                dic = dic - dt_bio * calpro
                alkalinity = alkalinity - 2.0 * dt_bio * calpro

        npp_sum = npp_sum + npp
        diazotroph_npp_sum = diazotroph_npp_sum + diazotroph_npp
        excretion_sum = excretion_sum + excretion
        remineralization_sum = remineralization_sum + remineralization
        export_sum = export_sum + detritus_export
        nitrogen_fixation_sum = nitrogen_fixation_sum + nitrogen_fixation

        # MOBI flags are irreversible within a tracer step: once a pool is
        # depleted, its outgoing source terms remain disabled until the next
        # Veros step.
        phyto_flag = npx.logical_and(phyto_flag, phyto > settings.trcmin)
        zoo_flag = npx.logical_and(zoo_flag, zoo > settings.trcmin)
        detritus_flag = npx.logical_and(detritus_flag, detritus > settings.trcmin)
        po4_flag = npx.logical_and(po4_flag, po4 > settings.trcmin)
        if settings.enable_nitrogen:
            no3_flag = npx.logical_and(no3_flag, no3 > settings.trcmin)
            dop_flag = npx.logical_and(dop_flag, dop > settings.trcmin)
            don_flag = npx.logical_and(don_flag, don > settings.trcmin)
            diazotrophs_flag = npx.logical_and(
                diazotrophs_flag, diazotrophs > settings.trcmin
            )

    # MOBI applies seafloor remineralization, denitrification, and oxygen
    # coupling in ``mobi_driver`` after the biological substeps. Keeping that
    # ordering also prevents freshly remineralized nutrients from feeding back
    # into production during the same tracer step.
    bottom_export_rate = export_sum / n_substeps * vs.bottom_mask
    po4 = po4 + settings.dt_tracer * settings.redfield_ratio_PN * bottom_export_rate
    mean_organic_dic_rate = (
        organic_dic_rate_sum / n_substeps
        + settings.redfield_ratio_CN * bottom_export_rate
    )

    if settings.enable_carbon:
        dic = dic + settings.dt_tracer * settings.redfield_ratio_CN * bottom_export_rate
        alkalinity = alkalinity - settings.dt_tracer * bottom_export_rate

    if settings.enable_nitrogen:
        benthic_denitrification = 0.06 + 0.19 * 0.99 ** (
            npx.maximum(oxygen_initial, settings.trcmin)
            - npx.maximum(no3_initial, settings.trcmin)
        )
        benthic_denitrification = (
            benthic_denitrification
            * npx.maximum(bottom_export_rate, settings.trcmin)
            * settings.redfield_ratio_CN
        )
        benthic_denitrification = npx.minimum(
            npx.maximum(benthic_denitrification, 0.0), bottom_export_rate
        )
        benthic_denitrification = (
            benthic_denitrification
            * (0.5 + 0.5 * npx.tanh(no3_initial * 10.0 - 5.0))
            * npx.logical_and(mask, no3_initial > settings.trcmin)
        )
        no3 = no3 + settings.dt_tracer * (bottom_export_rate - benthic_denitrification)

        nitrogen_fixation_rate = nitrogen_fixation_sum / n_substeps
        oxygen_demand = (
            mean_organic_dic_rate
            * settings.redfield_ratio_ON
            / settings.redfield_ratio_CN
            + 1.25 * nitrogen_fixation_rate
        )
        water_column_denitrification = npx.maximum(
            0.0,
            0.8
            * oxygen_demand
            * (0.5 - 0.5 * npx.tanh(oxygen_initial - 2.5))
            * (0.5 + 0.5 * npx.tanh(no3_initial - 2.5))
            * npx.logical_and(mask, no3_initial > settings.trcmin),
        )
        no3 = no3 - settings.dt_tracer * water_column_denitrification
        oxygen = oxygen - settings.dt_tracer * oxygen_demand * (
            0.5 + 0.5 * npx.tanh(oxygen_initial - 2.5)
        )

        if settings.enable_carbon:
            alkalinity = alkalinity + settings.dt_tracer * (
                water_column_denitrification
                + benthic_denitrification
                - nitrogen_fixation_rate
            )
    else:
        nitrogen_fixation_rate = npx.zeros_like(phyto)
        water_column_denitrification = npx.zeros_like(phyto)
        benthic_denitrification = npx.zeros_like(phyto)

    out = dict(
        phytoplankton_change=phyto - phyto_initial,
        zooplankton_change=zoo - zoo_initial,
        detritus_change=detritus - detritus_initial,
        po4_change=po4 - po4_initial,
        rctheta=rctheta,
        dayfrac=dayfrac,
        excretion_total=excretion_sum / n_substeps,
        net_primary_production=(npp_sum + diazotroph_npp_sum) / n_substeps,
        detritus_remineralization=remineralization_sum / n_substeps,
        detritus_export=export_sum / n_substeps,
    )

    if settings.enable_nitrogen:
        out.update(
            no3_change=no3 - no3_initial,
            dop_change=dop - dop_initial,
            don_change=don - don_initial,
            diazotrophs_change=diazotrophs - diazotrophs_initial,
            oxygen_change=oxygen - oxygen_initial,
            diazotroph_primary_production=diazotroph_npp_sum / n_substeps,
            nitrogen_fixation=nitrogen_fixation_rate,
            water_column_denitrification=water_column_denitrification,
            benthic_denitrification=benthic_denitrification,
        )

    if settings.enable_carbon:
        if settings.enable_implicit_calcite:
            calcite_inventory = npx.sum(
                calcite_production * vs.dzt[npx.newaxis, npx.newaxis, :], axis=2
            )
            dissolution = calcite_inventory[:, :, npx.newaxis] * vs.rcak
            dic = dic + dissolution
            alkalinity = alkalinity + 2.0 * dissolution

        out.update(
            dic_change=dic - dic_initial,
            alkalinity_change=alkalinity - alkalinity_initial,
        )

    return KernelOutput(**out)


def _evans_parslow_phi(u):
    root = npx.sqrt(1.0 + u**2)
    return npx.log(u + root) - (root - 1.0) / u


@veros_kernel
def transport_tracer(state, tracer, tendency, source_change):
    """Apply Veros transport with BGC-specific advection and bounds handling."""

    vs = state.variables
    settings = state.settings

    advective_tendency = advect_bgc_tracer(state, tracer[..., vs.tau])
    tendency = update(tendency, at[..., vs.tau], advective_tendency)
    tracer = update(
        tracer,
        at[..., vs.taup1],
        tracer[..., vs.tau]
        + settings.dt_tracer
        * (
            (1.5 + settings.AB_eps) * tendency[..., vs.tau]
            - (0.5 + settings.AB_eps) * tendency[..., vs.taum1]
        )
        * vs.maskT,
    )

    if settings.enable_hor_diffusion:
        horizontal_change, _, _ = diffusion.horizontal_diffusion(
            state, tracer[..., vs.tau], settings.K_h
        )
        tracer = update_add(
            tracer,
            at[..., vs.taup1],
            settings.dt_tracer * horizontal_change,
        )

    if settings.enable_biharmonic_mixing:
        biharmonic_change, _, _ = diffusion.biharmonic_diffusion(
            state, tracer[..., vs.tau], npx.sqrt(npx.abs(settings.K_hbi))
        )
        tracer = update_add(
            tracer,
            at[..., vs.taup1],
            settings.dt_tracer * biharmonic_change,
        )

    if settings.enable_neutral_diffusion:
        isoneutral_tendency = allocate(state.dimensions, ("xt", "yt", "zt"))
        tracer, isoneutral_tendency, _, _, _ = isoneutral_diffusion_tracer(
            state,
            tracer,
            isoneutral_tendency,
            iso=True,
            skew=False,
        )
        if settings.enable_skew_diffusion:
            tracer, _, _, _, _ = isoneutral_diffusion_tracer(
                state,
                tracer,
                isoneutral_tendency,
                iso=False,
                skew=True,
            )

    tracer = _vertical_mixing(state, tracer)
    tracer = update_add(tracer, at[..., vs.taup1], source_change)
    transported = tracer[..., vs.taup1]
    if settings.enable_bgc_conservative_clipping:
        transported = enforce_conservative_tracer_floor(state, transported)
    else:
        transported = npx.maximum(transported, settings.trcmin) * vs.maskT
    tracer = update(
        tracer,
        at[..., vs.taup1],
        transported,
    )
    tracer = update(
        tracer,
        at[..., vs.taup1],
        utilities.enforce_boundaries(tracer[..., vs.taup1], settings.enable_cyclic_x),
    )

    return tracer, tendency


@veros_kernel
def advect_bgc_tracer(state, tracer):
    """Return a conservative advection tendency for a MOBI tracer.

    The physical four-degree setup intentionally retains its configured
    temperature/salinity scheme.  MOBI tracers default to Superbee because
    centered second-order advection creates negative plankton undershoots that
    cannot be clipped without adding elemental inventory.
    """

    vs = state.variables
    settings = state.settings

    if settings.enable_bgc_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee(state, tracer)
    else:
        flux_east, flux_north, flux_top = advection.adv_flux_2nd(state, tracer)

    tendency = allocate(state.dimensions, ("xt", "yt", "zt"))
    tendency = update(
        tendency,
        at[2:-2, 2:-2, :],
        vs.maskT[2:-2, 2:-2, :]
        * (
            -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (
                vs.cost[npx.newaxis, 2:-2, npx.newaxis]
                * vs.dxt[2:-2, npx.newaxis, npx.newaxis]
            )
            - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (
                vs.cost[npx.newaxis, 2:-2, npx.newaxis]
                * vs.dyt[npx.newaxis, 2:-2, npx.newaxis]
            )
        ),
    )
    tendency = update_add(
        tendency,
        at[:, :, 0],
        -1.0 * vs.maskT[:, :, 0] * flux_top[:, :, 0] / vs.dzt[0],
    )
    tendency = update_add(
        tendency,
        at[:, :, 1:],
        -1.0
        * vs.maskT[:, :, 1:]
        * (flux_top[:, :, 1:] - flux_top[:, :, :-1])
        / vs.dzt[npx.newaxis, npx.newaxis, 1:],
    )
    return tendency


@veros_kernel
def enforce_conservative_tracer_floor(state, tracer):
    """Apply ``trcmin`` while preserving every water-column inventory.

    Numerical transport can leave tiny negative concentrations.  A direct
    ``maximum`` creates tracer mass.  Here the clipped excess is removed
    proportionally from concentrations above the floor in the same column.
    The correction is therefore positive, horizontally local, and conservative
    under the model's thickness-weighted inventory.
    """

    vs = state.variables
    settings = state.settings
    mask = vs.maskT
    thickness = vs.dzt[npx.newaxis, npx.newaxis, :]
    floor = settings.trcmin * mask
    clipped = npx.maximum(tracer, settings.trcmin) * mask
    excess = clipped - floor

    raw_inventory = npx.sum(tracer * mask * thickness, axis=2)
    floor_inventory = npx.sum(floor * thickness, axis=2)
    excess_inventory = npx.sum(excess * thickness, axis=2)
    target_excess = npx.maximum(raw_inventory - floor_inventory, 0.0)
    scale = npx.where(
        excess_inventory > 0.0,
        target_excess / npx.maximum(excess_inventory, 1e-30),
        0.0,
    )
    scale = npx.clip(scale, 0.0, 1.0)

    return (floor + excess * scale[:, :, npx.newaxis]) * mask


@veros_kernel
def _vertical_mixing(state, tracer):
    vs = state.variables
    settings = state.settings

    a_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    b_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    c_tri = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]
    delta = allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2]

    _, water_mask, edge_mask = utilities.create_water_masks(
        vs.kbot[2:-2, 2:-2], settings.nz
    )
    delta = update(
        delta,
        at[:, :, :-1],
        settings.dt_tracer
        / vs.dzw[npx.newaxis, npx.newaxis, :-1]
        * vs.kappaH[2:-2, 2:-2, :-1],
    )
    delta = update(delta, at[:, :, -1], 0.0)
    a_tri = update(
        a_tri,
        at[:, :, 1:],
        -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, 1:],
    )
    b_tri = update(
        b_tri,
        at[:, :, 1:],
        1.0
        + (delta[:, :, 1:] + delta[:, :, :-1]) / vs.dzt[npx.newaxis, npx.newaxis, 1:],
    )
    b_tri_edge = 1.0 + delta / vs.dzt[npx.newaxis, npx.newaxis, :]
    c_tri = update(
        c_tri,
        at[:, :, :-1],
        -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, :-1],
    )
    rhs = tracer[2:-2, 2:-2, :, vs.taup1]
    solution = utilities.solve_implicit(
        a_tri,
        b_tri,
        c_tri,
        rhs,
        water_mask,
        edge_mask,
        b_edge=b_tri_edge,
    )
    return update(
        tracer,
        at[2:-2, 2:-2, :, vs.taup1],
        npx.where(water_mask, solution, rhs),
    )


# Compatibility aliases retained for existing setup files.
setupNPZD = setup_npzd
biogeochemistry = mobi_biology
