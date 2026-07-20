"""Air-sea carbon exchange and MOBI carbonate chemistry.

The carbonate-system expressions are adapted from ``co2calc_SWS`` in the
UVic/MOBI source.  The original backend-specific root finder has been replaced
by a fixed-iteration safeguarded Newton method, which works with both NumPy and
JAX backends.
"""

from veros import KernelOutput, veros_kernel
from veros.core import utilities
from veros.core.operators import numpy as npx


@veros_kernel
def carbon_flux(state):
    """Calculate air-sea CO2 exchange at the surface.

    The returned flux is positive into the ocean and has units of
    ``mmol C m-2 s-1``.
    """

    vs = state.variables
    settings = state.settings

    temperature = vs.temp[:, :, -1, vs.tau]
    salinity = vs.salt[:, :, -1, vs.tau]
    surface_mask = vs.maskT[:, :, -1]

    chemistry = carbonate_system(
        temperature,
        salinity,
        vs.dic[:, :, -1, vs.tau] * 1e-3,
        vs.alkalinity[:, :, -1, vs.tau] * 1e-3,
        vs.atmospheric_co2,
        settings.atmospheric_pressure,
        vs.hSWS,
        surface_mask,
    )

    # Veros currently has no explicit atmosphere/sea-ice component in this
    # setup.  This is the same diagnostic ice mask used by the physical 4deg
    # setup and by the previous Veros-BGC implementation.
    ice_mask = npx.logical_and(
        temperature * surface_mask < -1.8, vs.forc_temp_surface <= 0.0
    )
    open_water = npx.logical_and(surface_mask, npx.logical_not(ice_mask))

    # Keep the legacy UVic/Veros-BGC wind-stress conversion for reproducibility.
    wind_speed = settings.wind_speed_from_stress_factor * npx.sqrt(
        npx.abs(vs.surface_taux / settings.rho_0)
        + npx.abs(vs.surface_tauy / settings.rho_0)
    )

    # Schmidt number for CO2 (Wanninkhof, 1992).
    schmidt = (
        2073.1
        - 125.62 * temperature
        + 3.6276 * temperature**2
        - 0.043219 * temperature**3
    )
    schmidt = npx.maximum(schmidt, 1.0)
    piston_velocity = (
        open_water
        * settings.co2_transfer_coefficient
        * wind_speed**2
        * (schmidt / 660.0) ** -0.5
    )

    cflux = piston_velocity * chemistry.dco2star * 1e3
    cflux = utilities.enforce_boundaries(cflux, settings.enable_cyclic_x)

    return KernelOutput(
        cflux=cflux,
        wind_speed=wind_speed,
        hSWS=chemistry.hSWS,
        pCO2=chemistry.pCO2,
        dpCO2=chemistry.dpCO2,
        co2star=chemistry.co2star,
        dco2star=chemistry.dco2star,
    )


@veros_kernel
def carbonate_system(
    temperature,
    salinity,
    dic_in,
    alkalinity_in,
    atmospheric_co2,
    atmospheric_pressure,
    hydrogen_guess,
    mask,
):
    """Solve the surface seawater carbonate system.

    ``dic_in`` and ``alkalinity_in`` are in mol/m3, atmospheric CO2 is in
    ppmv, and the hydrogen-ion concentration is on the seawater scale.
    """

    salinity = npx.maximum(salinity, 0.0)
    temperature_kelvin = npx.maximum(temperature + 273.15, 200.0)

    sit_in = npx.full_like(temperature, 7.6875e-3)
    pt_in = npx.full_like(temperature, 0.5125e-3)

    # Convert mol/m3 to mol/kg using MOBI's fixed surface density.
    permil = 1.0 / 1024.5
    pt = pt_in * permil
    sit = sit_in * permil
    ta = alkalinity_in * permil
    dic = dic_in * permil

    permeg = 1e-6
    co2 = atmospheric_co2 * permeg
    scl = salinity / 1.80655
    ionic_strength = 19.924 * salinity / npx.maximum(1000.0 - 1.005 * salinity, 1.0)

    bt = 0.000232 * scl / 10.811
    st = 0.14 * scl / 96.062
    ft = 0.000067 * scl / 18.9984

    t100 = temperature_kelvin / 100.0
    ff = npx.exp(
        -162.8301
        + 218.2968 / t100
        + 90.9241 * npx.log(t100)
        - 1.47696 * t100**2
        + salinity * (0.025695 - 0.025225 * t100 + 0.0049867 * t100**2)
    )
    k0 = npx.exp(
        93.4517 / t100
        - 60.2409
        + 23.3585 * npx.log(t100)
        + salinity * (0.023517 - 0.023656 * t100 + 0.0047036 * t100**2)
    )

    rt_x = 83.1451 * temperature_kelvin
    delta_x = 57.7 - 0.118 * temperature_kelvin
    b_x = (
        -1636.75
        + 12.0408 * temperature_kelvin
        - 0.0327957 * temperature_kelvin**2
        + 3.16528e-5 * temperature_kelvin**3
    )
    fugacity_factor = npx.exp((b_x + 2.0 * delta_x) / rt_x)

    k1 = 10.0 ** -(
        3670.7 / temperature_kelvin
        - 62.088
        + 9.7944 * npx.log(temperature_kelvin)
        - 0.0118 * salinity
        + 0.000116 * salinity**2
    )
    k2 = 10.0 ** -(
        1394.7 / temperature_kelvin + 4.777 - 0.0184 * salinity + 0.000118 * salinity**2
    )
    k1p = npx.exp(
        -4576.752 / temperature_kelvin
        + 115.540
        - 18.453 * npx.log(temperature_kelvin)
        + (-106.736 / temperature_kelvin + 0.69171) * npx.sqrt(salinity)
        + (-0.65643 / temperature_kelvin - 0.01844) * salinity
    )
    k2p = npx.exp(
        -8814.715 / temperature_kelvin
        + 172.1033
        - 27.927 * npx.log(temperature_kelvin)
        + (-160.340 / temperature_kelvin + 1.3566) * npx.sqrt(salinity)
        + (0.37335 / temperature_kelvin - 0.05778) * salinity
    )
    k3p = npx.exp(
        -3070.75 / temperature_kelvin
        - 18.126
        + (17.27039 / temperature_kelvin + 2.81197) * npx.sqrt(salinity)
        + (-44.99486 / temperature_kelvin - 0.09984) * salinity
    )
    ksi = npx.exp(
        -8904.2 / temperature_kelvin
        + 117.400
        - 19.334 * npx.log(temperature_kelvin)
        + (-458.79 / temperature_kelvin + 3.5913) * npx.sqrt(ionic_strength)
        + (188.74 / temperature_kelvin - 1.5998) * ionic_strength
        + (-12.1652 / temperature_kelvin + 0.07871) * ionic_strength**2
        + npx.log(npx.maximum(1.0 - 0.001005 * salinity, 1e-12))
    )
    kw = npx.exp(
        -13847.26 / temperature_kelvin
        + 148.9802
        - 23.6521 * npx.log(temperature_kelvin)
        + (118.67 / temperature_kelvin - 5.977 + 1.0495 * npx.log(temperature_kelvin))
        * npx.sqrt(salinity)
        - 0.01615 * salinity
    )
    ks = npx.exp(
        -4276.1 / temperature_kelvin
        + 141.328
        - 23.093 * npx.log(temperature_kelvin)
        + (
            -13856.0 / temperature_kelvin
            + 324.57
            - 47.986 * npx.log(temperature_kelvin)
        )
        * npx.sqrt(ionic_strength)
        + (
            35474.0 / temperature_kelvin
            - 771.54
            + 114.723 * npx.log(temperature_kelvin)
        )
        * ionic_strength
        - 2698.0 / temperature_kelvin * ionic_strength**1.5
        + 1776.0 / temperature_kelvin * ionic_strength**2
        + npx.log(npx.maximum(1.0 - 0.001005 * salinity, 1e-12))
    )
    kf = npx.exp(
        1590.2 / temperature_kelvin
        - 12.641
        + 1.525 * npx.sqrt(ionic_strength)
        + npx.log(npx.maximum(1.0 - 0.001005 * salinity, 1e-12))
    )
    kb = npx.exp(
        (
            -8966.90
            - 2890.53 * npx.sqrt(salinity)
            - 77.942 * salinity
            + 1.728 * salinity**1.5
            - 0.0996 * salinity**2
        )
        / temperature_kelvin
        + 148.0248
        + 137.1942 * npx.sqrt(salinity)
        + 1.62142 * salinity
        + (-24.4344 - 25.085 * npx.sqrt(salinity) - 0.2474 * salinity)
        * npx.log(temperature_kelvin)
        + 0.053105 * npx.sqrt(salinity) * temperature_kelvin
        + npx.log((1.0 + st / ks + ft / kf) / (1.0 + st / ks))
    )

    args = (k1, k2, k1p, k2p, k3p, st, ks, kf, ft, dic, ta, sit, ksi, pt, bt, kw, kb)
    hydrogen = _solve_hydrogen(hydrogen_guess, args)
    hydrogen = npx.where(mask, hydrogen, hydrogen_guess)

    co2star_kg = dic * hydrogen**2 / (hydrogen**2 + k1 * hydrogen + k1 * k2)
    co2star_air_kg = co2 * ff * atmospheric_pressure
    dco2star_kg = co2star_air_kg - co2star_kg

    pco2 = co2star_kg / (k0 * fugacity_factor)
    dpco2 = pco2 - co2 * atmospheric_pressure

    return KernelOutput(
        hSWS=hydrogen,
        co2star=co2star_kg / permil,
        dco2star=dco2star_kg / permil,
        pCO2=pco2 / permeg,
        dpCO2=dpco2 / permeg,
    )


def _solve_hydrogen(initial, args):
    """Safeguarded, fixed-iteration Newton solve for hydrogen concentration."""

    lower = npx.full_like(initial, 1e-10)
    upper = npx.full_like(initial, 1e-6)
    value = npx.clip(initial, lower, upper)
    f_lower, _ = alkalinity_residual(lower, *args)

    # A fixed iteration count is deliberate: it keeps this routine JIT-safe and
    # converges the 1e-10--1e-6 bracket well below double-precision needs.
    for _ in range(24):
        residual, derivative = alkalinity_residual(value, *args)
        same_side = f_lower * residual > 0.0
        lower = npx.where(same_side, value, lower)
        f_lower = npx.where(same_side, residual, f_lower)
        upper = npx.where(same_side, upper, value)

        midpoint = 0.5 * (lower + upper)
        newton = value - residual / derivative
        use_newton = npx.logical_and(
            npx.isfinite(newton),
            npx.logical_and(newton > lower, newton < upper),
        )
        value = npx.where(use_newton, newton, midpoint)

    return npx.clip(value, lower, upper)


def alkalinity_residual(
    x, k1, k2, k1p, k2p, k3p, st, ks, kf, ft, dic, ta, sit, ksi, pt, bt, kw, kb
):
    """Return the total-alkalinity residual and its derivative."""

    x2 = x * x
    x3 = x2 * x
    k12 = k1 * k2
    k12p = k1p * k2p
    k123p = k12p * k3p
    c = 1.0 + st / ks + ft / kf
    a = x3 + k1p * x2 + k12p * x + k123p
    a2 = a * a
    da = 3.0 * x2 + 2.0 * k1p * x + k12p
    b = x2 + k1 * x + k12
    b2 = b * b
    db = 2.0 * x + k1

    residual = (
        k1 * x * dic / b
        + 2.0 * dic * k12 / b
        + bt / (1.0 + x / kb)
        + kw / x
        + pt * k12p * x / a
        + 2.0 * pt * k123p / a
        + sit / (1.0 + x / ksi)
        - x / c
        - st / (1.0 + ks / (x / c))
        - ft / (1.0 + kf / (x / c))
        - pt * x3 / a
        - ta
    )
    derivative = (
        (k1 * dic * b - k1 * x * dic * db) / b2
        - 2.0 * dic * k12 * db / b2
        - bt / kb / (1.0 + x / kb) ** 2
        - kw / x2
        + pt * k12p * (a - x * da) / a2
        - 2.0 * pt * k123p * da / a2
        - sit / ksi / (1.0 + x / ksi) ** 2
        - 1.0 / c
        - st * (1.0 + ks / (x / c)) ** -2 * (ks * c / x2)
        - ft * (1.0 + kf / (x / c)) ** -2 * (kf * c / x2)
        - pt * x2 * (3.0 * a - x * da) / a2
    )

    return residual, derivative


# Backwards-compatible spelling used in Veros-BGC 0.1.x.
co2calc_SWS = carbonate_system
