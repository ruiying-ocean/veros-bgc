Model implementation
====================

Scientific scope
----------------

The reference implementation is ``mobi_src/updates/npzd_src.F``. Veros-BGC
currently follows the code selected by these Fortran options:

* ``O_npzd``: phosphate, phytoplankton, zooplankton, and sinking detritus;
* ``O_npzd_nitrogen`` and ``O_npzd_o2``: nitrate, dissolved organic N/P,
  diazotrophs, oxygen coupling, nitrogen fixation, and denitrification;
* ``O_carbon`` and ``O_npzd_alk``: DIC and total alkalinity;
* the non-``O_npzd_caco3`` branch: implicit calcite production and
  depth-dependent dissolution.

The full reference configuration in ``mobi_src/mk.in`` additionally enables
iron, prognostic calcite and coccolithophores, carbon-13, and nitrogen-15.
Those branches require additional prognostic tracers and are outside the
current plugin scope.

Veros coupling
--------------

Plugin settings and variables are declared in :mod:`veros_bgc.settings` and
:mod:`veros_bgc.variables`. Conditional variables use ``Variable.active`` and
are materialized after :meth:`VerosSetup.set_parameter`.

Each tracer step performs the following operations:

#. Diagnose air--sea CO2 exchange when carbon is enabled.
#. Integrate MOBI source and sink terms with ``dt_bio`` substeps.
#. Advect every prognostic tracer with Veros' configured tracer advection.
#. Apply harmonic or biharmonic horizontal diffusion when enabled.
#. Apply neutral and skew diffusion when enabled.
#. Apply implicit vertical mixing with Veros' ``kappaH``.
#. Add the finite biogeochemical increment, enforce positivity, and exchange
   cyclic boundaries.

All numerical kernels use :mod:`veros.core.operators`; no in-place NumPy
mutation is used. The same implementation therefore runs with NumPy and JAX.

Extending the ecosystem
-----------------------

The pre-Veros-1.x object/rule API stored Python objects and dictionaries in the
model state. That representation cannot be traced by JAX and has been retired.
To add a prognostic tracer now:

#. Add its metadata and Adams--Bashforth tendency to ``VARIABLES``.
#. Add its source/sink equation to :func:`veros_bgc.core.npzd.mobi_biology`.
#. Pass the tracer and its tendency through
   :func:`veros_bgc.core.npzd.transport_tracer`.
#. Add conservation and positivity tests for both NumPy and JAX backends.

Keep settings as scalar, immutable model configuration. Time-varying arrays
belong in ``state.variables`` and kernel changes must be returned explicitly
through ``KernelOutput``.
