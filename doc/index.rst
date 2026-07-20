:tocdepth: 5

A biogeochemistry module for Veros
==================================

Veros-BGC adds a MOBI nutrients--phytoplankton--zooplankton--detritus (NPZD)
model to `Veros <https://veros.readthedocs.io>`_. It uses the current Veros
state, plugin, diagnostic, NumPy, and JAX APIs.

The implemented scientific scope is MOBI's ``O_npzd`` branch, its coupled
``O_npzd_nitrogen`` and ``O_npzd_o2`` branches, and optional DIC, alkalinity,
air--sea CO2 exchange, and implicit calcite redistribution. Iron, prognostic
CaCO3 and coccolithophores, and isotope branches are not implemented yet.

Veros-BGC is based on `MOBI <http://people.oregonstate.edu/~schmita2/Models/MOBI/index.html>`_ by `Andreas Schmittner <http://people.oregonstate.edu/~schmita2/>`_, Oregon State University.

`Steffen Randrup <https://github.com/SteffenRandrup>`_ created Veros-BGC as a part of his `Master's thesis <https://sid.erda.dk/share_redirect/CVvcrowL22/Thesis/SteffenRandrup_MSc_thesis.pdf>`_.

.. toctree::
   :maxdepth: 2
   :caption: Usage

   quickstart/user-guide
   quickstart/developer-guide

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/setup-gallery
   reference/settings
   reference/variables
   reference/diagnostics

.. toctree::
   :maxdepth: 2
   :caption: More Information

   Veros core documentation <https://veros.readthedocs.io>
   Visit us on GitHub <https://github.com/team-ocean/veros-bgc>
