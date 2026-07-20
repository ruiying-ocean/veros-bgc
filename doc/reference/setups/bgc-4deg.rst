Global four-degree model with BioGeoChemistry
---------------------------------------------

.. autoclass:: veros_bgc.setup.bgc_global_4deg.GlobalFourDegreeBGC

The setup enables MOBI's nitrogen cycle and oxygen coupling. Its kinetic,
stoichiometric, and sinking parameters retain the defaults in the Fortran
``mobi_init`` routine, including a maximum phytoplankton growth rate of 0.6 per
day, ``bbio = 1.066``, and a surface detritus sinking speed of 16 m per day.
