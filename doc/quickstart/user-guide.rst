Basic usage
===========

First, install Veros-BGC:

::

   $ pip install "veros>=1.6.2" veros-bgc

To get started with a new setup, you can use :obj:`bgc_global_4deg` as a template:

::

   $ veros copy-setup bgc_global_4deg
   $ cd bgc_global_4deg
   $ veros run bgc_global_4deg.py


To enable Veros-BGC on a new setup, you will have to register it as a Veros plugin.
Add the following to your setup definition:

::

   import veros_bgc

   class MySetup(VerosSetup):
       __veros_plugins__ = (veros_bgc,)

This registers the plugin for use with Veros.
Then, you can use :doc:`the Veros-BGC settings </reference/settings>` to configure Veros-BGC.
The most important setting is :obj:`enable_npzd`, which acts as a master switch.
Settings now live in ``state.settings``:

::

   from veros import VerosSetup, veros_routine

   class MySetup(VerosSetup):
       # ...

       @veros_routine
       def set_parameter(self, state):
           settings = state.settings
           settings.enable_npzd = True
           settings.enable_carbon = True
           settings.enable_nitrogen = True
           settings.dt_bio = settings.dt_tracer / 4

The setup must initialize ``phytoplankton``, ``zooplankton``, ``detritus``,
and ``po4`` in :meth:`set_initial_conditions`, and update surface shortwave
radiation ``swr`` in :meth:`set_forcing`. If carbon is enabled it must also
initialize ``dic``, ``alkalinity``, and ``atmospheric_co2``. See
:obj:`bgc_global_4deg` for a complete example. If nitrogen is enabled, initialize
``no3``, ``dop``, ``don``, ``diazotrophs``, and ``oxygen`` as well.
Restart files written by a phosphate-only configuration cannot be reused after
enabling nitrogen because those five prognostic fields are absent.


.. seealso::

   All new :doc:`settings </reference/settings>` and :doc:`variables </reference/variables>` defined by Veros-BGC in their respective sections.
