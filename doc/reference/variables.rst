.. _variables:

Available variables
===================

Attributes:
  | :fa:`clock-o`: Time-dependent
  | :fa:`repeat`: Written to restart files

Variables whose ``active`` predicate depends on ``enable_npzd``,
``enable_nitrogen``, or ``enable_carbon`` are allocated after model settings
have been configured.

.. exec::
  from veros_bgc.variables import VARIABLES
  for key, var in VARIABLES.items():
      flags = ""
      if var.time_dependent:
          flags += ":fa:`clock-o` "
      if var.write_to_restart:
          flags += ":fa:`repeat` "
      dimensions = "scalar" if var.dims is None else ", ".join(var.dims)
      print(".. py:attribute:: VerosState.variables.{}".format(key))
      print("")
      print("  :units: {}".format(var.units))
      print("  :dimensions: {}".format(dimensions))
      print("  :type: :py:class:`{}`".format(var.dtype or "float"))
      print("  :attributes: {}".format(flags))
      print("")
      print("  {}".format(var.long_description))
      print("")
