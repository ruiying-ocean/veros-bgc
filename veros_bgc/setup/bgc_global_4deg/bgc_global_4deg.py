"""Entry file used by ``veros copy-setup bgc_global_4deg``.

Load the adjacent implementation explicitly so a copied setup remains editable
and can be run from any working directory.
"""

import importlib.util
import sys
from pathlib import Path

_implementation = Path(__file__).with_name("bgc_global_four_degree.py")
_spec = importlib.util.spec_from_file_location(
    "_veros_bgc_global_four_degree", _implementation
)
if _spec is None or _spec.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"cannot load Veros setup implementation from {_implementation}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
GlobalFourDegreeBGC = _module.GlobalFourDegreeBGC

del _implementation, _module, _spec


__all__ = ("GlobalFourDegreeBGC",)
