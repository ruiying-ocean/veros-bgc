try:
    import veros  # noqa: F401
except ImportError:
    raise RuntimeError(
        "veros-bgc needs Veros to be installed (run `uv sync`)"
    )

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("veros-bgc")
except PackageNotFoundError:
    __version__ = "0.1.4.dev0"

from veros_bgc.core.npzd import npzd, setup_npzd  # noqa: E402
from veros_bgc.diagnostics.npzd_monitor import NPZDMonitor  # noqa: E402
from veros_bgc.settings import SETTINGS  # noqa: E402
from veros_bgc.variables import VARIABLES  # noqa: E402

__VEROS_INTERFACE__ = dict(
    name="biogeochemistry",
    setup_entrypoint=setup_npzd,
    run_entrypoint=npzd,
    settings=SETTINGS,
    variables=VARIABLES,
    dimensions={},
    diagnostics=[NPZDMonitor],
)
