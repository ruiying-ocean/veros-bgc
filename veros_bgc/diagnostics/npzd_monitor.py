"""Conservation diagnostics for MOBI tracers."""

from veros import logger
from veros.core.operators import numpy as npx
from veros.diagnostics.base import VerosDiagnostic
from veros.distributed import global_sum
from veros.variables import Variable


class NPZDMonitor(VerosDiagnostic):
    """Monitor global phosphorus, nitrogen, and carbon inventories."""

    name = "npzd"
    output_frequency = None
    sampling_frequency = 0.0

    def __init__(self, state=None):
        # Current Veros instantiates plugin diagnostics without passing state;
        # accepting it remains useful for forwards compatibility.
        self.var_meta = {
            "po4_total": Variable(
                "Previous phosphorus inventory", None, "mmol P", write_to_restart=True
            ),
            "dic_total": Variable(
                "Previous carbon inventory", None, "mmol C", write_to_restart=True
            ),
            "nitrogen_total": Variable(
                "Previous nitrogen inventory", None, "mmol N", write_to_restart=True
            ),
        }

    def initialize(self, state):
        self.initialize_variables(state)
        phosphorus, nitrogen, carbon = _global_inventories(state)
        self.variables.po4_total = phosphorus
        self.variables.nitrogen_total = nitrogen
        self.variables.dic_total = carbon

    def diagnose(self, state):
        pass

    def output(self, state):
        phosphorus, nitrogen, carbon = _global_inventories(state)

        old_phosphorus = self.variables.po4_total
        phosphorus_change = npx.where(
            old_phosphorus != 0.0,
            (phosphorus - old_phosphorus) / old_phosphorus,
            0.0,
        )
        logger.diagnostic(
            f" Total phosphorus: {phosphorus:.8e} mmol; "
            f"relative change: {phosphorus_change:.3e}"
        )
        self.variables.po4_total = phosphorus

        if state.settings.enable_nitrogen:
            old_nitrogen = self.variables.nitrogen_total
            nitrogen_change = npx.where(
                old_nitrogen != 0.0,
                (nitrogen - old_nitrogen) / old_nitrogen,
                0.0,
            )
            logger.diagnostic(
                f" Total ocean nitrogen: {nitrogen:.8e} mmol; "
                f"relative change: {nitrogen_change:.3e}"
            )
            self.variables.nitrogen_total = nitrogen

        if state.settings.enable_carbon:
            old_carbon = self.variables.dic_total
            carbon_change = npx.where(
                old_carbon != 0.0,
                (carbon - old_carbon) / old_carbon,
                0.0,
            )
            logger.diagnostic(
                f" Total ocean carbon: {carbon:.8e} mmol; "
                f"relative change: {carbon_change:.3e}"
            )
            self.variables.dic_total = carbon


def _global_inventories(state):
    vs = state.variables
    settings = state.settings

    cell_volume = (
        vs.area_t[2:-2, 2:-2, npx.newaxis]
        * vs.dzt[npx.newaxis, npx.newaxis, :]
        * vs.maskT[2:-2, 2:-2, :]
    )
    redfield_organic_nitrogen = (
        vs.phytoplankton[2:-2, 2:-2, :, vs.tau]
        + vs.zooplankton[2:-2, 2:-2, :, vs.tau]
        + vs.detritus[2:-2, 2:-2, :, vs.tau]
    )
    if settings.enable_nitrogen:
        diazotrophs = vs.diazotrophs[2:-2, 2:-2, :, vs.tau]
        phosphorus_field = (
            vs.po4[2:-2, 2:-2, :, vs.tau]
            + vs.dop[2:-2, 2:-2, :, vs.tau]
            + settings.redfield_ratio_PN * redfield_organic_nitrogen
            + diazotrophs / settings.diazotroph_NP_ratio
        )
        nitrogen_field = (
            vs.no3[2:-2, 2:-2, :, vs.tau]
            + vs.don[2:-2, 2:-2, :, vs.tau]
            + redfield_organic_nitrogen
            + diazotrophs
        )
        carbon_organic_nitrogen = (
            redfield_organic_nitrogen + diazotrophs + vs.don[2:-2, 2:-2, :, vs.tau]
        )
    else:
        phosphorus_field = (
            vs.po4[2:-2, 2:-2, :, vs.tau]
            + settings.redfield_ratio_PN * redfield_organic_nitrogen
        )
        nitrogen_field = npx.zeros_like(phosphorus_field)
        carbon_organic_nitrogen = redfield_organic_nitrogen

    phosphorus = global_sum(npx.sum(phosphorus_field * cell_volume))
    nitrogen = global_sum(npx.sum(nitrogen_field * cell_volume))

    if settings.enable_carbon:
        carbon = global_sum(
            npx.sum(
                (
                    vs.dic[2:-2, 2:-2, :, vs.tau]
                    + settings.redfield_ratio_CN * carbon_organic_nitrogen
                )
                * cell_volume
            )
        )
    else:
        carbon = npx.asarray(0.0)

    return phosphorus, nitrogen, carbon
