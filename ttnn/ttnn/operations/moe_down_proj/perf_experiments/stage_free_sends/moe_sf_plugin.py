# pytest plugin (-p moe_sf_plugin, this dir on PYTHONPATH): with MOE_VARIANT=sf the op builds its program from pd_sf
# (kernels_sf, knobs SF_X / SF_W / SF_XD / SF_WD / SF_KMAX / SF_X_ISSUE_STEPS / SF_W_ISSUE_STEPS). MOE_VARIANT=base
# leaves the op untouched. MOE_VARIANT=grad: grad/ (the clean x-only graduation candidate, X_DEPTH 3).
import importlib.util
import os
import sys
from pathlib import Path


def pytest_configure(config):
    variant = os.environ.get("MOE_VARIANT", "sf")
    if variant not in ("sf", "grad"):
        return
    import ttnn.operations.moe_down_proj  # noqa: F401

    op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
    here = Path(__file__).parent
    path = here / "pd_sf.py" if variant == "sf" else here / "grad" / "moe_down_proj_program_descriptor.py"
    spec = importlib.util.spec_from_file_location("moe_pd_sf", path)
    pd_sf = importlib.util.module_from_spec(spec)
    sys.modules["moe_pd_sf"] = pd_sf
    spec.loader.exec_module(pd_sf)
    if variant == "grad":
        pd_sf.KERNEL_DIR = here / "grad" / "kernels"
    op.create_program_descriptor = pd_sf.create_program_descriptor
