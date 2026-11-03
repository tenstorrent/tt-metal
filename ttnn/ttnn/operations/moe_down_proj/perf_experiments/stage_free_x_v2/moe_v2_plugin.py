# pytest plugin (-p moe_v2_plugin, this dir on PYTHONPATH): MOE_VARIANT=cand|gated builds the op's program from
# <variant>/moe_down_proj_program_descriptor.py with <variant>/kernels (V2_DEFINES: comma list of kernel defines). MOE_VARIANT=base leaves the op untouched.
import importlib.util
import os
import sys
from pathlib import Path


def pytest_configure(config):
    variant = os.environ.get("MOE_VARIANT", "cand")
    if variant not in ("cand", "gated"):
        return
    import ttnn.operations.moe_down_proj  # noqa: F401

    op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
    here = Path(__file__).parent / variant
    spec = importlib.util.spec_from_file_location("moe_pd_v2", here / "moe_down_proj_program_descriptor.py")
    pd = importlib.util.module_from_spec(spec)
    sys.modules["moe_pd_v2"] = pd
    spec.loader.exec_module(pd)
    pd.KERNEL_DIR = here / "kernels"
    defines = [
        tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in os.environ.get("V2_DEFINES", "").split(",") if d
    ]

    def create(*args, **kwargs):
        desc = pd.create_program_descriptor(*args, **kwargs)
        for kernel in desc.kernels:
            kernel.defines = list(kernel.defines) + defines
        return desc

    op.create_program_descriptor = create if defines else pd.create_program_descriptor
