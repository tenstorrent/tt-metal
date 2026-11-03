# pytest plugin (-p moe_tail_plugin, this dir on PYTHONPATH): with MOE_VARIANT=tail, the op builds its program from
# pd_tail (kernels_tail: last-wave store split across both dataflow RISCs); MOE_VARIANT=grad: pd_grad + kernels_grad
# (the clean graduation candidate). Used with the probes test_perf_bench.py.
import importlib.util
import os
from pathlib import Path


def pytest_configure(config):
    variant = os.environ.get("MOE_VARIANT", "tail")
    if variant not in ("tail", "grad"):
        return
    import sys

    import ttnn.operations.moe_down_proj  # noqa: F401  (the package re-exports the function under the module name)

    op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]

    spec = importlib.util.spec_from_file_location(f"moe_pd_{variant}", Path(__file__).parent / f"pd_{variant}.py")
    pd_tail = importlib.util.module_from_spec(spec)
    sys.modules[f"moe_pd_{variant}"] = pd_tail
    spec.loader.exec_module(pd_tail)
    if variant == "grad":
        pd_tail.KERNEL_DIR = Path(__file__).parent / "kernels_grad"
    op.create_program_descriptor = pd_tail.create_program_descriptor
    print("MOE_TAIL_PLUGIN patched", op, op.__file__)
