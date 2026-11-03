# pytest plugin (-p cp_plugin, this dir on PYTHONPATH) for the probes test_perf_bench.py:
#   adds case t200_rect (rect layout, core_n_tiles 3, non-trivial compute);
#   CP_SAVE=<prefix>: saves each op output to <prefix>_<case>.pt (bit-exactness compare in compare.py).
#   CF_FIDELITY=<MathFidelity name>: runs the op with that fidelity (fp32 DEST, no approx) instead of the default.
import os
import re
import sys


def pytest_configure(config):
    import torch
    import ttnn
    import ttnn.operations.moe_down_proj as pkg

    prefix = os.environ.get("CP_SAVE")
    fidelity = os.environ.get("CF_FIDELITY")
    if not prefix and not fidelity:
        return
    orig = pkg.moe_down_proj

    def saving(*args, **kwargs):
        if fidelity:
            kwargs["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
                math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True, math_approx_mode=False
            )
        out = orig(*args, **kwargs)
        if not prefix:
            return out
        case = re.search(r"\[(.*)\]", os.environ["PYTEST_CURRENT_TEST"]).group(1)
        torch.save(ttnn.to_torch(out), f"{prefix}_{case}.pt")
        return out

    pkg.moe_down_proj = saving


def pytest_collection_finish(session):
    import ttnn

    for name, mod in list(sys.modules.items()):
        if name.endswith("test_perf_bench") and hasattr(mod, "CASES"):
            mod.CASES["t200_rect"] = (
                200,
                (90, 10, 60, 70, 5, 55, 40, 70),
                ttnn.L1_MEMORY_CONFIG,
                ttnn.L1_MEMORY_CONFIG,
                False,
            )
