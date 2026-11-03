# early_ack_handshake bench: the probe bench (focus + guard cases) with a kernel-dir override, preprocessor defines,
# and one extra data-ready semaphore per channel (xa/xb/wa/wb "_dr1") attached to every core of the program.
#   MOE_CASES, MOE_CHECK as in tests/.../probes/test_perf_bench.py
#   EA_KDIR   kernels dir (default: ./kernels)     EA_DEFINES  comma list (EARLY_ACK_X, EARLY_ACK_W)
import os
import sys
from pathlib import Path

import pytest
import ttnn

from ttnn.operations.moe_down_proj import moe_down_proj_program_descriptor as pd
from ttnn.operations.moe_down_proj import moe_down_proj
from tests.ttnn.unit_tests.operations.moe_down_proj.probes.test_perf_bench import CASES, MODEL_FREE, L1, occupy
from tests.ttnn.unit_tests.operations.moe_down_proj.test_moe_down_proj import check, make_operands, rows_for

HERE = Path(__file__).parent
SELECTED = os.environ.get("MOE_CASES", "focus").split(",")
DR1_NAMES = ("xa_dr1", "xb_dr1", "wa_dr1", "wb_dr1")


def _patched(orig, defines):
    def create(*args, **kwargs):
        desc = orig(*args, **kwargs)
        cores = desc.kernels[0].core_ranges
        next_id = max((s.id for s in desc.semaphores), default=-1) + 1
        sems = list(desc.semaphores)
        named = []
        for i, name in enumerate(DR1_NAMES):
            sems.append(ttnn.SemaphoreDescriptor(id=next_id + i, core_ranges=cores, initial_value=0))
            named.append((name, next_id + i))
        desc.semaphores = sems
        for kernel in desc.kernels:
            kernel.named_compile_time_args = list(kernel.named_compile_time_args) + named
            kernel.defines = list(kernel.defines) + [(d, "1") for d in defines]
        return desc

    return create


@pytest.mark.parametrize("case", SELECTED, ids=SELECTED)
def test_bench(device, case, monkeypatch):
    tokens, counts, x_mem, out_mem, occupied = CASES[case]
    monkeypatch.setattr(pd, "KERNEL_DIR", Path(os.environ.get("EA_KDIR", HERE / "kernels")))
    defines = [d for d in os.environ.get("EA_DEFINES", "").split(",") if d]
    create = _patched(pd.create_program_descriptor, defines)
    monkeypatch.setattr(pd, "create_program_descriptor", create)
    op = sys.modules["ttnn.operations.moe_down_proj.moe_down_proj"]
    if hasattr(op, "create_program_descriptor"):
        monkeypatch.setattr(op, "create_program_descriptor", create)
    device.disable_and_clear_program_cache()
    x_dev, w_dev, counts_dev, xq, wq = make_operands(device, tokens, counts, x_mem)
    filler = None
    if occupied:
        out_bank = 0 if out_mem.buffer_type != ttnn.BufferType.L1 else pd._div_up(rows_for(tokens), 110) * 1536
        filler = occupy(device, MODEL_FREE + out_bank)
    out = moe_down_proj(x_dev, w_dev, counts_dev, memory_config=out_mem)
    if filler is not None:
        ttnn.deallocate(filler)
    if os.environ.get("MOE_CHECK", "1") == "1":
        check(out, xq, wq, counts, out_mem)
