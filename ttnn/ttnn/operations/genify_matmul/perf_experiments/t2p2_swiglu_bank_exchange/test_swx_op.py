# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""t2p2_swiglu_bank_exchange bench: the op end to end on its perf cases, variant `base` = the op's own descriptor and
kernels (untouched), `swx` = this dir's pd_swx.py + kernels/ (BANK_LOCAL_SWIGLU_XCHG). Collected through the shim in
tests/ttnn/unit_tests/operations/genify_matmul/perf_experiments/t2p2_swiglu_bank_exchange/.

env: SWX_CASES="q128_gu,..."  SWX_VARIANTS="base,swx"  SWX_KNOBS="NAME=value,..." (applied to the variant's module)
     SWX_CHECK=1 (PCC / rel-RMS vs torch, and bit / ULP equality of every variant vs the first variant of its case)
Prints SWXPLAN (geometry xchg, grid, blocking) and SWXCHECK lines."""
import ast
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import ttnn
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

HERE = Path(__file__).resolve().parent

# id: (m, k, n, A dtype, B dtype, out dtype, fidelity, epilogue, A in L1) -- r15 / r16 perf cases at their config
CASES = {
    "q128_gu": (128, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", True),
    "f256_ff1": (256, 6144, 18432, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "swiglu", False),
    "q512_gu": (512, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", False),
    "q128_down": (128, 4352, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", True),
    "q128_in": (128, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none", True),
    "q256_gu": (256, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", True),
    "d32_gu": (32, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", True),
    "g2048_gu": (2048, 5120, 8704, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", True),  # guard PERF FOCUS 1
    "s_ff1": (2304, 6144, 18432, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "swiglu", False),  # r11
    # small swiglu cells (unit-style shapes) for correctness
    "s64_gu": (64, 1024, 2048, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "swiglu", False),
    "s128_gu_fp32o": (128, 2048, 4096, "bfloat16", "bfloat8_b", "float32", "HiFi2", "swiglu", False),
    "s32_gu_b16": (32, 512, 1024, "bfloat16", "bfloat16", "bfloat16", "HiFi4", "swiglu", False),
}
SEL = [c for c in os.environ.get("SWX_CASES", "q128_gu").split(",") if c]
VARIANTS = [v for v in os.environ.get("SWX_VARIANTS", "base,swx").split(",") if v]
_outs = {}


def _module(variant):
    if variant == "base":
        return importlib.import_module("ttnn.operations.genify_matmul.genify_matmul_program_descriptor")
    spec = importlib.util.spec_from_file_location("_t2p2_pd_swx", HERE / "pd_swx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case_id", SEL)
def test_swx_op(case_id, variant, device, monkeypatch):
    import torch  # local: ttnn's global-torch-import check

    gm = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")
    mod = _module(variant)
    for item in filter(None, (s.strip() for s in os.environ.get("SWX_KNOBS", "").split(","))):
        name, value = item.split("=", 1)
        assert hasattr(mod, name), name
        monkeypatch.setattr(mod, name, ast.literal_eval(value))
    scale = float(os.environ.get("SWX_BUDGET_SCALE", "1"))
    if scale != 1:  # test hook: a smaller L1 budget (multi-M-block walks)
        orig_budget = mod.l1_cb_budget
        monkeypatch.setattr(mod, "l1_cb_budget", lambda d, t: int(orig_budget(d, t) * scale))
    orig_plan = mod.plan

    def _plan(*args, **kwargs):
        geom, grid, blk, gxy = orig_plan(*args, **kwargs)
        print(f"SWXPLAN {case_id}/{variant} xchg={getattr(geom, 'xchg', False)} {grid} {blk}")
        return geom, grid, blk, gxy

    monkeypatch.setattr(mod, "plan", _plan)
    monkeypatch.setattr(gm, "create_program_descriptor", mod.create_program_descriptor)

    m, k, n, a_dt, w_dt, o_dt, fid, epilogue, a_l1 = CASES[case_id]
    torch.manual_seed(0)
    A = torch.randn((1, m, k))
    B = torch.randn(k, n) / math.sqrt(k)
    a_mc = ttnn.L1_MEMORY_CONFIG if a_l1 else ttnn.DRAM_MEMORY_CONFIG
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT, memory_config=a_mc)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    kwargs = {"activation": "swiglu"} if epilogue == "swiglu" else {}
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = genify_matmul(
        ta, tb, dtype=getattr(ttnn, o_dt), compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs
    )
    ttnn.synchronize_device(device)
    if os.environ.get("SWX_CHECK") != "1":
        return
    got = ttnn.to_torch(out)
    ref = ttnn.to_torch(ta).float() @ ttnn.to_torch(tb).float()
    if epilogue == "swiglu":
        t = ref.reshape(*ref.shape[:-1], n // 64, 2, 32)
        ref = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(*ref.shape[:-1], n // 2)
    g = got.float()
    pcc = torch.corrcoef(torch.stack([g.flatten(), ref.flatten()]))[0, 1].item()
    rms = ((g - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    line = f"SWXCHECK {case_id}/{variant} pcc={pcc:.6f} rel_rms={rms:.6f}"
    if variant == VARIANTS[0]:
        _outs[case_id] = (variant, got)  # (a collect pass may have run in this process before: overwrite)
    first = _outs[case_id]
    if first[0] != variant:
        a, b = first[1], got
        if a.dtype == torch.bfloat16:
            ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
        else:
            ia, ib = a.float().view(torch.int32).long(), b.float().view(torch.int32).long()
        same_sign = (ia < 0) == (ib < 0)
        ulp = torch.where(same_sign, (ia - ib).abs(), torch.full_like(ia, 1 << 30))
        line += (
            f" vs_{first[0]}: mismatches={int((ia != ib).sum())}/{ia.numel()} max_ulp={int(ulp.max())}"
            f" max_abs={float((a.float() - b.float()).abs().max()):.3e}"
        )
    print(line)
    assert pcc > 0.999, pcc
