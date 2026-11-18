# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf bake-off t2_msplit_b_resident: walk each core's rows as M-blocks with B L1-resident (read once, during the
first M-block), so M-block i's output writes overlap M-block i+1's K-walk.

Runs a COPY of the op (pd_copy.py + kernels/ in this dir) by swapping the op module's create_program_descriptor; the
real op is untouched. Variant "base" runs the REAL op. Every variant runs on the SAME inputs in one process; outputs
must be bit-identical to the first variant's, and each is checked against torch (PCC, rel-RMS).

Env:
  MS_CASE      case name (CASES below)
  MS_VARIANTS  comma list; "base" = the real op, else '+'-joined knob items of the copy:
               h=12-4 (MSPLIT_HEIGHTS), d=2 (MSPLIT_DEPTH_ACT), bk=5 (MSPLIT_BLOCK_K), ah=2 (MSPLIT_B_AHEAD),
               inj=2 (MSPLIT_INJECTORS), abl=SKIP_COMPUTE.GM_ABLATE_B (ABLATE_DEFINES, '.'-joined),
               copy (the copy with MSPLIT off: a no-op control)
Skipped unless GENIFY_MATMUL_PERF=1.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.genify_matmul.genify_matmul  # noqa: F401
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
import ttnn.operations.genify_matmul.genify_matmul_program_descriptor as pd_real

op_mod = sys.modules["ttnn.operations.genify_matmul.genify_matmul"]
pytestmark = pytest.mark.skipif(os.environ.get("GENIFY_MATMUL_PERF") != "1", reason="perf bench")
EXP = Path(__file__).resolve().parent

# name -> (m, k, n, A dtype, B dtype, out dtype, fidelity, epilogue)
CASES = {
    "te_gate": (512, 5120, 16384, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", "none"),
    "te_down": (512, 16384, 5120, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", "none"),
    "te_qkv": (512, 5120, 3072, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", "none"),
    "te_o": (512, 4096, 2560, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", "none"),
    "q512_in": (512, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q512_down": (512, 4352, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q512_qkvg": (512, 5120, 3584, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q512_out": (512, 1536, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q1024_out": (1024, 1536, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q1024_qkvg": (1024, 5120, 3584, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q1024_in": (1024, 5120, 4160, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q1024_down": (1024, 4352, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "q2048_out": (2048, 1536, 5120, "bfloat16", "bfloat4_b", "bfloat16", "LoFi", "none"),
    "f256_qkv": (256, 6144, 9216, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none"),
    "f256_ff2": (256, 9216, 6144, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none"),
    "f256_ctx": (256, 15360, 3072, "bfloat16", "bfloat16", "bfloat16", "HiFi2", "none"),
    # small correctness-only cells (<= 32 tiles per operand would be too small for R1b; these are mid cells)
    "small_a": (256, 2048, 2048, "bfloat16", "bfloat8_b", "bfloat16", "HiFi2", "none"),
}


def _load_copy():
    spec = importlib.util.spec_from_file_location("ms_pd_copy", EXP / "pd_copy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ms_pd_copy"] = mod
    spec.loader.exec_module(mod)
    assert mod.KERNEL_DIR == EXP / "kernels"
    return mod


def _apply(mod, variant):
    mod.MSPLIT_HEIGHTS = None
    mod.MSPLIT_DEPTH_ACT = 2
    mod.MSPLIT_BLOCK_K = None
    mod.MSPLIT_B_AHEAD = 2
    mod.MSPLIT_INJECTORS = None
    mod.ABLATE_DEFINES = ""
    mod.MSPLIT_SAME_ROUND = True
    mod.MSPLIT_OUT_DEPTH = 2
    mod.MSPLIT_B_PACE = 0
    mod.MSPLIT_TAIL_PCT = 100
    for item in filter(None, variant.split("+")):
        if item == "copy":
            continue
        key, _, val = item.partition("=")
        if key == "h":
            mod.MSPLIT_HEIGHTS = "auto" if val == "auto" else tuple(int(v) for v in val.split("-"))
        elif key == "d":
            mod.MSPLIT_DEPTH_ACT = int(val)
        elif key == "bk":
            mod.MSPLIT_BLOCK_K = int(val)
        elif key == "ah":
            mod.MSPLIT_B_AHEAD = int(val)
        elif key == "inj":
            mod.MSPLIT_INJECTORS = int(val)
        elif key == "tp":
            mod.MSPLIT_TAIL_PCT = int(val)
        elif key == "pace":
            mod.MSPLIT_B_PACE = int(val)
        elif key == "od":
            mod.MSPLIT_OUT_DEPTH = int(val)
        elif key == "sr":
            mod.MSPLIT_SAME_ROUND = bool(int(val))
        elif key == "abl":
            mod.ABLATE_DEFINES = val.replace(".", "+")
        else:
            raise ValueError(item)


def test_msplit(device, monkeypatch):
    variants = [v for v in os.environ.get("MS_VARIANTS", "base,h=12-4").split(",") if v]
    mod = _load_copy()
    for case in os.environ.get("MS_CASE", "te_gate").split(","):
        _run_case(case, variants, device, monkeypatch, mod)


def _run_case(case, variants, device, monkeypatch, mod):
    import torch  # local: the ttnn tree forbids a global torch import

    m, k, n, a_dt, w_dt, o_dt, fid, epilogue = CASES[case]
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    ref = A.float() @ ttnn.to_torch(tb).float()
    if os.environ.get("MS_PLAN_ONLY") == "1":
        _apply(mod, next((v for v in variants if v != "base"), "copy"))
        out_t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        geom, grid, blk, _ = mod.plan(ta, tb, out_t, True, device, math_fidelity=getattr(ttnn.MathFidelity, fid))
        mode = mod.operand_mode(geom, grid, blk)
        ms = None
        if mode == 2 and mod.MSPLIT_HEIGHTS:
            budget = mod.l1_cb_budget(device, [ta, tb, out_t])
            ms = mod.msplit_blocking(geom, grid, blk, budget)
        print(
            f"MS_PLAN case={case} mode={mode} grid={grid} blk={blk} msplit={ms and (ms[1], ms[3], ms[0].block_k_tiles, ms[0].depth_act, ms[2])} terms_us={[round(t / 1350, 1) for t in mod.split_cost_terms(geom, grid, blk)]}"
        )
        return
    orig_cpd = pd_real.create_program_descriptor  # the REAL op (op_mod may still hold the copy from a previous case)
    first = None
    for v in variants:
        if v == "grad":
            spec = importlib.util.spec_from_file_location(
                "ms_pd_grad", EXP / "grad" / "genify_matmul_program_descriptor.py"
            )
            gmod = importlib.util.module_from_spec(spec)
            sys.modules["ms_pd_grad"] = gmod
            spec.loader.exec_module(gmod)
            gmod.KERNEL_DIR = EXP / "kernels"  # the graduation patch's kernels (identical to kernels/ here)
            monkeypatch.setattr(op_mod, "create_program_descriptor", gmod.create_program_descriptor)
        elif v == "base":
            monkeypatch.setattr(op_mod, "create_program_descriptor", orig_cpd)
        else:
            _apply(mod, v)
            mod.MSPLIT_LOG.clear()
            monkeypatch.setattr(op_mod, "create_program_descriptor", mod.create_program_descriptor)
        plan_log = {}
        target = pd_real if v == "base" else gmod if v == "grad" else mod
        orig_plan = target.plan

        def plan(*a, _orig=orig_plan, _t=target, **kw):
            r = _orig(*a, **kw)
            plan_log["p"] = f"mode={_t.operand_mode(r[0], r[1], r[2])} grid={r[1]} blk={r[2]}"
            return r

        monkeypatch.setattr(target, "plan", plan)
        out = genify_matmul(ta, tb, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        monkeypatch.setattr(target, "plan", orig_plan)
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        ms = ""
        if v not in ("base", "grad") and mod.MSPLIT_LOG:
            h, b2, foot, budget, ks = mod.MSPLIT_LOG[-1]
            ms = (
                f" MSPLIT h={h} bk={b2.block_k_tiles} nkb={b2.num_k_blocks} sb={b2.sb_h}x{b2.sb_w} "
                f"d={b2.depth_act} ks={ks} foot={foot} budget={budget}"
            )
        elif v not in ("base", "grad") and mod.MSPLIT_HEIGHTS:
            ms = " MSPLIT=not-applied"
        same = ""
        if first is None:
            first = got
        else:
            same = f" bit_identical={bool(torch.equal(got, first))} max_abs_diff={(got - first).abs().max().item():.3g}"
        import hashlib

        digest = hashlib.md5(got.numpy().tobytes()).hexdigest()[:10]
        print(
            f"MS_RESULT case={case} variant={v} pcc={pcc:.6f} rel_rms={rms:.5f} md5={digest}{same}{ms} PLAN {plan_log.get('p')}"
        )
        assert "abl=" in v or pcc > 0.999, (v, pcc)
