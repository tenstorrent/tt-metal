# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""fused_rowsum_finalize bake-off (perf-part-optimizer, rms_norm Perf 1). Not collected by a directory run.

    FRF_CELLS=a,b FRF_VARIANTS=op,0,1,3 scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/fused_rowsum_finalize/bench_fused_rowsum_finalize.py

Per cell, the SAME input tensors are dispatched once per variant, in FRF_VARIANTS order:
  op   the unmodified real op (ttnn/ttnn/operations/rms_norm) — the honest baseline
  g    graduate/ — the clean single-path candidate (what the coordinator would graduate; = variant 7)
  N    the clone in this dir (clone_descriptor.py + kernels/) with RMS_FRONT = N (see kernels/rms_norm_compute.cpp)
Each dispatch is one GenericOp row in the profiler CSV; the (cell, variant) order is appended to order.jsonl.
Correctness gate per variant: the golden check_output (dtype tolerance, perf cells' pcc_threshold) vs torch.
Also printed: PCC vs torch and max-abs vs the `op` output (identical-config comparison).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm import helpers as H
from eval.metrics import CheckOutputError, check_output
from eval.sharding import shard_config
import ttnn.operations.rms_norm  # noqa: F401  (the package re-exports the rms_norm FUNCTION over the submodule name)

op_mod = sys.modules["ttnn.operations.rms_norm.rms_norm"]

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("frf_clone_descriptor", HERE / "clone_descriptor.py")
clone = importlib.util.module_from_spec(_spec)
sys.modules["frf_clone_descriptor"] = clone  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(clone)
_gspec = importlib.util.spec_from_file_location(
    "frf_graduate_descriptor", HERE / "graduate" / "rms_norm_program_descriptor.py"
)
grad = importlib.util.module_from_spec(_gspec)
sys.modules["frf_graduate_descriptor"] = grad
_gspec.loader.exec_module(grad)
REAL_PD = op_mod.create_program_descriptor

ML = ttnn.TensorMemoryLayout
HIFI2 = ttnn.MathFidelity.HiFi2
RM = ttnn.ROW_MAJOR_LAYOUT


def _cell(
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    ml=ML.INTERLEAVED,
    fp32=True,
    gdtype=ttnn.bfloat16,
    glayout=ttnn.TILE_LAYOUT,
    extras=None,
):
    return dict(
        shape=shape, dtype=dtype, layout=layout, ml=ml, fp32=fp32, gdtype=gdtype, glayout=glayout, extras=extras or {}
    )


def _perf(shape, ml=ML.INTERLEAVED, shard=None, **kw):
    ex = {"math_fidelity": HIFI2, "pcc_threshold": 0.9995}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    return _cell(shape, ml=ml, fp32=False, extras=ex, **kw)


CELLS = {
    # targets
    "wid_tile_2048x256": _cell((1, 1, 2048, 256), ml=ML.WIDTH_SHARDED),
    "blk_perf_7168x1024": _perf((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    "blk_perf_8192x1024": _perf((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    "reg_f32_2x1x128x256": _cell((2, 1, 128, 256), dtype=ttnn.float32, gdtype=None),
    "hgt_tile_g": _cell((128, 8192), ml=ML.HEIGHT_SHARDED),
    "hgt_perf_8192x1024": _perf((1, 1, 8192, 1024), ML.HEIGHT_SHARDED),  # tall HEIGHT shard, auto geometry
    "hgt_7136x736": _cell((7136, 736), ml=ML.HEIGHT_SHARDED),  # golden resilience shape
    "hgt_perf_7136x736": _perf((7136, 736), ML.HEIGHT_SHARDED),
    # at-risk
    "blk_tail_224x1000": _cell((1, 1, 224, 1000), ml=ML.BLOCK_SHARDED),  # sharded W tail (RMS_MASK_TAIL)
    "blk_tail_224x1000_bf16dst": _cell((1, 1, 224, 1000), ml=ML.BLOCK_SHARDED, fp32=False),
    "wid_tail_64x1000": _cell((1, 1, 64, 1000), ml=ML.WIDTH_SHARDED),
    "hgt_tail_1024x1000": _cell((1, 1, 1024, 1000), ml=ML.HEIGHT_SHARDED),
    "blk_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
    "int_rm_128x4096_gf32": _cell((1, 1, 128, 4096), layout=RM, glayout=RM, gdtype=ttnn.float32),
    "int_prefill_7168": _perf((1, 1, 8192, 7168)),
    "int_prefill_1024": _perf((1, 1, 8192, 1024)),
    "int_mid_7168": _perf((1, 1, 640, 7168)),
    "int_decode_32768": _cell((1, 1, 32, 32768)),
    "int_tile_f32_128x8192": _cell((128, 8192), dtype=ttnn.float32, gdtype=ttnn.float32),
    "int_tile_bf8_128x8192": _cell((128, 8192), dtype=ttnn.bfloat8_b, gdtype=ttnn.bfloat16),
    "int_bf8_2048x1024_bf16dst": _cell((1, 1, 2048, 1024), dtype=ttnn.bfloat8_b, fp32=False),
    "int_f32_2048x1024": _cell((1, 1, 2048, 1024), dtype=ttnn.float32, gdtype=ttnn.float32),
    "wid_perf_32x7168": _perf((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    # precision at the lowest fidelity (the FPU collapse is fidelity-dependent; the SFPU one is not)
    "lofi_bf16_1024x1024": _cell((1, 1, 1024, 1024), fp32=False, extras={"math_fidelity": ttnn.MathFidelity.LoFi}),
    "lofi_f32_1024x1024": _cell(
        (1, 1, 1024, 1024), dtype=ttnn.float32, gdtype=ttnn.float32, extras={"math_fidelity": ttnn.MathFidelity.LoFi}
    ),
    "lofi_blk_7168x1024": _cell(
        (1, 1, 7168, 1024),
        ml=ML.BLOCK_SHARDED,
        fp32=False,
        extras={"math_fidelity": ttnn.MathFidelity.LoFi, "shard_shape": [736, 128], "core_grid": (8, 10)},
    ),
    "hifi2_f32_1024x1024": _cell(
        (1, 1, 1024, 1024), dtype=ttnn.float32, gdtype=ttnn.float32, extras={"math_fidelity": ttnn.MathFidelity.HiFi2}
    ),
    # untested-regime sweep
    "int_tail_bf16_256x1000": _cell((1, 1, 256, 1000)),
    "int_tail_bf8_256x1000": _cell((1, 1, 256, 1000), dtype=ttnn.bfloat8_b),
    "int_tail_f32_256x1000": _cell((1, 1, 256, 1000), dtype=ttnn.float32, gdtype=ttnn.float32),
    "blk_tail_f32_224x1000": _cell((1, 1, 224, 1000), dtype=ttnn.float32, ml=ML.BLOCK_SHARDED, gdtype=ttnn.float32),
    "blk_tail_bf8_224x1000": _cell((1, 1, 224, 1000), dtype=ttnn.bfloat8_b, ml=ML.BLOCK_SHARDED),
    "wid_tail_nogamma_64x1000": _cell((1, 1, 64, 1000), ml=ML.WIDTH_SHARDED, gdtype=None),
    "hgt_rm_grm": _cell((128, 8192), layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM),
    "wid_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    "hgt_tile_4096_g": _cell((1, 1, 128, 4096), ml=ML.HEIGHT_SHARDED),
    "int_tile_f32_2x4x128x512": _cell((2, 4, 128, 512), dtype=ttnn.float32, gdtype=ttnn.float32),
    "int_rm_tail_96x1000": _cell((1, 1, 96, 1000), layout=RM, glayout=RM),
    "blk_perf_640x6144": _perf((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    "blk_perf_7168x2048": _perf((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
}
_sel = os.environ.get("FRF_CELLS")
IDS = [c for c in (_sel.split(",") if _sel else CELLS)]
VARIANTS = os.environ.get("FRF_VARIANTS", "op,0,1,2,3,4,5").split(",")
ORDER_LOG = HERE / os.environ.get("FRF_ORDER", "order.jsonl")


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))


@pytest.mark.parametrize("cell", IDS)
def test_bench(device, cell, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    c = CELLS[cell]
    shape, dtype, layout, ex = c["shape"], c["dtype"], c["layout"], c["extras"]
    torch.manual_seed(0)
    tdt = H._TORCH_DTYPE[dtype]
    x = torch.randn(shape, dtype=torch.float32).to(tdt)
    gamma = None
    tg = None
    if c["gdtype"] is not None:
        W = shape[-1]
        tg = torch.randn(W, dtype=torch.float32).to(H._TORCH_DTYPE[c["gdtype"]])
        gamma = H.create_ttnn_input_tensor(tg.reshape(1, 1, 1, W), device, dtype=c["gdtype"], layout=c["glayout"])
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ex.get("math_fidelity", ttnn.MathFidelity.HiFi4)
    cfg.fp32_dest_acc_en = c["fp32"]
    cfg.math_approx_mode = False
    expected = H.pytorch_rms_norm(x, gamma=tg, epsilon=1e-6)
    mc = None
    if ex.get("shard_shape") is not None:
        mc = shard_config(ex["shard_shape"], ex["core_grid"], c["ml"], layout=layout, dtype=dtype, device=device)
    xt = H.create_ttnn_input_tensor(x, device, dtype=dtype, layout=layout, memory_layout=c["ml"], memory_config=mc)
    kw = {}
    if c["ml"] != ML.INTERLEAVED:
        kw["memory_config"] = xt.memory_config()
    tol = H.TOLERANCES[dtype]
    if ex.get("pcc_threshold") is not None:
        tol = (ex["pcc_threshold"], tol[1])

    ref_out = None
    failures = []
    for v in VARIANTS:
        if v == "op":
            monkeypatch.setattr(op_mod, "create_program_descriptor", REAL_PD)
        elif (
            v == "g"
        ):  # graduate/: the clean single-path candidate (= variant 7, CB_SUMSQ_ACC dropped, CB_ONES planned)
            monkeypatch.setattr(op_mod, "create_program_descriptor", grad.create_program_descriptor)
        else:
            monkeypatch.setattr(clone, "FRONT_VARIANT", int(v))
            monkeypatch.setattr(op_mod, "create_program_descriptor", clone.create_program_descriptor)
        out = op_mod.rms_norm(xt, gamma=gamma, epsilon=1e-6, compute_kernel_config=cfg, **kw)
        with open(ORDER_LOG, "a") as f:
            f.write(json.dumps({"cell": cell, "variant": v}) + "\n")
        o = ttnn.to_torch(out).float()
        if ref_out is None:
            ref_out = o
        pcc = _pcc(o, expected.float())
        maxabs = float((o - ref_out).abs().max())
        maxabs_t = float((o - expected.float()).abs().max())
        status = "ok"
        try:
            check_output(out, expected, shape=shape, dtype=dtype, expected_layout=layout, tolerance=tol)
        except CheckOutputError as e:
            status = f"FAIL({str(e)[:160]})"
            failures.append((v, status))
        print(f"FRF {cell} v={v} pcc={pcc:.8f} maxabs_vs_first={maxabs:.4g} maxabs_vs_torch={maxabs_t:.4g} {status}")
        ttnn.deallocate(out)
    assert not failures, failures
