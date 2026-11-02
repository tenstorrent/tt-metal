# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""height_rm_flat bake-off: HEIGHT_SHARDED ROW_MAJOR, tilized (op) vs flat-tile path (this dir).

    HRF_CELLS=hgt_rm_f32_128x4096 HRF_VARIANTS=base,zc,st ./run.sh <tag>

Per cell the input is built exactly like the golden harness (seed 0, auto_shard_config or a pinned shard geometry,
RM gamma, compute config), then dispatched once per variant:
  base = the real op (unmodified descriptor + kernels)
  zc   = flat path, zero-copy body (cb_x / cb_out backed on the shards) + staged tail tile
  st   = flat path, every stick staged into zero-padded flat tiles
The flat variants are FORCED (FLAT_FORCE: no cost-model gate) so the crossover can be measured; `grad` runs the
graduation defaults (gate on).
Correctness gate: golden tolerance vs torch for every variant; PCC / max-abs vs torch and vs `base` are printed.
One GenericOp row per dispatch, in the printed HRF_ORDER.
"""

import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import pytorch_rms_norm, create_ttnn_input_tensor, TOLERANCES
from eval.metrics import check_output, compute_metrics_torch
from eval.sharding import shard_config
import importlib
import importlib.util

from ttnn.operations.rms_norm.rms_norm import rms_norm

op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")  # the module (the package exports the function)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("flat_program_descriptor", HERE / "flat_program_descriptor.py")
fpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpd)
_gspec = importlib.util.spec_from_file_location(
    "hrf_grad_descriptor", HERE / "graduation" / "rms_norm_program_descriptor.py"
)
gpd = importlib.util.module_from_spec(_gspec)
sys.modules["hrf_grad_descriptor"] = gpd  # dataclasses resolve their module through sys.modules
_gspec.loader.exec_module(gpd)

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
F32, BF16 = ttnn.float32, ttnn.bfloat16
HIFI4, HIFI2 = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2

VARIANTS = {
    "base": None,
    "zc": "zc",
    "st": "staged",
    "zc32": "zc+ones_acc",
    "zc32c0": "zc+ones_acc+RMS_FLAT_FIN_COL0",  # finalize all of column 0 (the op's SFPU body)
    "st32": "staged+ones_acc",
    "grad": "GRAD",  # graduation defaults: cost-model gate (tilized where it is faster), zc, fp32-acc scaler
    "gradmod": "GRADMOD",  # graduation/rms_norm_program_descriptor.py (the op file as proposed) + graduation/kernels
}


def _c(shape, *, dtype=BF16, fp32=False, g=BF16, fid=HIFI4, shard=None):
    return dict(shape=shape, dtype=dtype, fp32=fp32, g=g, fid=fid, shard=shard)


CELLS = {
    # ---- targets (golden-harness geometry: auto HEIGHT shard, rows over up to 130 Tensix cores) ----
    "hgt_rm_f32_128x4096": _c((1, 1, 128, 4096), dtype=F32, fp32=True, g=F32),  # cartesian, 1 stick
    "hgt_rm_224x11008": _c((1, 224, 11008)),  # resilience, 2 sticks, W % 1024 = 768
    "hgt_rm_128x8192_gf32": _c((128, 8192), g=F32),  # cartesian fp32 DEST off, fp32 RM gamma, 1 stick
    "hgt_rm_96x6144": _c((1, 1, 96, 6144)),  # resilience, 1 stick
    "hgt_rm_100x5120": _c((100, 5120)),  # resilience, 1 stick, W = 5 flat tiles
    "hgt_rm_160x11008": _c((1, 1, 160, 11008)),  # resilience, 2 sticks
    # ---- domain: dtype / fp32 DEST / gamma ----
    "hgt_rm_128x4096_bf16_fp32d": _c((1, 1, 128, 4096), fp32=True),  # cartesian bf16 fp32 DEST, bf16 RM gamma
    "hgt_rm_128x4096_nog": _c((1, 1, 128, 4096), fp32=True, g=None),
    "hgt_rm_f32_128x8192_nog": _c((128, 8192), dtype=F32, fp32=True, g=None),
    "hgt_rm_32x4096": _c((1, 1, 32, 4096), fp32=True),  # 32 Tensix cores x 1 stick
    "hgt_rm_128x1024": _c((1, 1, 128, 1024), fp32=True),  # W = 1 flat tile
    "hgt_rm_128x2047": _c((1, 1, 128, 2047), fp32=True),  # W odd: tail 1023 datums, stick bytes % 16 != 0
    "hgt_rm_352x1023": _c((1, 1, 352, 1023)),  # resilience, 3 sticks, W < 1024 (tail only)
    "hgt_rm_f32_2x512x1024": _c((2, 512, 1024), dtype=F32, fp32=True, g=F32),  # 8 sticks / core
    # ---- crossover sweep: pinned shards, 64 Tensix cores (8x8), S sticks per core ----
    **{f"sw4096_s{s}": _c((1, 1, 64 * s, 4096), shard=([s, 4096], (8, 8))) for s in (1, 2, 4, 8, 16, 32, 64)},
    **{f"sw1024_s{s}": _c((1, 1, 64 * s, 1024), shard=([s, 1024], (8, 8))) for s in (1, 4, 16, 24, 32, 48, 64)},
    **{f"sw2048_s{s}": _c((1, 1, 64 * s, 2048), shard=([s, 2048], (8, 8))) for s in (8, 16, 24, 32, 48, 64)},
    **{f"sw256_s{s}": _c((1, 1, 64 * s, 256), shard=([s, 256], (8, 8))) for s in (1, 2, 4, 8, 16)},
    **{f"sw512_s{s}": _c((1, 1, 64 * s, 512), shard=([s, 512], (8, 8))) for s in (2, 4, 8, 16)},
    **{
        f"sw4096f_s{s}": _c((1, 1, 64 * s, 4096), dtype=F32, fp32=True, g=F32, shard=([s, 4096], (8, 8)))
        for s in (8, 16, 32)
    },
    # ---- golden HEIGHT RM cells near the gate ----
    "hgt_rm_3104x4064": _c((3104, 4064)),  # resilience, 24 sticks, nt = 4 (tail 992)
    "hgt_rm_3x736x5119": _c((3, 1, 736, 5119)),  # resilience, 17 sticks, ragged last shard, odd W
    "hgt_rm_1500x1000": _c((1500, 1000)),  # resilience, 12 sticks, W < 1024
    "hgt_rm_2047x2047": _c((2047, 2047)),  # resilience, 16 sticks, ragged
    "hgt_rm_992x3000": _c((1, 1, 992, 3000)),  # resilience, 8 sticks
    "hgt_rm_32x64": _c((1, 1, 32, 64), fp32=True),  # cartesian, 1 stick, W = 64
    "hgt_rm_1024x1024_gf32": _c((1024, 1024), fp32=True, g=F32),  # cartesian, 8 sticks
    # ---- at-risk (many sticks per Tensix core, W < 1024) ----
    "hgt_rm_2048x256": _c((1, 1, 2048, 256), fp32=True),  # cartesian (16 sticks / core)
    "hgt_rm_3104x416": _c((1, 1, 3104, 416)),  # resilience (24 sticks / core)
    "hgt_rm_99991x64": _c((99991, 64)),  # resilience (770 sticks / core)
}

_sel = os.environ.get("HRF_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]
VSEL = os.environ.get("HRF_VARIANTS", "base,zc").split(",")


@pytest.mark.parametrize("cell", IDS)
def test_bench(device, cell, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    device.disable_and_clear_program_cache()
    cfg = CELLS[cell]
    shape, dtype = cfg["shape"], cfg["dtype"]
    torch_dtype = torch.float32 if dtype == F32 else torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    W = shape[-1]
    g, tg = None, None
    if cfg["g"] is not None:
        g = torch.randn(W, dtype=torch.float32).to(torch.float32 if cfg["g"] == F32 else torch.bfloat16)
        tg = create_ttnn_input_tensor(g.reshape(1, 1, 1, W), device, dtype=cfg["g"], layout=RM)
    mc = None
    if cfg["shard"]:
        mc = shard_config(cfg["shard"][0], cfg["shard"][1], ML.HEIGHT_SHARDED, layout=RM, dtype=dtype, device=device)
    tx = create_ttnn_input_tensor(x, device, dtype=dtype, layout=RM, memory_layout=ML.HEIGHT_SHARDED, memory_config=mc)
    ck = ttnn.ComputeConfigDescriptor()
    ck.math_fidelity = cfg["fid"]
    ck.fp32_dest_acc_en = cfg["fp32"]
    ck.math_approx_mode = False
    expected = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    ss = tx.memory_config().shard_spec
    print(f"HRF_GEOM {cell}: {shape} shard={list(ss.shape)} cores={ss.grid.num_cores()}")

    ref = None
    for v in VSEL:
        mode = VARIANTS[v]
        if mode is None:
            monkeypatch.setattr(op_mod, "create_program_descriptor", fpd.pd.create_program_descriptor)
        elif mode == "GRADMOD":
            monkeypatch.setattr(op_mod, "create_program_descriptor", gpd.create_program_descriptor)
        elif mode == "GRAD":
            for k, val in (
                ("FLAT_MODE", "zc"),
                ("FLAT_ONES_ACC", True),
                ("FLAT_EXTRA_DEFINES", ()),
                ("FLAT_FORCE", False),
            ):
                monkeypatch.setattr(fpd, k, val)
            monkeypatch.setattr(op_mod, "create_program_descriptor", fpd.create_program_descriptor)
        else:
            parts = mode.split("+")
            monkeypatch.setattr(fpd, "FLAT_MODE", parts[0])
            monkeypatch.setattr(fpd, "FLAT_ONES_ACC", "ones_acc" in parts)
            monkeypatch.setattr(fpd, "FLAT_EXTRA_DEFINES", tuple(p for p in parts[1:] if p.startswith("RMS_")))
            monkeypatch.setattr(fpd, "FLAT_FORCE", True)
            monkeypatch.setattr(op_mod, "create_program_descriptor", fpd.create_program_descriptor)
        out = rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=ck, memory_config=tx.memory_config())
        print(f"HRF_ORDER {cell} {v}")
        res = ttnn.to_torch(out)
        ttnn.deallocate(out)
        m = compute_metrics_torch(res, expected, readback_dtype=res.dtype)
        line = f"HRF_PREC {cell} {v} pcc={m.pcc:.6f} max_abs={m.max_abs_diff:.4g} rms={m.rms:.4g}"
        if ref is not None:
            mb = compute_metrics_torch(res, ref, readback_dtype=res.dtype)
            line += f" | vs_{VSEL[0]}: pcc={mb.pcc:.6f} max_abs={mb.max_abs_diff:.4g} exact={torch.equal(res, ref)}"
        print(line)
        tol = TOLERANCES[dtype]
        assert m.pcc >= tol[0] and not (m.has_nan or m.has_inf), line
        if ref is None:
            ref = res
