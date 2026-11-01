# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""owner_raw_partials bake-off (Perf 1). Not collected by a directory run (no test_ prefix).

Variants (same cell, same pytest test, back to back, so the profiler CSV has GenericOp rows in this order):
  base     the op as committed (ttnn/ttnn/operations/rms_norm/...)
  raw      the clone in this dir with RMS_RAW_PARTIALS on the OWNER topology (members ship the raw sum-x^2 tile,
           4 faces; the owner adds k raw tiles + collapses once per row)
  raw_sym  (only when RB_SYM=1) the clone with RMS_RAW_PARTIALS on the SYMMETRIC all-gather too

  RB_CELLS=a,b  RB_SYM=1  scripts/run_safe_pytest.sh --profile <this file>::test_perf
  RB_CORR=1 scripts/run_safe_pytest.sh <this file>::test_correct      (no profiling; base vs raw vs torch)

Every variant is gated by the golden harness's check_output (PCC vs torch); raw is also compared to base
(PCC + max-abs), printed as `RB <cell> <variant> pcc_vs_base=.. maxabs_vs_base=..`.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

import eval.golden_tests.rms_norm.helpers as H
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HERE = Path(__file__).parent
REPO = HERE.parents[5]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


clone = _load("rms_norm_pd_raw", HERE / "rms_norm_program_descriptor_raw.py")
probe = _load("perf_p1_probe", REPO / "tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py")
import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

OPMOD = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = OPMOD.create_program_descriptor

CELLS = dict(probe.CELLS)
_sel = os.environ.get("RB_CELLS")
IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]


def _pcc(a, b):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _run_variant(device, axes, variant, monkeypatch):
    captured = {}
    orig = H.rms_norm

    def wrap(*a, **kw):
        out = orig(*a, **kw)
        captured["out"] = ttnn.to_torch(out).float()
        return out

    monkeypatch.setattr(H, "rms_norm", wrap)
    if variant == "base":
        monkeypatch.setattr(OPMOD, "create_program_descriptor", BASE_PD)
    else:
        monkeypatch.setattr(clone, "RAW_OWNER", True)
        monkeypatch.setattr(clone, "RAW_SYM", variant == "raw_sym")
        monkeypatch.setattr(clone, "RAW_REARM", variant == "raw_rearm")
        monkeypatch.setattr(clone, "RAW_BF16", variant == "raw_bf16")
        monkeypatch.setattr(OPMOD, "create_program_descriptor", clone.create_program_descriptor)
    run_rms_norm(device=device, **axes)
    monkeypatch.setattr(H, "rms_norm", orig)
    return captured["out"]


def _compare(tag, outs):
    base = outs["base"]
    for v, o in outs.items():
        if v == "base":
            continue
        print(f"RB {tag} {v} pcc_vs_base={_pcc(o, base):.7f} maxabs_vs_base={(o - base).abs().max().item():.5g}")


@pytest.mark.parametrize("cell", IDS)
def test_perf(device, cell, monkeypatch):
    axes = dict(CELLS[cell])
    variants = os.environ.get("RB_VARIANTS", "base,raw").split(",")
    if os.environ.get("RB_SYM") == "1":
        variants.append("raw_sym")
    outs = {v: _run_variant(device, axes, v, monkeypatch) for v in variants}
    _compare(cell, outs)


ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT


def _c(shape, ml, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, fp32=True, shard=None, poison=None, glayout=None):
    ex = {}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    if poison is not None:
        ex["poison_padding"] = poison
    return dict(
        inputs=(shape,),
        dtype=dtype,
        layout=layout,
        memory_layout=ml,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=glayout or layout,
        extras=ex,
    )


CORR = {
    # W-non-aligned sharded tails (RMS_MASK_TAIL: the tail member's raw tile must be masked), poisoned padding
    "ws_tail_32x4095": _c((1, 1, 32, 4095), ML.WIDTH_SHARDED, poison=1000.0),
    "bs_tail_416x1025": _c((1, 1, 416, 1025), ML.BLOCK_SHARDED, poison=1000.0),
    "ws_tail_32x200_f16": _c((1, 1, 32, 200), ML.WIDTH_SHARDED, fp32=False, poison=1000.0),
    "bs_tail_224x72_f16": _c((1, 1, 224, 72), ML.BLOCK_SHARDED, fp32=False, poison=1000.0),
    "ws_tail_100x47": _c((2, 1, 100, 47), ML.WIDTH_SHARDED),  # fill_pad: N_slices > 1 unsupported -> no poison
    "bs_tail_100x47_f16": _c((2, 1, 100, 47), ML.BLOCK_SHARDED, fp32=False),
    "bs_tail_256x2000": _c((1, 1, 256, 2000), ML.BLOCK_SHARDED, poison=1000.0),
    # RM sharded (tilize in compute; owner exchange)
    "ws_rm_2048x256": _c((1, 1, 2048, 256), ML.WIDTH_SHARDED, layout=RM),
    "bs_rm_333x1000": _c((1, 1, 333, 1000), ML.BLOCK_SHARDED, layout=RM),
    "ws_rm_32x4096_f16": _c((1, 1, 32, 4096), ML.WIDTH_SHARDED, layout=RM, fp32=False),
    # fp32 DEST on/off, fp32 / bf8b inputs
    "bs_1024_f32dest": _c((1, 1, 1024, 1024), ML.BLOCK_SHARDED, fp32=True),
    "bs_1024_f16dest": _c((1, 1, 1024, 1024), ML.BLOCK_SHARDED, fp32=False),
    "ws_f32_128x4096": _c((1, 1, 128, 4096), ML.WIDTH_SHARDED, dtype=ttnn.float32),
    "ws_bf8_128x4096": _c((1, 1, 128, 4096), ML.WIDTH_SHARDED, dtype=ttnn.bfloat8_b),
    "bs_bf8_1024x1024_f16dest": _c((1, 1, 1024, 1024), ML.BLOCK_SHARDED, dtype=ttnn.bfloat8_b, fp32=False),
    # interleaved owner (k > 8) and symmetric (k <= 8)
    "int_owner_32x32768": _c((1, 1, 32, 32768), ML.INTERLEAVED),
    "int_owner_160x11008_f16": _c((1, 1, 160, 11008), ML.INTERLEAVED, fp32=False),
    "int_sym_640x7168_f16": _c((1, 1, 640, 7168), ML.INTERLEAVED, fp32=False),
    "int_sym_tail_64x5000": _c((1, 1, 64, 5000), ML.INTERLEAVED, poison=1000.0),
}


_csel = os.environ.get("RB_CASES")


@pytest.mark.parametrize("case", [c for c in CORR if not _csel or c in _csel.split(",")])
def test_correct(device, case, monkeypatch):
    axes = dict(CORR[case])
    outs = {v: _run_variant(device, axes, v, monkeypatch) for v in ("base", "raw", "raw_sym")}
    _compare(case, outs)


def _p(shape, ml, shard=None, fp32=False, dtype=ttnn.bfloat16):
    ex = {"math_fidelity": ttnn.MathFidelity.HiFi2, "pcc_threshold": 0.9995}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    return dict(
        inputs=(shape,),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=ml,
        fp32_dest_acc_en=fp32,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.TILE_LAYOUT,
        extras=ex,
    )


# crossover sweep: tile-rows per block (R) vs k (perf config unless noted)
SWEEP = {
    "bk8_r2": _p((1, 1, 640, 1024), ML.BLOCK_SHARDED, ([64, 128], (8, 10))),
    "bk8_r4": _p((1, 1, 1280, 1024), ML.BLOCK_SHARDED, ([128, 128], (8, 10))),
    "bk8_r8": _p((1, 1, 2560, 1024), ML.BLOCK_SHARDED, ([256, 128], (8, 10))),
    "bk8_r16": _p((1, 1, 5120, 1024), ML.BLOCK_SHARDED, ([512, 128], (8, 10))),
    "bk4_r1": _p((1, 1, 320, 1024), ML.BLOCK_SHARDED, ([32, 256], (4, 10))),
    "bk4_r2": _p((1, 1, 640, 1024), ML.BLOCK_SHARDED, ([64, 256], (4, 10))),
    "bk4_r4": _p((1, 1, 1280, 1024), ML.BLOCK_SHARDED, ([128, 256], (4, 10))),
    "bk2_r2": _p((1, 1, 640, 512), ML.BLOCK_SHARDED, ([64, 256], (2, 10))),
    "wk8_r1": _p((1, 1, 32, 1024), ML.WIDTH_SHARDED, ([32, 128], (8, 1))),
    "wk8_r4": _p((1, 1, 128, 1024), ML.WIDTH_SHARDED, ([128, 128], (8, 1))),
    "wk8_r8": _p((1, 1, 256, 1024), ML.WIDTH_SHARDED, ([256, 128], (8, 1))),
    "wk16_r16": _p((1, 1, 512, 2048), ML.WIDTH_SHARDED, ([512, 128], (8, 2))),
    "int_160x11008": _p((1, 1, 160, 11008), ML.INTERLEAVED),
    "wk16_r1": _p((1, 1, 32, 2048), ML.WIDTH_SHARDED, ([32, 128], (8, 2))),
    "wk24_r1": _p((1, 1, 32, 3072), ML.WIDTH_SHARDED, ([32, 128], (8, 3))),
    "wk16_r1_f32d": _p((1, 1, 32, 2048), ML.WIDTH_SHARDED, ([32, 128], (8, 2)), fp32=True),
    "wk24_r1_f32d": _p((1, 1, 32, 3072), ML.WIDTH_SHARDED, ([32, 128], (8, 3)), fp32=True),
    "wk16_r4_f32d": _p((1, 1, 128, 2048), ML.WIDTH_SHARDED, ([128, 128], (8, 2)), fp32=True),
    "blk7168x1024_f32d": _p((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10)), fp32=True),
}
_ssel = os.environ.get("RB_SWEEP")


@pytest.mark.parametrize("cell", [c for c in SWEEP if not _ssel or c in _ssel.split(",")])
def test_sweep(device, cell, monkeypatch):
    monkeypatch.setenv("RB_DEBUG", "1")
    axes = dict(SWEEP[cell])
    outs = {v: _run_variant(device, axes, v, monkeypatch) for v in ("base", "raw")}
    _compare(cell, outs)
