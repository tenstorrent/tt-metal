# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""gamma_stream_l1 bake-off: the STREAMED-gamma fallback of rms_norm (HEIGHT wide W), op vs cloned variants.

    GS_CELLS=hgt_tile_g GS_VARIANTS=op,base,sf scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/gamma_stream_l1/test_gs_bench.py::test_bench --import-mode=prepend
(pytest.ini's importlib mode would import this file as ttnn.ttnn.operations... and register ttnn twice)

Per cell the inputs are built once (seed 0, like eval/golden_tests/rms_norm/helpers.run_rms_norm), then every
variant dispatches once, in GS_VARIANTS order -> the GenericOp profiler rows are (cell-major, variant-minor).
"op" = the real, unmodified op; every other variant = the clone in this dir (gs_descriptor.py + kernels/) with the
variant's defines / knobs. Correctness: every variant's output must be BIT-IDENTICAL to the op's (the expanded
gamma tiles are the same bytes), plus PCC vs the fp32 torch reference.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.sharding import auto_shard_config, shard_config

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("gs_descriptor", HERE / "gs_descriptor.py")
gsd = importlib.util.module_from_spec(_spec)
sys.modules["gs_descriptor"] = gsd  # dataclasses resolve the module through sys.modules
_spec.loader.exec_module(gsd)
op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
real_pd = importlib.import_module("ttnn.operations.rms_norm.rms_norm_program_descriptor")

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT

# variant -> (GS_DEFINES, clone knob overrides)
VARIANTS = {
    "base": ((), {}),
    "sf": (("GS_STATEFUL",), {}),
    "compact": (("GS_COMPACT",), {}),
    "compact_sf": (("GS_COMPACT", "GS_STATEFUL"), {}),
    "compact_sf_d4": (("GS_COMPACT", "GS_STATEFUL"), {"GAMMA_STREAM_DEPTH": 4}),
    "compact_sf_g32": (("GS_COMPACT", "GS_STATEFUL"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "compact_g32": (("GS_COMPACT",), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "compact_g64": (("GS_COMPACT",), {"GAMMA_STREAM_CHUNK_CAP": 64}),
    "compact_d4": (("GS_COMPACT",), {"GAMMA_STREAM_DEPTH": 4}),
    "split": (("GS_SPLIT",), {}),
    "compact_split": (("GS_COMPACT", "GS_SPLIT"), {}),
    "compact_split_g32": (("GS_COMPACT", "GS_SPLIT"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "base_g32": ((), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "ablate_fill": (("GS_ABLATE_FILL",), {}),
    "compact_g32_rv": (("GS_COMPACT", "GS_RM_SCATTER_RISCV"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "res": (("GS_RES_LOCAL",), {}),
    "res_split": (("GS_RES_LOCAL", "GS_SPLIT"), {}),
    "lean": (("GS_COMPACT", "GS_RES_LOCAL"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "full": (("GS_COMPACT", "GS_SPLIT", "GS_RES_LOCAL"), {}),
    "full_g32": (("GS_COMPACT", "GS_SPLIT", "GS_RES_LOCAL"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "full_mc": (("GS_COMPACT", "GS_SPLIT", "GS_RES_LOCAL", "GS_RES_MCAST"), {}),
    "lean_mc": (("GS_COMPACT", "GS_RES_LOCAL", "GS_RES_MCAST"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "res_c32": (("GS_RES_LOCAL",), {"GS_RES_CHUNK_TILE": 32}),
    "fix": (("GS_WINDOW_FIX",), {}),
    "full_mc_fix": (("GS_COMPACT", "GS_SPLIT", "GS_RES_LOCAL", "GS_RES_MCAST", "GS_WINDOW_FIX"), {}),
    "lean_mc_fix": (("GS_COMPACT", "GS_RES_LOCAL", "GS_RES_MCAST", "GS_WINDOW_FIX"), {"GAMMA_STREAM_CHUNK_CAP": 32}),
    "compact_sf_g32_d4": (("GS_COMPACT", "GS_STATEFUL"), {"GAMMA_STREAM_CHUNK_CAP": 32, "GAMMA_STREAM_DEPTH": 4}),
}

# cell -> (shape, layout, x dtype, gamma dtype, gamma layout)  — HEIGHT, auto shard, HiFi4, fp32 DEST (golden cells)
CELLS = {
    "hgt_tile_g": ((128, 8192), TL, ttnn.bfloat16, ttnn.bfloat16, TL),
    "hgt_tile_grm": ((128, 8192), TL, ttnn.bfloat16, ttnn.bfloat16, RM),
    "hgt_tile_gbf8": ((128, 8192), TL, ttnn.bfloat16, ttnn.bfloat8_b, TL),
    "hgt_tile_4096_g": ((1, 1, 128, 4096), TL, ttnn.bfloat16, ttnn.bfloat16, TL),
    "hgt_rm_grm": ((128, 8192), RM, ttnn.bfloat16, ttnn.bfloat16, RM),
    # domain sweep
    "hgt_tile_gf32": ((128, 8192), TL, ttnn.bfloat16, ttnn.float32, TL),
    "hgt_tile_f32_gf32": ((128, 8192), TL, ttnn.float32, ttnn.float32, TL),
    "hgt_tile_160x11008": ((1, 1, 160, 11008), TL, ttnn.bfloat16, ttnn.bfloat16, TL),
    "hgt_tile_32x8192": ((1, 1, 32, 8192), TL, ttnn.bfloat16, ttnn.bfloat16, TL),
    "hgt_tile_1024x8192": ((1, 1, 1024, 8192), TL, ttnn.bfloat16, ttnn.bfloat16, TL),
    "hgt_tile_4160x8192": ((1, 1, 4160, 8192), TL, ttnn.bfloat16, ttnn.bfloat16, TL),  # 130 cores, streamed
    "hgt_tile_4096x2048": ((1, 1, 4096, 2048), TL, ttnn.bfloat16, ttnn.bfloat16, TL),  # 128 cores, resident
    "hgt_tile_8192x1024_gbf8": ((1, 1, 8192, 1024), TL, ttnn.bfloat16, ttnn.bfloat8_b, TL),  # 2 tile-rows/core
    "hgt_tile_8320x4608_gf32": ((1, 1, 8320, 4608), TL, ttnn.bfloat16, ttnn.float32, TL),  # 2 rows/shard, streamed?
    "hgt_rm_32x4096_gf32": ((1, 1, 32, 4096), RM, ttnn.float32, ttnn.float32, RM),
}


def _td(dt):
    import torch  # function-local: the ttnn tree forbids global torch imports

    return {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}[dt]


def _pcc(a, b):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _use_clone(monkeypatch, variant):
    defs, knobs = VARIANTS[variant]
    monkeypatch.setattr(gsd, "GS_DEFINES", defs)
    for k_, v_ in knobs.items():
        monkeypatch.setattr(gsd, k_, v_)
    monkeypatch.setattr(op_mod, "create_program_descriptor", gsd.create_program_descriptor)


def run_variants(
    device,
    monkeypatch,
    shape,
    layout,
    dtype,
    gdtype,
    glayout,
    variants,
    *,
    ml=HS,
    pinned=None,
    force_chunk=0,
    rm_cap=None,
    fidelity=ttnn.MathFidelity.HiFi4,
    fp32=True,
    tag="",
):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_td(dtype))
    W = shape[-1]
    g_t = torch.randn(W, dtype=torch.float32).to(_td(gdtype))
    g = ttnn.from_torch(
        g_t.reshape(1, 1, 1, W), dtype=gdtype, layout=glayout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if pinned is not None:
        mc = shard_config(pinned[0], pinned[1], ml, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32
    cfg.math_approx_mode = False
    xr = ttnn.to_torch(xt).float()
    gr = ttnn.to_torch(g).float().reshape(-1)
    ref = xr / torch.sqrt(torch.mean(xr * xr, dim=-1, keepdim=True) + 1e-6) * gr

    for mod in (real_pd, gsd):
        monkeypatch.setattr(mod, "GAMMA_STREAM_FORCE_CHUNK", force_chunk)
        if rm_cap is not None:
            monkeypatch.setattr(mod, "RM_CHUNK_CAP", rm_cap)
    log = os.environ.get("GS_LOG")
    monkeypatch.setattr(gsd, "GS_LOG", log)
    outs = {}
    orig = op_mod.create_program_descriptor
    for v in variants:
        with monkeypatch.context() as m:
            if v != "op":
                _use_clone(m, v)
            else:
                m.setattr(op_mod, "create_program_descriptor", orig)
            o = op_mod.rms_norm(xt, gamma=g, epsilon=1e-6, compute_kernel_config=cfg, memory_config=mc)
            outs[v] = ttnn.to_torch(o).float()
            ttnn.deallocate(o)  # the sharded output holds half the L1: free it before the next variant
            del o
    base = outs.get("op")
    lines = []
    for v, o in outs.items():
        p = _pcc(o, ref)
        same = (
            ""
            if base is None
            else ("bit-exact" if torch.equal(o, base) else f"DIFF maxabs={(o - base).abs().max().item():.4g}")
        )
        if base is not None and not torch.equal(o, base):
            d = (o != base).reshape(-1, o.shape[-1])
            bad_r = torch.nonzero(d.any(dim=1)).flatten()
            bad_c = torch.nonzero(d.any(dim=0)).flatten()
            same += (
                f" bad_rows={bad_r.numel()}[{bad_r.min().item()}..{bad_r.max().item()}]"
                f" bad_tilecols={sorted(set((bad_c // 32).tolist()))[:40]}"
            )
        lines.append(f"{tag or shape} {v}: pcc_ref={p:.6f} maxabs_ref={(o - ref).abs().max().item():.4g} vs_op={same}")
    if log:
        with open(log, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    thr = 0.99 if ttnn.bfloat8_b in (dtype, gdtype) else 0.999
    op_ok = base is None or _pcc(base, ref) >= thr
    for v, o in outs.items():
        if v.startswith("ablate") or v == "op" or (v == "base" and not op_ok):
            continue  # ablations are wrong by design; the op (and its clone) are reported, judged separately
        assert _pcc(o, ref) >= thr, lines
        if base is not None and op_ok:
            assert torch.equal(o, base), lines  # same expanded gamma bytes -> bit-identical output
    assert op_ok or os.environ.get("GS_ALLOW_OP_FAIL") == "1", lines


_sel = os.environ.get("GS_CELLS")
CELL_IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]


@pytest.mark.parametrize("cell", CELL_IDS)
def test_bench(device, monkeypatch, cell):
    variants = os.environ.get("GS_VARIANTS", "op,base").split(",")
    shape, layout, dtype, gdtype, glayout = CELLS[cell]
    run_variants(device, monkeypatch, shape, layout, dtype, gdtype, glayout, variants, tag=cell)


# ---- at-risk: the op's forced-streamed / streamed tests, clone variant vs op bit-exact ----
FORCED = [
    ((1, 1, 2048, 1024), HS, 5, TL, None, None),
    ((1, 1, 416, 1025), HS, 4, TL, None, None),
    ((1, 1, 1024, 1024), BS, 3, TL, None, None),
    ((1, 1, 64, 4095), WS, 2, TL, None, None),
    ((1, 1, 2048, 1024), HS, 5, RM, None, None),  # TILE x, RM gamma
    ((1, 1, 416, 1025), HS, 4, RM, None, None),
    ((1, 1, 1024, 1024), BS, 3, RM, None, None),
    ((1, 1, 64, 4095), WS, 2, RM, None, None),
    ((7136, 736), HS, 16, TL, None, None),  # 2 tile-rows per shard x 2 chunks (16 + 7)
]
FORCED_RM = [
    ((1, 1, 2048, 1024), HS, 8, ([256, 1024], (8, 1))),
    ((1, 1, 333, 1000), HS, 4, None),
    ((1, 1, 256, 2000), BS, 4, None),
]
STREAMED = [
    ((1, 1, 32, 8192), ttnn.bfloat16, ttnn.bfloat16, TL, TL),
    ((1, 1, 32, 8192), ttnn.bfloat8_b, ttnn.float32, TL, TL),
    ((1, 1, 32, 4096), ttnn.float32, ttnn.float32, RM, TL),
    ((1, 1, 160, 11008), ttnn.bfloat16, ttnn.bfloat16, TL, TL),
    ((1, 1, 32, 4096), ttnn.float32, ttnn.float32, RM, RM),
    ((1, 1, 160, 11008), ttnn.bfloat16, ttnn.bfloat16, RM, RM),
]


def _risk_variants():
    return ["op"] + os.environ.get("GS_RISK_VARIANTS", "compact_sf").split(",")


@pytest.mark.parametrize("i", range(len(FORCED)))
def test_forced(device, monkeypatch, i):
    shape, ml, chunk, glayout, _, _ = FORCED[i]
    run_variants(
        device,
        monkeypatch,
        shape,
        TL,
        ttnn.bfloat16,
        ttnn.bfloat16,
        glayout,
        _risk_variants(),
        ml=ml,
        force_chunk=chunk,
        tag=f"forced{i}",
    )


@pytest.mark.parametrize("i", range(len(FORCED_RM)))
def test_forced_rm(device, monkeypatch, i):
    shape, ml, cap, pinned = FORCED_RM[i]
    run_variants(
        device,
        monkeypatch,
        shape,
        RM,
        ttnn.bfloat16,
        ttnn.bfloat16,
        RM,
        _risk_variants(),
        ml=ml,
        pinned=pinned,
        force_chunk=1,
        rm_cap=cap,
        tag=f"forced_rm{i}",
    )


@pytest.mark.parametrize("i", range(len(STREAMED)))
def test_streamed(device, monkeypatch, i):
    shape, dtype, gdtype, glayout, layout = STREAMED[i]
    run_variants(device, monkeypatch, shape, layout, dtype, gdtype, glayout, _risk_variants(), tag=f"streamed{i}")


HEIGHT_SHAPES = [
    (1, 1, 32, 2048),
    (1, 1, 32, 64),
    (1, 1, 32, 32),
    (1, 1, 128, 4096),
    (1, 1, 32, 8192),
    (4, 8, 32, 256),
    (7136, 736),
    (99991, 64),
    (100, 5120),
]


@pytest.mark.parametrize("layout", [TL, RM])
@pytest.mark.parametrize("i", range(len(HEIGHT_SHAPES)))
def test_height_res(device, monkeypatch, i, layout):
    """test_rms_norm_sharded.test_height cells (HEIGHT k = 1, resident gamma in the op) — GS_RES_LOCAL domain."""
    run_variants(
        device,
        monkeypatch,
        HEIGHT_SHAPES[i],
        layout,
        ttnn.bfloat16,
        ttnn.bfloat16,
        layout,
        _risk_variants(),
        fidelity=ttnn.MathFidelity.HiFi2,
        fp32=False,
        tag=f"height{i}_{layout}",
    )
