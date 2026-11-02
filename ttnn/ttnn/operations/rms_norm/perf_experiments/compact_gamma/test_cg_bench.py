# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compact_gamma bake-off: rms_norm with a compact cb_gamma (192 B page stride) vs the unmodified op.

    CG_CELLS=hgt_tile_res_128x8192 CG_VARIANTS=op,cg scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/compact_gamma/test_cg_bench.py::test_bench --import-mode=prepend

Per cell the inputs are built once (seed 0, like eval/golden_tests/rms_norm/helpers.run_rms_norm: x = randn, gamma =
randn(W), same compute config / shard geometry), then every variant dispatches once, in CG_VARIANTS order -> the
GenericOp profiler rows are (cell-major, variant-minor). "op" = the real op; "cg" = the clone in this dir
(cg_descriptor.py + kernels/) with compact pages; "cgfull" = the same clone with GAMMA_COMPACT = False (== the op).
Correctness: every variant must be BIT-IDENTICAL to the op's output (only the cb_gamma page stride changes; the
gamma bytes the unpacker sees in srcB row 0 are the same), plus PCC vs the fp32 torch reference.
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
_spec = importlib.util.spec_from_file_location("cg_descriptor", HERE / "cg_descriptor.py")
cgd = importlib.util.module_from_spec(_spec)
sys.modules["cg_descriptor"] = cgd  # dataclasses resolve the module through sys.modules
_spec.loader.exec_module(cgd)
_gspec = importlib.util.spec_from_file_location("cg_graduate", HERE / "graduate" / "rms_norm_program_descriptor.py")
grad = importlib.util.module_from_spec(_gspec)
sys.modules["cg_graduate"] = grad
_gspec.loader.exec_module(grad)
op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
real_pd = importlib.import_module("ttnn.operations.rms_norm.rms_norm_program_descriptor")

ML = ttnn.TensorMemoryLayout
HS, WS, BS, IL = ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED, ML.INTERLEAVED
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
F32, BF16, BF8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
HIFI2, HIFI4 = ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4

VARIANTS = {
    "cg": {},
    "cgfull": {"GAMMA_COMPACT": False},
    "cgrf": {"GAMMA_RESIDENT_FIRST": True},
    "cgnorf": {"GAMMA_RESIDENT_FIRST": False},
    "cgpoison": {"CG_TEST_DEFINES": ("RMS_CG_POISON",)},
    "cgfullpoison": {"GAMMA_COMPACT": False, "CG_TEST_DEFINES": ("RMS_CG_POISON",)},
}


def C(shape, ml, *, dtype=BF16, layout=TL, gdtype=BF16, glayout=TL, fp32=True, fid=HIFI4, pinned=None, force=0):
    return dict(
        shape=shape,
        ml=ml,
        dtype=dtype,
        layout=layout,
        gdtype=gdtype,
        glayout=glayout,
        fp32=fp32,
        fid=fid,
        pinned=pinned,
        force=force,
    )


def RES(shape, ml, layout=TL):  # golden resilience: bf16, fp32 DEST off, HiFi4, gamma bf16 in the input's Layout
    return C(shape, ml, layout=layout, glayout=layout, fp32=False)


def PERF(shape, ml, pinned=None):  # golden perf profile: bf16, fp32 DEST off, HiFi2, bf16 TILE gamma
    return C(shape, ml, fp32=False, fid=HIFI2, pinned=pinned)


CELLS = {
    # ---- targets: HEIGHT wide W, gamma streamed today ----
    "hgt_tile_res_128x8192": C((128, 8192), HS, fp32=False, gdtype=F32, glayout=RM),
    "hgt_rm_f32_128x4096": C((1, 1, 128, 4096), HS, dtype=F32, layout=RM, gdtype=F32, glayout=RM),
    "hgt_rm_224x11008": RES((1, 224, 11008), HS, RM),
    # ---- other streamed-today golden HEIGHT cells (largest) ----
    "hgt_tile_160x11008": RES((1, 1, 160, 11008), HS),
    "hgt_tile_224x11008": RES((1, 224, 11008), HS),
    "hgt_tile_128x8192_gf32": C((128, 8192), HS, gdtype=F32),
    "hgt_tile_128x8192_gbf8": C((128, 8192), HS, gdtype=BF8),
    "hgt_tile_f32_128x8192": C((128, 8192), HS, dtype=F32, gdtype=F32),
    "hgt_tile_f32_128x8192_grm": C((128, 8192), HS, dtype=F32, gdtype=F32, glayout=RM),
    "hgt_rm_128x8192": C((128, 8192), HS, layout=RM, glayout=RM),
    "hgt_rm_f32_128x8192": C((128, 8192), HS, dtype=F32, layout=RM, gdtype=F32, glayout=RM),
    "hgt_rm_f32_128x8192_gbf16": C((128, 8192), HS, dtype=F32, layout=RM, gdtype=BF16, glayout=RM),
    "hgt_rm_f32_32x8192_gbf16": C((1, 1, 32, 8192), HS, dtype=F32, layout=RM, gdtype=BF16, glayout=RM),
    "hgt_tile_32x8192_gf32": C((1, 1, 32, 8192), HS, gdtype=F32),
    "hgt_tile_32x8192_grmf32": C((1, 1, 32, 8192), HS, fp32=False, gdtype=F32, glayout=RM),
    "hgt_rm_f32_32x4096": C((1, 1, 32, 4096), HS, dtype=F32, layout=RM, gdtype=F32, glayout=RM),
    "hgt_rm_f32_2x64x4096": C((2, 1, 64, 4096), HS, dtype=F32, layout=RM, gdtype=F32, glayout=RM),
    "hgt_tile_100x5120": RES((100, 5120), HS),
    "hgt_rm_100x5120": RES((100, 5120), HS, RM),
    "hgt_tile_3104x4064": RES((3104, 4064), HS),
    "hgt_rm_3104x4064": RES((3104, 4064), HS, RM),
    # ---- at-risk: resident gamma today (multicast / per-core) ----
    "hgt_tile_128x8192": C((128, 8192), HS),
    "blk_perf_7168x1024": PERF((1, 1, 7168, 1024), BS, ([736, 128], (8, 10))),
    "blk_perf_7168x2048": PERF((1, 1, 7168, 2048), BS, ([736, 192], (11, 10))),
    "wid_perf_32x7168": PERF((1, 1, 32, 7168), WS, ([32, 256], (7, 4))),
    "int_prefill_1024": PERF((1, 1, 8192, 1024), IL),
    "int_tile_bf8_128x8192": C((128, 8192), IL, dtype=BF8, gdtype=BF8),
    "int_tile_f32_128x8192": C((128, 8192), IL, dtype=F32, gdtype=F32),
    "int_rm_128x4096_gf32": C((1, 1, 128, 4096), IL, layout=RM, gdtype=F32, glayout=RM),
    "wid_tile_160x11008": RES((1, 1, 160, 11008), WS),
    "blk_tile_128x4096": C((1, 1, 128, 4096), BS, gdtype=F32, glayout=RM),
    "blk_rm_128x4096": C((1, 1, 128, 4096), BS, layout=RM, fp32=False, gdtype=F32, glayout=RM),
    "wid_rm_1024x1024": C((1024, 1024), WS, layout=RM, gdtype=F32, glayout=RM),
    "blk_rm_96x6144": RES((1, 1, 96, 6144), BS, RM),
    "hgt_tile_bf8_2048x256": C((1, 1, 2048, 256), HS, dtype=BF8, fp32=False, gdtype=BF8),
    # ---- forced-streamed (GAMMA_STREAM_FORCE_CHUNK): the streamed path with compact chunks ----
    "f_hgt_2048x1024": C((1, 1, 2048, 1024), HS, force=5),
    "f_hgt_416x1025_grm": C((1, 1, 416, 1025), HS, glayout=RM, force=4),
    "f_blk_1024x1024": C((1, 1, 1024, 1024), BS, force=3),
    "f_wid_64x4095_grm": C((1, 1, 64, 4095), WS, glayout=RM, force=2),
    "f_hgt_7136x736": C((7136, 736), HS, force=16),
    "f_hgt_rm_2048x1024": C((1, 1, 2048, 1024), HS, layout=RM, glayout=RM, force=8, pinned=([256, 1024], (8, 1))),
    "f_hgt_rm_333x1000": C((1, 1, 333, 1000), HS, layout=RM, glayout=RM, force=4),
    "f_blk_rm_256x2000": C((1, 1, 256, 2000), BS, layout=RM, glayout=RM, force=4),
    "f_hgt_gbf8_2048x1024": C((1, 1, 2048, 1024), HS, gdtype=BF8, force=5),
    "f_hgt_gf32_2048x1024": C((1, 1, 2048, 1024), HS, gdtype=F32, force=5),
    "f_hgt_rm_f32_333x1000": C((1, 1, 333, 1000), HS, dtype=F32, layout=RM, gdtype=F32, glayout=RM, force=4),
}


def _td(dt):
    import torch  # function-local: the ttnn tree forbids global torch imports

    return {F32: torch.float32, BF16: torch.bfloat16, BF8: torch.bfloat16}[dt]


def _pcc(a, b):
    import torch  # function-local

    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _record_plan(m, mod, rec):
    """Wrap mod's blocking solvers so the plan actually used by the dispatch (budget with the output allocated) is
    recorded: (k, block rows, streamed gamma chunk G, RM chunk C)."""
    for name in ("select_sharded_blocking", "select_blocking"):
        fn = getattr(mod, name)

        def wrap(*a, _fn=fn, **kw):
            blk = _fn(*a, **kw)
            rec["plan"] = f"k={blk.k} b={blk.block_row_tiles} G={blk.opts.gamma_chunk} C={blk.rm_chunk_w_tiles}"
            return blk

        m.setattr(mod, name, wrap)


def run_cell(device, monkeypatch, cell, variants):
    import torch  # function-local

    c = CELLS[cell]
    shape, ml, dtype, layout = c["shape"], c["ml"], c["dtype"], c["layout"]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_td(dtype))
    W = shape[-1]
    g_t = torch.randn(W, dtype=torch.float32).to(_td(c["gdtype"]))
    g = ttnn.from_torch(
        g_t.reshape(1, 1, 1, W),
        dtype=c["gdtype"],
        layout=c["glayout"],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if ml == IL:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif c["pinned"] is not None:
        mc = shard_config(c["pinned"][0], c["pinned"][1], ml, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = c["fid"]
    cfg.fp32_dest_acc_en = c["fp32"]
    cfg.math_approx_mode = False
    xr = ttnn.to_torch(xt).float()
    gr = ttnn.to_torch(g).float().reshape(-1)
    ref = xr / torch.sqrt(torch.mean(xr * xr, dim=-1, keepdim=True) + 1e-6) * gr
    for mod in (real_pd, cgd, grad):
        monkeypatch.setattr(mod, "GAMMA_STREAM_FORCE_CHUNK", c["force"])
    out_kw = {} if ml == IL else {"memory_config": mc}

    outs, plans = {}, {}
    orig = op_mod.create_program_descriptor
    for v in variants:
        with monkeypatch.context() as m:
            if v == "op":
                m.setattr(op_mod, "create_program_descriptor", orig)
                mod = real_pd
            elif v == "grad":  # graduate/: the clean graduation copy (bench hooks stripped)
                m.setattr(op_mod, "create_program_descriptor", grad.create_program_descriptor)
                mod = grad
            else:
                for k_, v_ in VARIANTS[v].items():
                    m.setattr(cgd, k_, v_)
                m.setattr(op_mod, "create_program_descriptor", cgd.create_program_descriptor)
                mod = cgd
            rec = {}
            _record_plan(m, mod, rec)
            o = op_mod.rms_norm(xt, gamma=g, epsilon=1e-6, compute_kernel_config=cfg, **out_kw)
            outs[v] = ttnn.to_torch(o).float()
            plans[v] = rec.get("plan", "?") + (
                f" S,tail={mod._gamma_page_layout(c['gdtype'])}" if mod is not real_pd else ""
            )
            ttnn.deallocate(o)  # a sharded output holds L1: free it before the next variant
            del o
    base = outs.get("op")
    lines = []
    for v, o in outs.items():
        same = (
            ""
            if base is None
            else ("bit-exact" if torch.equal(o, base) else f"DIFF maxabs={(o - base).abs().max():.4g}")
        )
        lines.append(
            f"CG {cell} {v}: [{plans[v]}] pcc_ref={_pcc(o, ref):.6f} maxabs_ref={(o - ref).abs().max():.4g} vs_op={same}"
        )
    log = os.environ.get("CG_LOG")
    if log:
        with open(log, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    thr = 0.99 if BF8 in (dtype, c["gdtype"]) else 0.999
    for v, o in outs.items():
        assert _pcc(o, ref) >= thr, lines
        if base is not None:
            assert torch.equal(o, base), lines


_sel = os.environ.get("CG_CELLS")
CELL_IDS = [c for c in CELLS if not _sel or c in _sel.split(",")]


@pytest.mark.parametrize("cell", CELL_IDS)
def test_bench(device, monkeypatch, cell):
    run_cell(device, monkeypatch, cell, os.environ.get("CG_VARIANTS", "op,cg").split(","))
