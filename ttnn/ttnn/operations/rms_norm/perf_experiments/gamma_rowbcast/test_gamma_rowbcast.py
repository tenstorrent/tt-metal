# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment gamma_rowbcast: gamma consumed as a ROW-BROADCAST operand (no writer expand) vs the op.

    RB_SUITE=perf  RB_CELLS=a,b  RB_VARIANTS=base,rb_full,...  scripts/run_safe_pytest.sh --profile <this file>
    RB_SUITE=corr  scripts/run_safe_pytest.sh <this file> --import-mode=prepend
(pytest.ini's importlib mode would import this file as ttnn.ttnn.operations... and register ttnn twice)

Every cell is built exactly as eval/golden_tests/rms_norm/helpers.run_rms_norm builds it (seed 0, same gamma,
same shard geometry, same compute config). Per cell the SAME input tensors go through the unmodified op
("base") and then each variant of the clone (pd_rowbcast.py + kernels/), one dispatch each, so profiler GenericOp
rows map 1:1 to the (cell, variant) list written to dispatch_order.json. Correctness (PCC vs torch fp32 reference,
max-abs vs the op's own output) is recorded per dispatch to results_<suite>.jsonl; a variant below its golden PCC
gate fails the test. Perf is measured, never asserted.

Variants: base (the op as committed) | expand (clone, the op's path — sanity) | rb_full / rb_half / rb_compact
(A2 row-broadcast dest-reuse multiply; cb_gamma page = whole tile / faces 0-1 / 192 B overlapping) | rb_half_1r
(rb_half + ONE read of faces 0/1 per TILE gamma tile) | a1_full (A1 helper-only reorder).
"""

import json
import os
import pathlib

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import _TORCH_DTYPE, TOLERANCES, pytorch_rms_norm, create_ttnn_input_tensor
from eval.sharding import shard_config

import importlib
import importlib.util

_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("pd_rowbcast", _HERE / "pd_rowbcast.py")
pd = importlib.util.module_from_spec(_spec)
import sys

sys.modules["pd_rowbcast"] = pd  # dataclasses resolve the module through sys.modules
_spec.loader.exec_module(pd)
op_rms_norm = importlib.import_module("ttnn.operations.rms_norm.rms_norm").rms_norm
# "grad": the clean graduation candidate (graduate/: the op's files with ONLY the recommended change applied)
_gspec = importlib.util.spec_from_file_location("pd_grad", _HERE / "graduate" / "rms_norm_program_descriptor.py")
pd_grad = importlib.util.module_from_spec(_gspec)
sys.modules["pd_grad"] = pd_grad
_gspec.loader.exec_module(pd_grad)

ML = ttnn.TensorMemoryLayout
HIFI2 = ttnn.MathFidelity.HiFi2
RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT


def _cell(
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=TL,
    ml=ML.INTERLEAVED,
    fp32=True,
    gdtype=ttnn.bfloat16,
    glayout=TL,
    extras=None,
):
    return dict(
        shape=shape, dtype=dtype, layout=layout, ml=ml, fp32=fp32, gdtype=gdtype, glayout=glayout, extras=extras or {}
    )


def _perf(shape, ml=ML.INTERLEAVED, shard=None):
    ex = {"math_fidelity": HIFI2, "pcc_threshold": 0.9995}
    if shard:
        ex["shard_shape"], ex["core_grid"] = shard
    return _cell(shape, ml=ml, fp32=False, extras=ex)


PERF_CELLS = {
    # identical to tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py
    "int_prefill_7168": _perf((1, 1, 8192, 7168)),
    "blk_perf_7168x1024": _perf((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    "blk_perf_8192x1024": _perf((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    "blk_perf_640x6144": _perf((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    "wid_perf_32x7168": _perf((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    "hgt_tile_g": _cell((128, 8192), ml=ML.HEIGHT_SHARDED),
    "hgt_tile_grm": _cell((128, 8192), ml=ML.HEIGHT_SHARDED, glayout=RM),
    "hgt_tile_gbf8": _cell((128, 8192), ml=ML.HEIGHT_SHARDED, gdtype=ttnn.bfloat8_b),
    "hgt_tile_4096_g": _cell((1, 1, 128, 4096), ml=ML.HEIGHT_SHARDED),
    "hgt_rm_grm": _cell((128, 8192), layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM),
    "wid_tile_2048x256": _cell((1, 1, 2048, 256), ml=ML.WIDTH_SHARDED),
    "int_rm_128x4096_gf32": _cell((1, 1, 128, 4096), layout=RM, glayout=RM, gdtype=ttnn.float32),
    "int_tile_f32_128x8192": _cell((128, 8192), dtype=ttnn.float32, gdtype=ttnn.float32),
    "int_decode_32768": _cell((1, 1, 32, 32768)),
    "int_prefill_1024": _perf((1, 1, 8192, 1024)),
    # domain sweep extras (same definitions as the probe)
    "int_mid_7168": _perf((1, 1, 640, 7168)),
    "blk_perf_7168x2048": _perf((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
    "wid_perf_32x5120": _perf((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),
    "wid_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    "blk_rm_2048x256": _cell((1, 1, 2048, 256), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
    "int_tile_bf8_128x8192": _cell((128, 8192), dtype=ttnn.bfloat8_b, gdtype=ttnn.bfloat16),
    "int_tile_f32_2x4x128x512": _cell((2, 4, 128, 512), dtype=ttnn.float32, gdtype=ttnn.float32),
}

POISON = {"poison_padding": 1000.0}
CORR_CELLS = {
    # gamma dtype x layout on interleaved TILE input (fp32 DEST on), k = 1 multicast + k > 1 per-core
    "c_int_g_bf16_t": _cell((1, 1, 64, 1024)),
    "c_int_g_f32_t": _cell((1, 1, 64, 1024), gdtype=ttnn.float32),
    "c_int_g_bf8_t": _cell((1, 1, 64, 1024), gdtype=ttnn.bfloat8_b),
    "c_int_g_bf16_rm": _cell((1, 1, 64, 1024), glayout=RM),
    "c_int_g_f32_rm": _cell((1, 1, 64, 1024), glayout=RM, gdtype=ttnn.float32),
    "c_int_wide_k_g_bf8": _cell((1, 1, 32, 32768), gdtype=ttnn.bfloat8_b),
    "c_int_wide_k_g_f32_rm": _cell((1, 1, 32, 16384), glayout=RM, gdtype=ttnn.float32),
    # fp32 DEST off, bf16 / bf8b / fp32 inputs
    "c_int_nofp32": _cell((1, 1, 256, 2048), fp32=False),
    "c_int_x_bf8": _cell((2, 1, 64, 1024), dtype=ttnn.bfloat8_b, fp32=False),
    "c_int_x_f32": _cell((1, 1, 64, 2048), dtype=ttnn.float32, gdtype=ttnn.float32),
    # RM input
    "c_int_rm_g_bf16_rm": _cell((1, 1, 64, 4096), layout=RM, glayout=RM),
    "c_int_rm_g_bf16_t": _cell((1, 1, 64, 1024), layout=RM),
    "c_int_rm_g_f32_rm_nofp32": _cell((1, 1, 100, 1000), layout=RM, glayout=RM, gdtype=ttnn.float32, fp32=False),
    # sharded
    "c_hgt_g_f32_t": _cell((1, 1, 256, 1024), ml=ML.HEIGHT_SHARDED, gdtype=ttnn.float32),
    "c_wid_g_bf8_t": _cell((1, 1, 64, 2048), ml=ML.WIDTH_SHARDED, gdtype=ttnn.bfloat8_b),
    "c_blk_g_bf16_rm": _cell((1, 1, 512, 2048), ml=ML.BLOCK_SHARDED, glayout=RM),
    "c_hgt_rm_g_f32_rm": _cell((1, 1, 256, 1024), layout=RM, ml=ML.HEIGHT_SHARDED, glayout=RM, gdtype=ttnn.float32),
    "c_wid_rm_g_bf16_rm": _cell((1, 1, 64, 2048), layout=RM, ml=ML.WIDTH_SHARDED, glayout=RM),
    "c_blk_rm_g_bf16_rm": _cell((1, 1, 2048, 256), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
    # W non-aligned + poisoned padding (golden loose-case corner: bf16, fp32 DEST off, gamma bf16 TILE)
    **{
        f"c_poison_{h}x{w}_{m.name if hasattr(m, 'name') else str(m).split('.')[-1]}": _cell(
            (1, 1, h, w), ml=m, fp32=False, extras=POISON
        )
        for (h, w) in ((32, 72), (40, 40), (224, 72))
        for m in (ML.INTERLEAVED, ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED)
    },
    "c_rm_nonaligned_g_bf16_rm_blk": _cell((1, 1, 100, 1000), layout=RM, ml=ML.BLOCK_SHARDED, glayout=RM),
}

VARIANTS = {
    "expand": dict(GAMMA_IMPL="expand", GAMMA_PAGE="full", GAMMA_WHOLE_READ=False),
    "rb_full": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="full", GAMMA_WHOLE_READ=False),
    "rb_half": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="half", GAMMA_WHOLE_READ=False),
    "rb_half_1r": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="half", GAMMA_WHOLE_READ=True),
    "rb_compact": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="compact", GAMMA_WHOLE_READ=False),
    "a1_full": dict(GAMMA_IMPL="a1", GAMMA_PAGE="full", GAMMA_WHOLE_READ=False),
    "rb_full_rmd": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="full", GAMMA_WHOLE_READ=False, GAMMA_RM_DIRECT=True),
    "rb_half_rmd": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="half", GAMMA_WHOLE_READ=False, GAMMA_RM_DIRECT=True),
    "rb_compact_rmd": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="compact", GAMMA_WHOLE_READ=False, GAMMA_RM_DIRECT=True),
    # GAMMA_RM_DIRECT=2: fp32 RM gamma read as 2 exact 64 B DRAM reads per tile (bf16 unchanged: 1 read + local copy)
    "rb_compact_rmd2": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="compact", GAMMA_WHOLE_READ=False, GAMMA_RM_DIRECT=2),
    "rb_half_rmd2": dict(GAMMA_IMPL="rowbcast", GAMMA_PAGE="half", GAMMA_WHOLE_READ=False, GAMMA_RM_DIRECT=2),
    "a1_compact": dict(GAMMA_IMPL="a1", GAMMA_PAGE="compact", GAMMA_WHOLE_READ=False),
}

SUITE = os.environ.get("RB_SUITE", "corr")
_cells_all = PERF_CELLS if SUITE == "perf" else CORR_CELLS
_sel = os.environ.get("RB_CELLS")
CELL_IDS = [c for c in (_sel.split(",") if _sel else _cells_all) if c in _cells_all]
VAR_IDS = os.environ.get("RB_VARIANTS", "base,rb_full,rb_compact,a1_full").split(",")
FORCE_STREAM = int(os.environ.get("RB_FORCE_STREAM", "0"))  # > 0: GAMMA_STREAM_FORCE_CHUNK in the clone


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 1.0


def _build(c, device):
    import torch  # function-local: the ttnn tree forbids global torch imports

    shape = c["shape"]
    extras = c["extras"]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[c["dtype"]])
    W = shape[-1]
    g = torch.randn(W, dtype=torch.float32).to(_TORCH_DTYPE[c["gdtype"]])
    tg = create_ttnn_input_tensor(g.reshape(1, 1, 1, W), device, dtype=c["gdtype"], layout=c["glayout"])
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
    cfg.fp32_dest_acc_en = c["fp32"]
    cfg.math_approx_mode = False
    mc = None
    if extras.get("shard_shape") is not None:
        mc = shard_config(
            extras["shard_shape"],
            extras["core_grid"],
            c["ml"],
            layout=c["layout"],
            dtype=c["dtype"],
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            device=device,
        )
    tx = create_ttnn_input_tensor(
        x, device, dtype=c["dtype"], layout=c["layout"], memory_layout=c["ml"], memory_config=mc
    )
    if "poison_padding" in extras:
        if c["layout"] == TL:
            tx = ttnn.fill_implicit_tile_padding(tx, extras["poison_padding"])
        if c["glayout"] == TL:
            tg = ttnn.fill_implicit_tile_padding(tg, extras["poison_padding"])
    ref = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    return tx, tg, cfg, ref


def _run_variant(var, tx, tg, cfg, monkeypatch):
    if var == "base":
        kw = {}
        if tx.memory_config().is_sharded():
            kw["memory_config"] = tx.memory_config()
        return op_rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg, **kw)
    mod = pd_grad if var == "grad" else pd
    if var != "grad":
        monkeypatch.setattr(pd, "GAMMA_RM_DIRECT", False)
        for k, v in VARIANTS[var].items():
            monkeypatch.setattr(pd, k, v)
    monkeypatch.setattr(mod, "GAMMA_STREAM_FORCE_CHUNK", FORCE_STREAM)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, tx.device(), tx.memory_config()
    )
    desc = mod.create_program_descriptor(tx, tg, out, epsilon=1e-6, compute_kernel_config=cfg)
    return ttnn.generic_op([tx, tg, out], desc)


def test_gamma_rowbcast(device, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    res_path = _HERE / f"results_{SUITE}.jsonl"
    order = []
    failures = []
    with open(res_path, "w") as res:  # the precompile collect pass (NO_DISPATCH) is overwritten by the real pass
        for cid in CELL_IDS:
            c = _cells_all[cid]
            tx, tg, cfg, ref = _build(c, device)
            outs = {}
            recs = []
            for var in VAR_IDS:
                rec = dict(suite=SUITE, cell=cid, variant=var, force_stream=FORCE_STREAM)
                try:
                    out = _run_variant(var, tx, tg, cfg, monkeypatch)
                    order.append(f"{cid}:{var}")
                    o = ttnn.to_torch(out).float().reshape(ref.shape)
                    outs[var] = o
                    rec["pcc"] = _pcc(o, ref.float())
                    rec["max_abs_ref"] = float((o - ref.float()).abs().max())
                    gate = c["extras"].get("pcc_threshold", TOLERANCES[c["dtype"]][0])
                    rec["gate"] = gate
                    rec["ok"] = bool(rec["pcc"] >= gate and torch.isfinite(o).all())
                    out.deallocate()
                except Exception as e:  # recorded; a variant that cannot run is a correctness failure
                    rec["ok"] = False
                    rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                recs.append(rec)
            # vs the op's own output, wherever "base" ran in the variant order (run it LAST to keep the op's expanded
            # gamma bytes out of the variants' cb_gamma L1 — stale expanded rows would mask a row-broadcast bug)
            for rec in recs:
                o = outs.get(rec["variant"])
                if "base" in outs and o is not None and rec["variant"] != "base":
                    rec["max_abs_vs_base"] = float((o - outs["base"]).abs().max())
                    rec["pcc_vs_base"] = _pcc(o, outs["base"])
                if not rec["ok"]:
                    failures.append(rec)
                print("RB", json.dumps(rec))
                res.write(json.dumps(rec) + "\n")
            # free this cell's L1 / DRAM tensors before the next cell (HEIGHT wide shards are 0.5 MB per core)
            tx.deallocate()
            tg.deallocate()
            del tx, tg
    (_HERE / f"dispatch_order_{SUITE}.json").write_text(json.dumps(order))
    assert not failures, failures
