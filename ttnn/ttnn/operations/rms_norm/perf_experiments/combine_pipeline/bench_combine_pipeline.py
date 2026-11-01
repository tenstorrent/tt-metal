# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""combine_pipeline bake-off: sharded k > 1 owner combine, resident shard sub-blocked + pipelined.

Not collected by a directory run (no `test_` prefix). Run:

    CP_CELLS=blk_perf_7168x1024 CP_VARIANTS=base,n2,n4 CP_TAG=run1 scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/combine_pipeline/bench_combine_pipeline.py

Variants (CP_VARIANTS, comma list):
  base         the UNMODIFIED op (ttnn/ttnn/operations/rms_norm, honest baseline)
  clone        this dir's clone with the pipeline off (baseline path of the modified kernels)
  n<N>         clone, N sub-blocks (s = ceil(crt / N) tile-rows)
  r<S>         clone, sub-blocks of S tile-rows
  suffixes     _lag0 (pass 2 of t right after combine t), _norot (no owner rotation),
               _gl / _ge (gamma finished late / at kernel start), _par (two-round window, parity semaphores)
Every dispatch is checked vs torch (PCC) and vs the `base` output of the same cell (max abs diff); the
per-dispatch label + metrics are appended to runs/<CP_TAG>_labels.jsonl in dispatch order (GenericOp rows of
the profiler CSV map 1:1, in order).
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_tests.rms_norm.helpers import _TORCH_DTYPE, pytorch_rms_norm
from eval.sharding import auto_shard_config, shard_config

HERE = Path(__file__).parent
REPO = HERE.resolve().parents[5]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


p1 = _load("cp_perf_p1_probe", REPO / "tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py")

import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

opmod = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = opmod.create_program_descriptor

ppd = _load("cp_pipe_pd", HERE / "pipe_program_descriptor.py")
gpd = _load("cp_grad_pd", HERE / "graduate" / "rms_norm_program_descriptor.py")  # clean graduation candidate

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT

CELLS = dict(p1.CELLS)
CELLS.update(
    {
        # correctness at-risk cells (HiFi4, fp32 DEST unless noted)
        # W tail (1000 = 31 full tiles + 8 datums) in the last BLOCK column, 4 tile-rows per core, k = 8
        "blk_tail_1024x1000": p1._cell(
            (1, 1, 1024, 1000), ml=ML.BLOCK_SHARDED, extras={"shard_shape": [128, 128], "core_grid": (8, 8)}
        ),
        # ragged last shard row: 32 tile-rows in 6-tile-row shards -> last group 2 tile-rows, k = 8
        "blk_ragged_1000x1024": p1._cell(
            (1, 1, 1000, 1024), ml=ML.BLOCK_SHARDED, extras={"shard_shape": [192, 128], "core_grid": (8, 6)}
        ),
        # bf16 perf cfg with a W tail + ragged rows at once
        "blk_tail_ragged_perf": p1._cell(
            (1, 1, 7000, 1000),
            ml=ML.BLOCK_SHARDED,
            fp32=False,
            extras={"math_fidelity": ttnn.MathFidelity.HiFi2, "shard_shape": [736, 128], "core_grid": (8, 10)},
        ),
        "blk_tile_2048x256": p1._cell((1, 1, 2048, 256), ml=ML.BLOCK_SHARDED),
        "wid_tile_1024x1000": p1._cell(
            (1, 1, 1024, 1000), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [1024, 128], "core_grid": (8, 1)}
        ),
        "blk_bf8_2048x1024": p1._cell((1, 1, 2048, 1024), dtype=ttnn.bfloat8_b, ml=ML.BLOCK_SHARDED),
        "blk_f32_1024x1024": p1._cell((1, 1, 1024, 1024), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.BLOCK_SHARDED),
        "blk_nog_2048x1024": p1._cell((1, 1, 2048, 1024), ml=ML.BLOCK_SHARDED, gdtype=None),
        "wid_f32_256x2048": p1._cell((1, 1, 256, 2048), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.WIDTH_SHARDED),
        # WIDTH, large k with > 1 tile-row (k vs per-member work boundary)
        "wid_perf_256x7168": p1._perf((1, 1, 256, 7168), ML.WIDTH_SHARDED, ([256, 256], (7, 4))),
        "wid_512x1024_k32": p1._cell(
            (1, 1, 512, 1024), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [512, 32], "core_grid": (8, 4)}
        ),
        "wid_f32_1024x2048": p1._cell((1, 1, 1024, 2048), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.WIDTH_SHARDED),
        "wid_bf16_256x2048": p1._cell((1, 1, 256, 2048), ml=ML.WIDTH_SHARDED),
        # interleaved owner combine (k > 8) with > 1 tile-row per core
        "int_owner_640x32768": p1._cell((1, 1, 640, 32768)),
    }
)


def parse_variant(v):
    if v.startswith("base"):
        return None
    knobs = dict(
        PIPE_SUBBLOCKS=0,
        PIPE_ROWS=0,
        PIPE_LAG=1,
        PIPE_ROTATE=True,
        PIPE_GAMMA_LATE=None,
        PIPE_PARITY=False,
        PIPE_SCHED=0,
        PIPE_FLIP=False,
        SUMSQ_TAIL_FIX=True,
        PIPE_AUTO=False,
    )
    if v == "auto":  # the candidate rule
        knobs.update(PIPE_AUTO=True, PIPE_PARITY=True, PIPE_GAMMA_LATE=False)
        return knobs
    head, *suf = v.split("_")
    m = re.fullmatch(r"n(\d+)", head)
    if m:
        knobs["PIPE_SUBBLOCKS"] = int(m.group(1))
    m = re.fullmatch(r"r(\d+)", head)
    if m:
        knobs["PIPE_ROWS"] = int(m.group(1))
    for s_ in suf:
        if s_ == "lag0":
            knobs["PIPE_LAG"] = 0
        elif s_ == "norot":
            knobs["PIPE_ROTATE"] = False
        elif s_ == "gl":
            knobs["PIPE_GAMMA_LATE"] = True
        elif s_ == "ge":
            knobs["PIPE_GAMMA_LATE"] = False
        elif s_ == "nofix":
            knobs["SUMSQ_TAIL_FIX"] = False
        elif s_ == "ff":
            knobs["PIPE_SCHED"] = 2
        elif s_ == "flip":
            knobs["PIPE_FLIP"] = True
        elif s_ == "par":
            knobs["PIPE_PARITY"] = True
        else:
            raise ValueError(v)
    return knobs


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


_sel = os.environ.get("CP_CELLS", "blk_perf_7168x1024")
IDS = [c for c in _sel.split(",") if c]
VARIANTS = [v for v in os.environ.get("CP_VARIANTS", "base,n2").split(",") if v]
TAG = os.environ.get("CP_TAG", "scratch")
LOG = HERE / "runs" / f"{TAG}_labels.jsonl"


def _build(device, axes):
    import torch  # function-local: the ttnn tree forbids global torch imports

    shape = axes["inputs"][0]
    dtype, layout, ml, extras = axes["dtype"], axes["layout"], axes["memory_layout"], axes["extras"]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    W = shape[-1]
    g = None
    tg = None
    if axes["gamma_mode"] == "gamma":
        g = torch.randn(W, dtype=torch.float32).to(_TORCH_DTYPE[axes["gamma_dtype"]])
        tg = ttnn.from_torch(
            g.reshape(1, 1, 1, W),
            dtype=axes["gamma_dtype"],
            layout=axes["gamma_layout"],
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
    cfg.fp32_dest_acc_en = axes["fp32_dest_acc_en"]
    cfg.math_approx_mode = False
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif extras.get("shard_shape") is not None:
        mc = shard_config(
            extras["shard_shape"],
            extras["core_grid"],
            ml,
            layout=layout,
            dtype=dtype,
            orientation=extras.get("shard_orientation", ttnn.ShardOrientation.ROW_MAJOR),
            device=device,
        )
    else:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dtype, device=device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    expected = pytorch_rms_norm(x, gamma=g, epsilon=1e-6)
    return tx, tg, cfg, expected


@pytest.mark.parametrize("cell", IDS)
def test_combine_pipeline(device, cell, monkeypatch):
    import torch  # function-local: the ttnn tree forbids global torch imports

    axes = CELLS[cell]
    tx, tg, cfg, expected = _build(device, axes)
    out_mc = tx.memory_config() if axes["memory_layout"] != ML.INTERLEAVED else None
    base_out = None
    failures = []
    for v in VARIANTS:
        knobs = parse_variant(v) if v not in ("grad", "grad_off") else None
        if v in ("grad", "grad_off"):
            monkeypatch.setattr(gpd, "COMBINE_PIPELINE", v == "grad")
            monkeypatch.setattr(opmod, "create_program_descriptor", gpd.create_program_descriptor)
        elif knobs is None:
            monkeypatch.setattr(opmod, "create_program_descriptor", BASE_PD)
        else:
            for kk, vv in knobs.items():
                monkeypatch.setattr(ppd, kk, vv)
            monkeypatch.setattr(opmod, "create_program_descriptor", ppd.create_program_descriptor)
        # record the blocking the variant actually picked (pipelined or not)
        info = {}
        if v in ("grad", "grad_off") and axes["memory_layout"] != ML.INTERLEAVED:
            geo = gpd.derive_geometry(tx.shape)
            sz = gpd._Sizes(
                tx.dtype,
                tg.dtype if tg is not None else tx.dtype,
                axes["layout"] == RM,
                tg is not None,
                cfg.fp32_dest_acc_en,
            )
            blk = gpd.select_sharded_blocking(geo, sz, tx, gpd.cb_l1_budget(device))
            info = dict(
                k=blk.k,
                b=blk.block_row_tiles,
                pipe=bool(blk.opts.pipe_nb),
                crt=max(g_[0].core_rows for g_ in blk.groups),
            )
        if knobs is not None and axes["memory_layout"] != ML.INTERLEAVED:
            geo = ppd.derive_geometry(tx.shape)
            sz = ppd._Sizes(
                tx.dtype,
                tg.dtype if tg is not None else tx.dtype,
                axes["layout"] == RM,
                tg is not None,
                cfg.fp32_dest_acc_en,
            )
            blk = ppd.select_sharded_blocking(geo, sz, tx, ppd.cb_l1_budget(device))
            info = dict(
                k=blk.k,
                b=blk.block_row_tiles,
                pipe=bool(blk.opts.pipe_inv_pages),
                crt=max(g_[0].core_rows for g_ in blk.groups),
            )
        out = opmod.rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg, memory_config=out_mc)
        o = ttnn.to_torch(out).float()
        pcc = _pcc(o, expected.float())
        rec = dict(cell=cell, variant=v, pcc=pcc, max_abs_vs_torch=float((o - expected.float()).abs().max()), **info)
        if v == os.environ.get("CP_REF", "base"):
            base_out = o
        if base_out is not None:
            rec["max_abs_vs_base"] = float((o - base_out).abs().max())
            rec["pcc_vs_base"] = _pcc(o, base_out)
            d = (o - base_out).abs().reshape(-1, o.shape[-1])
            bad = (d > 0).nonzero()
            if len(bad):
                trs = sorted(set((bad[:, 0] // 32).tolist()))
                tcs = sorted(set((bad[:, 1] // 32).tolist()))
                rec["n_diff"] = int(len(bad))
                rec["diff_tile_rows"] = trs[:40]
                rec["diff_tile_cols"] = tcs[:40]
                # per-row: is the whole row off by a constant factor (inv_rms) or scattered?
                r0 = int(bad[0, 0])
                ob, bb = o.reshape(-1, o.shape[-1])[r0], base_out.reshape(-1, o.shape[-1])[r0]
                m = bb.abs() > 1e-3
                ratio = ob[m] / bb[m]
                rec["row0"] = r0
                rec["row0_ratio_min_max"] = [float(ratio.min()), float(ratio.max())]
                rec["rows_diff"] = len(set(bad[:, 0].tolist()))
        thr = axes["extras"].get(
            "pcc_threshold", {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}[axes["dtype"]]
        )
        rec["ok"] = pcc >= thr
        if not rec["ok"]:
            failures.append(rec)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("CP", json.dumps(rec))
        ttnn.deallocate(out)
    assert not failures, failures
