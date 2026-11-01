# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""combine_pipeline PORT bench: HEAD op vs the pipeline ported onto HEAD (this dir's descriptor + kernels).

Not collected by a directory run (no `test_` prefix). Run via ./run.sh <tag> <cells> <variants>.

Variants (CP_VARIANTS, comma list; dispatched in the given order per cell):
  base        the UNMODIFIED op at HEAD (ttnn/ttnn/operations/rms_norm, honest baseline)
  port        the port with its rule (graduation candidate)
  port_off    the port with COMBINE_PIPELINE = False (must equal HEAD: same path)
  port_n<N>   the port, COMBINE_PIPELINE_SUBBLOCKS = N (rule otherwise unchanged)
  port_f<N>   the port forced on with N sub-blocks (MIN_ROWS = 1, K_PER_WORK = inf) — rule-boundary sweep
Every dispatch is checked vs torch (PCC) and vs the `base` output of the same cell (max abs diff, expected 0:
the pipeline is bit-exact); records go to runs/<CP_TAG>_labels.jsonl in dispatch order (GenericOp rows of the
profiler CSV map 1:1).
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
REPO = HERE.resolve().parents[6]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


p1 = _load("cpp_perf_p1_probe", REPO / "tests/ttnn/unit_tests/operations/rms_norm/perf_p1_rms_norm.py")

import ttnn.operations.rms_norm.rms_norm  # noqa: E402,F401

opmod = sys.modules["ttnn.operations.rms_norm.rms_norm"]
BASE_PD = opmod.create_program_descriptor
ppd = _load("cpp_port_pd", HERE / "rms_norm_program_descriptor.py")

ML = ttnn.TensorMemoryLayout
RM = ttnn.ROW_MAJOR_LAYOUT
HIFI2 = ttnn.MathFidelity.HiFi2

CELLS = dict(p1.CELLS)
CELLS.update(
    {
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
            extras={"math_fidelity": HIFI2, "shard_shape": [736, 128], "core_grid": (8, 10)},
        ),
        "blk_tile_2048x256": p1._cell((1, 1, 2048, 256), ml=ML.BLOCK_SHARDED),
        "wid_tile_1024x1000": p1._cell(
            (1, 1, 1024, 1000), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [1024, 128], "core_grid": (8, 1)}
        ),
        "blk_bf8_2048x1024": p1._cell((1, 1, 2048, 1024), dtype=ttnn.bfloat8_b, ml=ML.BLOCK_SHARDED),
        "blk_f32_1024x1024": p1._cell((1, 1, 1024, 1024), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.BLOCK_SHARDED),
        "blk_nog_2048x1024": p1._cell((1, 1, 2048, 1024), ml=ML.BLOCK_SHARDED, gdtype=None),
        "blk_grm_2048x1024": p1._cell((1, 1, 2048, 1024), ml=ML.BLOCK_SHARDED, glayout=RM),
        "blk_gbf8_2048x1024": p1._cell((1, 1, 2048, 1024), ml=ML.BLOCK_SHARDED, gdtype=ttnn.bfloat8_b),
        "blk_colmajor_2048x1024": p1._cell(
            (1, 1, 2048, 1024),
            ml=ML.BLOCK_SHARDED,
            extras={
                "shard_shape": [256, 128],
                "core_grid": (8, 8),
                "shard_orientation": ttnn.ShardOrientation.COL_MAJOR,
            },
        ),
        # crt = 7, s = 3 -> sub-blocks 3,3,1 (one-tile-row last sub-block, rotated owner 6 % 8)
        "blk_last1_224x1024": p1._cell(
            (1, 1, 224, 1024), ml=ML.BLOCK_SHARDED, extras={"shard_shape": [224, 128], "core_grid": (8, 1)}
        ),
        "wid_f32_256x2048": p1._cell((1, 1, 256, 2048), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.WIDTH_SHARDED),
        "wid_perf_256x7168": p1._perf((1, 1, 256, 7168), ML.WIDTH_SHARDED, ([256, 256], (7, 4))),
        "wid_512x1024_k32": p1._cell(
            (1, 1, 512, 1024), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [512, 32], "core_grid": (8, 4)}
        ),
        "wid_f32_1024x2048": p1._cell((1, 1, 1024, 2048), dtype=ttnn.float32, gdtype=ttnn.float32, ml=ML.WIDTH_SHARDED),
        "wid_bf16_256x2048": p1._cell((1, 1, 256, 2048), ml=ML.WIDTH_SHARDED),
        "int_owner_640x32768": p1._cell((1, 1, 640, 32768)),
        # deep shards (rule tuning): crt = 64 at w = 4 / w = 2, crt = 32 at w = 1, crt = 3
        "blk_deep_8192x1024": p1._perf((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([2048, 128], (8, 4))),
        "wid_tile_2048x512": p1._cell(
            (1, 1, 2048, 512), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [2048, 64], "core_grid": (8, 1)}
        ),
        "wid_tile_1024x256": p1._cell(
            (1, 1, 1024, 256), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [1024, 32], "core_grid": (8, 1)}
        ),
        "wid_2048x1024_k32": p1._cell(
            (1, 1, 2048, 1024), ml=ML.WIDTH_SHARDED, extras={"shard_shape": [2048, 32], "core_grid": (8, 4)}
        ),
        "blk_crt3_768x1024": p1._cell(
            (1, 1, 768, 1024), ml=ML.BLOCK_SHARDED, extras={"shard_shape": [96, 128], "core_grid": (8, 8)}
        ),
    }
)

KNOB_DEFAULTS = {k_: getattr(ppd, k_) for k_ in dir(ppd) if k_.startswith("COMBINE_PIPELINE")}


def parse_variant(v):
    if v == "base":
        return None
    knobs = dict(KNOB_DEFAULTS)
    if v == "port":
        return knobs
    if v == "port_off":
        knobs["COMBINE_PIPELINE"] = False
        return knobs
    if v.startswith("port~"):  # port~SUBBLOCKS=3~MAX_ROWS=12 -> COMBINE_PIPELINE_<NAME> = int(val)
        for kv in v.split("~")[1:]:
            name, val = kv.split("=")
            key = "COMBINE_PIPELINE_" + name
            assert key in knobs, key
            knobs[key] = int(val)
        return knobs
    m = re.fullmatch(r"port_n(\d+)", v)
    if m:
        knobs["COMBINE_PIPELINE_SUBBLOCKS"] = int(m.group(1))
        return knobs
    m = re.fullmatch(r"port_f(\d+)", v)
    if m:
        knobs.update(
            COMBINE_PIPELINE_SUBBLOCKS=int(m.group(1)), COMBINE_PIPELINE_MIN_ROWS=1, COMBINE_PIPELINE_K_PER_WORK=10**9
        )
        return knobs
    raise ValueError(v)


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


_sel = os.environ.get("CP_CELLS", "blk_perf_7168x1024")
IDS = [c for c in _sel.split(",") if c]
VARIANTS = [v for v in os.environ.get("CP_VARIANTS", "port,base").split(",") if v]
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
def test_port(device, cell, monkeypatch):
    axes = CELLS[cell]
    tx, tg, cfg, expected = _build(device, axes)
    out_mc = tx.memory_config() if axes["memory_layout"] != ML.INTERLEAVED else None
    outs = {}
    recs = []
    for v in VARIANTS:
        knobs = parse_variant(v)
        info = {}
        if knobs is None:
            monkeypatch.setattr(opmod, "create_program_descriptor", BASE_PD)
        else:
            for kk, vv in knobs.items():
                monkeypatch.setattr(ppd, kk, vv)
            monkeypatch.setattr(opmod, "create_program_descriptor", ppd.create_program_descriptor)
            if axes["memory_layout"] != ML.INTERLEAVED:
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
                    nb=blk.opts.pipe_nb,
                    w=blk.slice_w_tiles,
                    crt=max(g_[0].core_rows for g_ in blk.groups),
                )
        out = opmod.rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg, memory_config=out_mc)
        o = ttnn.to_torch(out).float()
        ttnn.deallocate(out)
        outs[v] = o
        rec = dict(
            cell=cell,
            variant=v,
            pcc=_pcc(o, expected.float()),
            max_abs_vs_torch=float((o - expected.float()).abs().max()),
            **info,
        )
        thr = axes["extras"].get(
            "pcc_threshold", {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}[axes["dtype"]]
        )
        rec["ok"] = rec["pcc"] >= thr
        recs.append(rec)
    base = outs.get("base")
    failures = []
    for rec in recs:
        if base is not None:
            o = outs[rec["variant"]]
            rec["max_abs_vs_base"] = float((o - base).abs().max())
            rec["n_diff_vs_base"] = int(((o - base) != 0).sum())
        if not rec["ok"]:
            failures.append(rec)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("CP", json.dumps(rec))
    assert not failures, failures
