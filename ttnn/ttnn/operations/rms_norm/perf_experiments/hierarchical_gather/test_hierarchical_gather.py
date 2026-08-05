# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: flat-root vs hierarchical rms_norm combine.

Correctness is the ONLY pass/fail.  Every variant's device kernel duration is read
from the profiler CSV afterwards by ``read_results.py`` (this file only emits the
run manifest, in execution order).

Run:
  scripts/run_safe_pytest.sh --profile <this file> -k focus
  scripts/run_safe_pytest.sh --profile <this file> -k sweep
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent

# Load bench.py BY PATH, deliberately NOT as `ttnn.operations....hierarchical_gather.bench`.
# `ttnn/ttnn/operations/__init__.py` does `pkgutil.walk_packages(__path__)` and EXECUTES
# every module of every subpackage at `import ttnn` -- so an `__init__.py` anywhere under
# perf_experiments/ makes this scratch bench run on every ttnn import in the repo.  No
# __init__.py here; a path import keeps the experiment invisible to that walk.
_spec = importlib.util.spec_from_file_location("_hg_bench", HERE / "bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
MANIFEST = HERE / "last_run_manifest.jsonl"

PCC_GATE = 0.9995
RELRMS_GATE = 0.04


def _compute_config():
    """The op's PINNED precision contract -- FIXED, identical for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return 1.0
    return float((a @ b).item() / denom)


def _rel_rms(got, ref):
    return float((got.double() - ref.double()).pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# configs: the geometries the op actually builds
# ---------------------------------------------------------------------------
# (id, group_size, num_groups, box_w, block_rows, num_rows)
FOCUS_CONFIGS = [
    # the perf-flagged focus shape (1,1,8192,1024) BLOCK_SHARDED 8x8: 8 groups of 8,
    # 32 tile-rows per core, BLOCK_ROWS=10 -> 4 combine rounds (10,10,10,2).
    ("focus_g8_ng8_r32b10", 8, 8, None, 10, 32),
    # the big-group decode case (1,1,32,5120) WIDTH_SHARDED 8x4: ONE group of 32,
    # one tile-row, one round.
    ("secondary_g32_ng1_r1b1", 32, 1, 8, 1, 1),
]

SWEEP_CONFIGS = [
    ("g4_ng8_r1b1", 4, 8, None, 1, 1),
    ("g4_ng8_r32b10", 4, 8, None, 10, 32),
    ("g8_ng8_r1b1", 8, 8, None, 1, 1),
    ("g8_ng1_r32b10", 8, 1, 8, 10, 32),  # single-group control for the focus
    ("g9_ng8_r1b1", 9, 8, None, 1, 1),
    ("g9_ng8_r32b10", 9, 8, None, 10, 32),
    ("g16_ng1_r1b1", 16, 1, 8, 1, 1),
    ("g16_ng1_r16b4", 16, 1, 8, 4, 16),
    ("g28_ng1_r1b1_packed", 28, 1, 8, 1, 1),  # NON-RECTANGULAR: 28 in an 8x4 box
    ("g32_ng1_r16b4", 32, 1, 8, 4, 16),
]

VARIANT_NAMES = [
    "flat",
    "tree_k2",
    "tree_k4",
    "tree_k8",
    "tree_ksqrt",
    "tree_grid_axis",
    "rowsplit_wmax",
    "rowsplit_w4",
]


def _resolve_variants(geo, block_rows):
    """Dedup the variant menu for one geometry -> [(label, variant, k, w_max)]."""
    out = []
    seen = set()
    for name in VARIANT_NAMES:
        if name == "tree_grid_axis":
            k = B.grid_axis_k(geo)
            spec = None if k is None else (B.V_TREE, k, 0)
        elif name == "rowsplit_wmax":
            w = min(block_rows, geo.group_size)
            spec = None if w < 2 else (B.V_ROWSPLIT, 1, w)
        else:
            spec = B.variant_spec(name, geo.group_size, block_rows)
        if spec is None:
            continue
        if spec in seen:
            continue
        seen.add(spec)
        label = name
        if name in ("tree_ksqrt", "tree_grid_axis"):
            label = f"{name}(k={spec[1]})"
        if name == "rowsplit_wmax":
            label = f"rowsplit_w{spec[2]}"
        out.append((label, *spec))
    return out


def _make_tensors(device, geo, num_rows, seed):
    import torch

    ncores = len(geo.cores)
    rows = num_rows * B.TILE
    torch.manual_seed(seed)
    # A strong per-row scale (spans ~20x) so the group sums are well spread and the
    # PCC of the finalized stat is well conditioned.
    row_scale = torch.exp(torch.empty(rows).uniform_(-1.5, 1.5))
    vals = row_scale.unsqueeze(0) * (0.5 + torch.rand(ncores, rows))
    vals = vals * 128.0  # a realistic per-core sum(x^2) over a 128-element slice
    x_t = vals.unsqueeze(-1).expand(ncores, rows, B.TILE).reshape(1, 1, ncores * rows, B.TILE).contiguous()

    from eval.sharding import shard_config

    mc = shard_config(
        [rows, B.TILE],
        geo.core_range_set,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
    )
    x = ttnn.from_torch(x_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    out = ttnn.from_torch(
        torch.zeros_like(x_t), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    return x, out, vals


def _reference(geo, vals, inv_w, eps):
    """ref[core_index] = the finalized stat of the group that core belongs to."""
    import torch

    index_of = {(c.x, c.y): i for i, c in enumerate(geo.cores)}
    ref = torch.zeros_like(vals)
    for group in geo.groups:
        idxs = [index_of[(c.x, c.y)] for c in group]
        total = vals[idxs, :].double().sum(0)
        stat = torch.rsqrt(total * inv_w + eps)
        for i in idxs:
            ref[i, :] = stat.float()
    return ref, {index_of[(c.x, c.y)] for g in geo.groups for c in g}


def _run_one(device, cfg, variant_row, records):
    cid, gs, ng, box_w, block_rows, num_rows = cfg
    label, variant, k, w_max = variant_row
    geo = B.build_geometry(device, group_size=gs, num_groups=ng, box_w=box_w)
    x, out, vals = _make_tensors(device, geo, num_rows, seed=1234)
    W = gs * 128
    inv_w = 1.0 / W
    eps = 1e-6
    prog = B.build_program(
        device,
        x,
        out,
        geo,
        variant=variant,
        k=k,
        w_max=w_max,
        block_rows=block_rows,
        num_rows=num_rows,
        inv_w=inv_w,
        eps=eps,
        compute_config=_compute_config(),
    )
    res = ttnn.generic_op([x, out], prog)
    got = ttnn.to_torch(res).reshape(len(geo.cores), num_rows * B.TILE, B.TILE)
    ref, active = _reference(geo, vals, inv_w, eps)

    # Only the lanes the datapath actually carries: GATHER_FACES == 2 ships faces 0
    # and 2, i.e. columns 0..15 (the op's consumer reads column 0).  Columns 16..31
    # are the boot-zeroed faces by construction on EVERY variant.
    act = sorted(active)
    g = got[act][:, :, :16]
    r = ref[act].unsqueeze(-1).expand(-1, -1, 16)
    pcc = _pcc(g, r)
    relrms = _rel_rms(g, r)
    rec = dict(
        config=cid,
        variant=label,
        group_size=gs,
        num_groups=ng,
        block_rows=block_rows,
        num_rows=num_rows,
        k=k,
        w_max=w_max,
        pcc=pcc,
        rel_rms=relrms,
    )
    records.append(rec)
    print(f"[bench] {cid:24s} {label:20s} pcc={pcc:.6f} rel_rms={relrms:.5f}")
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    assert pcc >= PCC_GATE, f"{cid}/{label}: pcc {pcc} < {PCC_GATE}"
    assert relrms <= RELRMS_GATE, f"{cid}/{label}: rel-RMS {relrms} > {RELRMS_GATE}"


WARMUP_CFG = ("warmup", 4, 1, 4, 1, 1)


def _drive(device, configs, tag):
    records = []
    # One throwaway program FIRST, so no measured variant is this process's first
    # device launch (dispatch/L1 first-touch would otherwise land on it).  Its row
    # is in the manifest and simply ignored by the reader.
    warm = []
    _run_one(device, WARMUP_CFG, ("flat", B.V_FLAT, 1, 0), warm)
    try:
        for cfg in configs:
            geo = B.build_geometry(device, group_size=cfg[1], num_groups=cfg[2], box_w=cfg[3])
            for vrow in _resolve_variants(geo, cfg[4]):
                _run_one(device, cfg, vrow, records)
    finally:
        mode = "a" if os.environ.get("HG_APPEND") == "1" else "w"
        with open(MANIFEST, mode) as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[bench] manifest ({tag}, {len(records)} runs) -> {MANIFEST}")


def test_focus(device):
    _drive(device, FOCUS_CONFIGS, "focus")


def test_sweep(device):
    _drive(device, SWEEP_CONFIGS, "sweep")


@pytest.mark.parametrize("cid", [c[0] for c in FOCUS_CONFIGS + SWEEP_CONFIGS])
def test_single(device, cid):
    """One config, all variants -- for bring-up under --dev."""
    cfg = next(c for c in FOCUS_CONFIGS + SWEEP_CONFIGS if c[0] == cid)
    _drive(device, [cfg], cid)
