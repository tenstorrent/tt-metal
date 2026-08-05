# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off round 2: flat root vs a (K slot chunks x m row subsets) combine.

Correctness is the ONLY pass/fail.  Every variant's device kernel duration is read from
the profiler CSV afterwards by ``read_results.py`` (this file only emits the run manifest,
in execution order).

Run:
  scripts/run_safe_pytest.sh --profile <this file> -k focus
  scripts/run_safe_pytest.sh --profile <this file> -k sweep_a
  scripts/run_safe_pytest.sh --profile <this file> -k sweep_b
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent

# Load bench.py BY PATH, deliberately NOT as `ttnn.operations....bench`.
# `ttnn/ttnn/operations/__init__.py` does `pkgutil.walk_packages(__path__)` and EXECUTES
# every module of every subpackage at `import ttnn`; no __init__.py in perf_experiments/
# keeps this scratch bench invisible to that walk (see perf_experiments/README.md).
_spec = importlib.util.spec_from_file_location("_hg2_bench", HERE / "bench.py")
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
# (id, group_size, num_groups, box_w, block_rows, num_rows)
# ---------------------------------------------------------------------------
FOCUS_CONFIGS = [
    # THE focus shape: (1,1,8192,1024) BLOCK_SHARDED 8x8 -> 8 groups of 8, 32 tile-rows
    # per core, BLOCK_ROWS = 8 -> 4 combine rounds.
    ("focus_g8_ng8_br8_r32", 8, 8, None, 8, 32),
]

# GROUP_SIZE x rows-per-block sweep.  Split in two so one device run stays bounded.
SWEEP_A = [
    ("g8_ng8_br1_r1", 8, 8, None, 1, 1),  # decode, 64 cores
    ("g8_ng8_br32_r32", 8, 8, None, 32, 32),  # one round of 32 rows
    ("g4_ng8_br8_r32", 4, 8, None, 8, 32),  # round 1's EARNED carve-out, re-tested
    ("g4_ng8_br1_r1", 4, 8, None, 1, 1),
    ("g16_ng1_br1_r1", 16, 1, 8, 1, 1),
    ("g16_ng1_br8_r32", 16, 1, 8, 8, 32),
]
SWEEP_B = [
    ("g28_ng1_br1_r1", 28, 1, 8, 1, 1),  # the op's WIDTH-shard decode, NON-RECTANGULAR
    ("g28_ng1_br8_r32", 28, 1, 8, 8, 32),
    ("g32_ng1_br1_r1", 32, 1, 8, 1, 1),  # the op's (1,1,32,5120) decode profile
    ("g32_ng1_br8_r32", 32, 1, 8, 8, 32),
    ("g32_ng1_br32_r32", 32, 1, 8, 32, 32),
]


# The ROW-SPLIT crossover probe.  The split buys (1 - 1/m) of the root's per-round fold and
# pays one extra hop (worker -> root's mcast buffer) whose cost is ~fixed per round, so the
# crossover must sit at the SMALLEST per-round fold that still has a splittable row axis:
# BLOCK_ROWS = 2.  If m = 2 still wins at G = 4 / BLOCK_ROWS = 2 there is no crossover
# anywhere in the op's space.
SWEEP_C = [
    ("g4_ng8_br2_r32", 4, 8, None, 2, 32),
    ("g8_ng8_br2_r32", 8, 8, None, 2, 32),
    ("g32_ng1_br2_r32", 32, 1, 8, 2, 32),
]

# The focus shape's three load-bearing points, repeated so the ~2-3% noise-band call on the
# generic-path control (grid_k1_m1 vs flat) is a median and not a single sample.
FOCUS_REPEAT = [("flat", 1, 1), ("grid_k1_m1", 1, 1), ("grid_k1_m8", 1, 8)]


def _candidates(group_size, block_rows):
    """The (K, m) menu for one geometry -> [(label, k, m)], flat first.

    Curated rather than exhaustive: the corners of the policy space (pure row split, pure
    slot tree) plus the diagonal combined points, so the ONE rule can be read off the
    numbers instead of a lookup table.
    """
    out = [("flat", 1, 1)]
    pairs = [(1, 1)]  # the generic-path overhead control
    for mm in (2, 4, 8, 16, 32):
        pairs.append((1, mm))
    for kk in (2, 4, 7, 8, 14, 16):
        pairs.append((kk, 1))
    for kk in (2, 4, 7, 8):
        for mm in (2, 4, 8):
            pairs.append((kk, mm))
    seen = set()
    for kk, mm in pairs:
        if (kk, mm) in seen or not B.legal(group_size, block_rows, kk, mm):
            continue
        seen.add((kk, mm))
        out.append((f"grid_k{kk}_m{mm}", kk, mm))
    return out


def _make_tensors(device, geo, num_rows, seed):
    import torch

    ncores = len(geo.cores)
    rows = num_rows * B.TILE
    torch.manual_seed(seed)
    # A strong per-row scale (spans ~20x) so the group sums are well spread and the PCC of
    # the finalized stat is well conditioned.
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


def _run_one(device, cfg, cand, records):
    cid, gs, ng, box_w, block_rows, num_rows = cfg
    label, k, m = cand
    variant = B.V_FLAT if label == "flat" else B.V_GRID
    geo = B.build_geometry(device, group_size=gs, num_groups=ng, box_w=box_w)
    x, out, vals = _make_tensors(device, geo, num_rows, seed=1234)
    W = gs * 128
    inv_w = 1.0 / W
    eps = 1e-6
    # A variant may simply not FIT in L1 -- the flat root's gather ring is
    # GROUP_SIZE * BLOCK_ROWS fp32 tiles, which at BLOCK_ROWS = 32 exceeds a core's L1 on
    # its own.  That is a real (and load-bearing) result, not a bench bug: record it as
    # `l1_oom` and carry on rather than aborting the sweep.
    try:
        prog = B.build_program(
            device,
            x,
            out,
            geo,
            variant=variant,
            k=k,
            m=m,
            block_rows=block_rows,
            num_rows=num_rows,
            inv_w=inv_w,
            eps=eps,
            compute_config=_compute_config(),
        )
        res = ttnn.generic_op([x, out], prog)
    except RuntimeError as e:
        if "L1" not in str(e) and "allocate" not in str(e).lower():
            raise
        print(f"[bench2] {cid:24s} {label:16s} L1_OOM ({B.l1_bytes(gs, block_rows, k, m, variant)//1024} kB)")
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        return
    got = ttnn.to_torch(res).reshape(len(geo.cores), num_rows * B.TILE, B.TILE)
    ref, active = _reference(geo, vals, inv_w, eps)

    # Only the lanes the datapath actually carries: GATHER_FACES == 2 ships faces 0 and 2
    # (columns 0..15) and D17's finalize scopes to the EVEN lanes of those, so the columns
    # the op's consumer (mul<BroadcastDim::Col>, column 0) reads are columns 0,2,..,14 --
    # identical lane set on EVERY variant, so the gate compares like with like.
    act = sorted(active)
    g = got[act][:, :, 0:16:2]
    r = ref[act].unsqueeze(-1).expand(-1, -1, 8)
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
        m=m,
        pcc=pcc,
        rel_rms=relrms,
        l1_bytes=B.l1_bytes(gs, block_rows, k, m, variant),
        sems=B.num_semaphores(k, m),
    )
    records.append(rec)
    print(f"[bench2] {cid:24s} {label:16s} pcc={pcc:.6f} rel_rms={relrms:.5f} l1={rec['l1_bytes']//1024}kB")
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    assert pcc >= PCC_GATE, f"{cid}/{label}: pcc {pcc} < {PCC_GATE}"
    assert relrms <= RELRMS_GATE, f"{cid}/{label}: rel-RMS {relrms} > {RELRMS_GATE}"


WARMUP_CFG = ("warmup", 4, 1, 4, 1, 1)


def _drive(device, configs, tag):
    records = []
    # One throwaway program FIRST, so no measured variant is this process's first device
    # launch (dispatch / L1 first-touch would otherwise land on it).  Its row is in the
    # manifest and is ignored by the reader.
    warm = []
    _run_one(device, WARMUP_CFG, ("flat", 1, 1), warm)
    try:
        for cfg in configs:
            for cand in _candidates(cfg[1], cfg[4]):
                _run_one(device, cfg, cand, records)
    finally:
        mode = "a" if os.environ.get("HG2_APPEND") == "1" else "w"
        with open(MANIFEST, mode) as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[bench2] manifest ({tag}, {len(records)} runs) -> {MANIFEST}")


def test_focus(device):
    _drive(device, FOCUS_CONFIGS, "focus")


def test_sweep_a(device):
    _drive(device, SWEEP_A, "sweep_a")


def test_sweep_b(device):
    _drive(device, SWEEP_B, "sweep_b")


def test_sweep_c(device):
    _drive(device, SWEEP_C, "sweep_c")


def test_focus_repeat(device):
    """3 fresh programs per point, so the median settles the noise-band calls."""
    cfg = FOCUS_CONFIGS[0]
    records = []
    warm = []
    _run_one(device, WARMUP_CFG, ("flat", 1, 1), warm)
    try:
        for _ in range(3):
            for cand in FOCUS_REPEAT:
                _run_one(device, cfg, cand, records)
    finally:
        with open(MANIFEST, "w") as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[bench2] manifest (focus_repeat, {len(records)} runs) -> {MANIFEST}")


@pytest.mark.parametrize("cid", [c[0] for c in FOCUS_CONFIGS + SWEEP_A + SWEEP_B + SWEEP_C])
def test_single(device, cid):
    """One config, all candidates -- for bring-up under --dev."""
    cfg = next(c for c in FOCUS_CONFIGS + SWEEP_A + SWEEP_B + SWEEP_C if c[0] == cid)
    _drive(device, [cfg], cid)
