# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: a k-ary SLOT tree (row split OFF) vs the op's flat fan-in + D22 root.

Correctness is the ONLY pass/fail.  Every variant's device kernel duration is read from the
profiler CSV afterwards by ``read_results.py`` (this file only emits the run manifest, in
execution order).

Run:
  scripts/run_safe_pytest.sh --profile <this file> -k focus
  scripts/run_safe_pytest.sh --profile <this file> -k width_decode
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
_spec = importlib.util.spec_from_file_location("_stg_bench", HERE / "bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
MANIFEST = HERE / "last_run_manifest.jsonl"

PCC_GATE = 0.9995  # the focus case's soft threshold
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
    # THE focus shape: (1,1,8192,1024) BLOCK_SHARDED shard [1024,128] grid (8,8) -> 8 groups
    # of 8, 32 tile-rows per core, BLOCK_ROWS = 8 -> 4 combine rounds.
    ("focus_g8_br8_r32", 8, 8, None, 8, 32),
    # THE COMPOSABILITY CELL.  `compact_partial_transpose` collapses a sender's BLOCK_ROWS
    # partials into ONE tile per round, so in the world where it lands the focus geometry's
    # transport is 1 tile per sender per round for the SAME 4 rounds.  That is exactly
    # BLOCK_ROWS = 1 with num_rows = num_blocks = 4.  It is also the cell where the row split
    # is INEXPRESSIBLE (m <= BLOCK_ROWS == 1), i.e. the tree is the only lever left.
    ("focus_compact_g8_br1_r4", 8, 8, None, 1, 4),
]

# The two WIDTH-sharded decode targets still off their reference, where BLOCK_ROWS == 1 makes
# the row split inexpressible and a flat fan-in of GROUP_SIZE partials into one root is the
# whole latency.
WIDTH_DECODE = [
    ("w5120_g32_br1_r1", 32, 1, 8, 1, 1),  # (1,1,32,5120) shard [32,160] grid (8,4) = 32 cores
    ("w7168_g28_br1_r1", 28, 1, 8, 1, 1),  # (1,1,32,7168) shard [32,256] grid (7,4) = 28 cores
]

# THE COMPOSABILITY CELLS AT A GROUP WIDE ENOUGH TO WIN.  `focus_compact_g8_br1_r4` shows the
# compaction-composed transport at GROUP_SIZE = 8 (where the tree loses anyway); these are the
# same 1-tile-per-sender-per-round transport over MULTIPLE rounds at the group sizes where the
# tree does win, i.e. the world the tree would actually have to live in if
# `compact_partial_transpose` lands on a WIDTH/BLOCK geometry with a wide group.
COMPOSE = [
    ("compact_g32_br1_r4", 32, 1, 8, 1, 4),
    ("compact_g16_br1_r4", 16, 1, 8, 1, 4),
]

SWEEP_A = [
    ("g4_br1_r1", 4, 8, None, 1, 1),
    ("g4_br8_r32", 4, 8, None, 8, 32),
    ("g4_br32_r32", 4, 8, None, 32, 32),
    ("g8_br1_r1", 8, 8, None, 1, 1),
    ("g8_br32_r32", 8, 8, None, 32, 32),
    ("g9_br1_r1", 9, 1, 8, 1, 1),  # ODD group: the evenness pad slot
    ("g9_br8_r32", 9, 1, 8, 8, 32),
]
SWEEP_B = [
    ("g16_br1_r1", 16, 1, 8, 1, 1),
    ("g16_br8_r32", 16, 1, 8, 8, 32),
    ("g28_br8_r32", 28, 1, 8, 8, 32),
    ("g32_br8_r32", 32, 1, 8, 8, 32),
]

# The arity menu per GROUP_SIZE.  Curated, not exhaustive: for each GROUP_SIZE the 2-level
# shapes around sqrt(GROUP_SIZE) (both orders, since the ROOT pays f_first + f_last while a
# level-0 gatherer pays only f_first), one lopsided 2-level shape, and the 3-/4-level shapes
# that minimise sum(F) -- which is what "log_k(GROUP_SIZE) levels" actually asks for.
# `(GROUP_SIZE,)` is the flat corner through the generic path: the overhead control.
ARITY_MENU = {
    4: [(4,), (2, 2)],
    8: [(8,), (4, 2), (2, 4), (2, 2, 2)],
    9: [(9,), (3, 3), (2, 5), (2, 2, 3)],
    16: [(16,), (4, 4), (8, 2), (2, 8), (2, 2, 4), (2, 2, 2, 2)],
    28: [(28,), (7, 4), (4, 7), (2, 14), (4, 8), (2, 2, 7), (3, 3, 4)],
    32: [(32,), (8, 4), (4, 8), (2, 16), (6, 6), (4, 4, 2), (2, 4, 4), (2, 2, 2, 4)],
}


def _candidates(group_size):
    out = [("flat", None)]
    for a in ARITY_MENU[group_size]:
        assert B.legal(group_size, a), f"arity {a} illegal at G={group_size}"
        out.append(("tree_" + "x".join(str(f) for f in a), a))
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
    label, arity = cand
    variant = B.V_FLAT if arity is None else B.V_TREE
    ar = (gs,) if arity is None else arity
    geo = B.build_geometry(device, group_size=gs, num_groups=ng, box_w=box_w)
    x, out, vals = _make_tensors(device, geo, num_rows, seed=1234)
    W = gs * 128
    inv_w = 1.0 / W
    eps = 1e-6
    # A variant may simply not FIT in L1 -- the flat root's GATHER_SLOTS * BLOCK_ROWS fp32
    # gather ring exceeds a core's L1 on its own at BLOCK_ROWS = 32.  That is a real (and
    # load-bearing) result, not a bench bug: record it as `l1_oom` and carry on.
    try:
        prog = B.build_program(
            device,
            x,
            out,
            geo,
            variant=variant,
            arity=ar,
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
        print(f"[stg] {cid:24s} {label:16s} L1_OOM ({B.l1_bytes(gs, block_rows, ar, variant)//1024} kB)")
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        return
    got = ttnn.to_torch(res).reshape(len(geo.cores), num_rows * B.TILE, B.TILE)
    ref, active = _reference(geo, vals, inv_w, eps)

    # Only the lanes the datapath actually carries: GATHER_FACES == 2 ships faces 0 and 2
    # (columns 0..15) and D17's finalize scopes to the EVEN lanes of those, so the columns the
    # op's consumer (mul<BroadcastDim::Col>, column 0) reads are 0,2,..,14 -- identical lane
    # set on EVERY variant, so the gate compares like with like.
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
        arity=list(ar),
        levels=len(ar) if variant == B.V_TREE else 1,
        pcc=pcc,
        rel_rms=relrms,
        l1_bytes=B.l1_bytes(gs, block_rows, ar, variant),
        sems=B.num_semaphores(ar, variant),
    )
    records.append(rec)
    print(f"[stg] {cid:24s} {label:16s} pcc={pcc:.6f} rel_rms={relrms:.5f} l1={rec['l1_bytes']//1024}kB")
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    assert pcc >= PCC_GATE, f"{cid}/{label}: pcc {pcc} < {PCC_GATE}"
    assert relrms <= RELRMS_GATE, f"{cid}/{label}: rel-RMS {relrms} > {RELRMS_GATE}"


WARMUP_CFG = ("warmup", 4, 1, 4, 1, 1)


def _drive(device, configs, tag, repeats=1):
    records = []
    # One throwaway program FIRST, so no measured variant is this process's first device
    # launch (dispatch / L1 first-touch would otherwise land on it).  Its row is in the
    # manifest and is ignored by the reader.
    warm = []
    _run_one(device, WARMUP_CFG, ("flat", None), warm)
    try:
        for _ in range(repeats):
            for cfg in configs:
                for cand in _candidates(cfg[1]):
                    _run_one(device, cfg, cand, records)
    finally:
        mode = "a" if os.environ.get("STG_APPEND") == "1" else "w"
        with open(MANIFEST, mode) as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[stg] manifest ({tag}, {len(records)} runs) -> {MANIFEST}")


def test_focus(device):
    _drive(device, FOCUS_CONFIGS, "focus")


def test_width_decode(device):
    _drive(device, WIDTH_DECODE, "width_decode")


def test_sweep_a(device):
    _drive(device, SWEEP_A, "sweep_a")


def test_sweep_b(device):
    _drive(device, SWEEP_B, "sweep_b")


def test_compose(device):
    _drive(device, COMPOSE, "compose")


def test_repeat(device):
    """3 fresh programs per point on the headline geometries, so any call inside the ~2-3%
    noise band is settled by a median instead of a single sample."""
    _drive(device, [FOCUS_CONFIGS[0], WIDTH_DECODE[0], FOCUS_CONFIGS[1]], "repeat", repeats=3)


_ALL = FOCUS_CONFIGS + WIDTH_DECODE + COMPOSE + SWEEP_A + SWEEP_B


@pytest.mark.parametrize("cid", [c[0] for c in _ALL])
def test_single(device, cid):
    """One config, all candidates -- for bring-up under --dev."""
    cfg = next(c for c in _ALL if c[0] == cid)
    _drive(device, [cfg], cid)
