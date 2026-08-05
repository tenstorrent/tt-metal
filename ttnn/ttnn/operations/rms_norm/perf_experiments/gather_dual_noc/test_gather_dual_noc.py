# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: the rms_norm cross-core combine on ONE NoC vs on TWO.

Correctness is the ONLY pass/fail.  Every variant is gated three ways:
  * pcc   vs a torch reference (the focus case's soft threshold, 0.9995);
  * rel-RMS vs the same reference (this op has twice been bitten by errors that held
    pcc >= 0.9997 and showed ONLY in rel-RMS);
  * BIT-EXACTNESS vs the `base` run of the SAME config.  This idea only changes WHICH RISC
    issues a write -- the arithmetic is untouched -- so anything but bit-identical output is
    a RACE (a torn partial, or a lost/early semaphore increment), not a precision effect.

Every variant's device kernel duration is read from the profiler CSV afterwards by
``read_results.py`` (this file only emits the run manifest, in execution order).

Run:
  scripts/run_safe_pytest.sh --profile <this file> -k focus_menu
  scripts/run_safe_pytest.sh --profile <this file> -k sweep_br
  scripts/run_safe_pytest.sh --profile <this file> -k sweep_g
  scripts/run_safe_pytest.sh --profile <this file> -k sweep_decode
  scripts/run_safe_pytest.sh --profile <this file> -k load
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent

# Load bench.py BY PATH, deliberately NOT as `ttnn.operations....bench` -- see
# perf_experiments/README.md (walk_packages would execute it at every `import ttnn`).
_spec = importlib.util.spec_from_file_location("_gdn_bench", HERE / "bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
MANIFEST = HERE / "last_run_manifest.jsonl"

PCC_GATE = 0.9995  # the focus case's soft pcc_threshold
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
# configs: (id, group_size, num_groups, box_w, block_rows, num_rows, rd_tiles)
# ---------------------------------------------------------------------------

# THE focus geometry: (1,1,8192,1024) BLOCK_SHARDED shard [1024,128] grid (8,8) ->
# 8 groups of 8, 32 tile-rows per core, BLOCK_ROWS = 8 -> 4 combine rounds, and a NATIVE
# zero-copy input shard, so rd_tiles = 0 (NoC0 idle).
FOCUS = ("focus_g8_br8_r32", 8, 8, None, 8, 32, 0)

# BLOCK_ROWS axis (how many face-writes a round has to split).
SWEEP_BR = [
    ("g8_br1_r1", 8, 8, None, 1, 1, 0),  # decode: ONE row -> nothing to split BY ROW
    ("g4_br32_r32", 4, 8, None, 32, 32, 0),  # one round of 32 rows (G=4: the ring fits L1)
]

# GROUP_SIZE axis (the gather's fan-in multiplier), BLOCK_ROWS = 8.
SWEEP_G = [
    ("g4_br8_r32", 4, 8, None, 8, 32, 0),
    ("g28_br8_r32", 28, 1, 8, 8, 32, 0),  # the op's WIDTH-shard grid, NON-RECTANGULAR
    ("g32_br8_r32", 32, 1, 8, 8, 32, 0),
]

# The decode profile at every group size: BLOCK_ROWS = 1, one round, 2 face-writes total.
SWEEP_DECODE = [
    ("g4_br1_r1", 4, 8, None, 1, 1, 0),
    ("g28_br1_r1", 28, 1, 8, 1, 1, 0),
    ("g32_br1_r1", 32, 1, 8, 1, 1, 0),
]

# THE CARVE-OUT PROBE: NoC0 is NOT idle.  A reader-fed (interleaved / non-native) placement
# makes the reader stream x and gamma over NoC0 before it can ship anything, so the idea's
# premise is gone and it is expected to REGRESS.  rd_tiles bf16 tiles per round per core.
SWEEP_LOAD = [
    ("focus_rd8", 8, 8, None, 8, 32, 8),
    ("focus_rd32", 8, 8, None, 8, 32, 32),
    ("g32_br8_rd32", 32, 1, 8, 8, 32, 32),
]

ALL_CONFIGS = [FOCUS] + SWEEP_BR + SWEEP_G + SWEEP_DECODE + SWEEP_LOAD

# The focus points repeated so the ~2-3% noise-band calls are a median, not one sample.
REPEAT_MENU = ["base", "z", "sf", "sfm", "mc", "mcs", "sfm_mc", "sfm_mcs", "sf_mcs"]

LOAD_MENU = ["base", "sf", "sfm", "mc", "mcs", "sfm_mc"]


def _menu(labels):
    by = {v.label: v for v in B.FULL_MENU}
    return [by[x] for x in labels]


def _filtered(menu):
    """GDN_ONLY=base,mc restricts any run to those variants -- bring-up / hang bisection."""
    only = os.environ.get("GDN_ONLY")
    if not only:
        return menu
    keep = set(only.split(","))
    return [v for v in menu if v.label in keep]


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
    # The synthetic NoC0 load source: plain DRAM interleaved bf16, never read for its value.
    load_t = ttnn.from_torch(
        torch.zeros(1, 1, B.LOAD_TILES_TOTAL * B.TILE, B.TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return x, out, load_t, vals


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


def _run_one(device, cfg, variant, records, baseline_out, ablate=0, fold_style=1):
    import torch

    cid, gs, ng, box_w, block_rows, num_rows, rd_tiles = cfg
    geo = B.build_geometry(device, group_size=gs, num_groups=ng, box_w=box_w)
    x, out, load_t, vals = _make_tensors(device, geo, num_rows, seed=1234)
    W = gs * 128
    inv_w = 1.0 / W
    eps = 1e-6
    try:
        prog = B.build_program(
            device,
            x,
            out,
            load_t,
            geo,
            variant=variant,
            block_rows=block_rows,
            num_rows=num_rows,
            rd_tiles=rd_tiles,
            inv_w=inv_w,
            eps=eps,
            compute_config=_compute_config(),
            ablate=ablate,
            fold_style=fold_style,
        )
        res = ttnn.generic_op([x, load_t, out], prog)
    except RuntimeError as e:
        if "L1" not in str(e) and "allocate" not in str(e).lower():
            raise
        print(
            f"[gdn] {cid:20s} {variant.label:10s} abl={ablate} L1_OOM ({B.l1_bytes(gs, block_rows, variant)//1024} kB)"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        ttnn.deallocate(load_t)
        return
    got = ttnn.to_torch(res).reshape(len(geo.cores), num_rows * B.TILE, B.TILE)
    ref, active = _reference(geo, vals, inv_w, eps)

    # Only the lanes the datapath actually carries: GATHER_FACES == 2 ships faces 0 and 2
    # (columns 0..15) and D17's finalize scopes to the EVEN lanes of those, so the columns
    # the op's consumer (mul<BroadcastDim::Col>, column 0) reads are 0,2,..,14 -- the same
    # lane set for every variant, so the gate compares like with like.
    act = sorted(active)
    g = got[act][:, :, 0:16:2].clone()
    r = ref[act].unsqueeze(-1).expand(-1, -1, 8)
    pcc = _pcc(g, r)
    relrms = _rel_rms(g, r)
    if ablate:
        # The fold PAYLOAD is stubbed, so the output is garbage by construction.  This mode
        # exists only to expose the transport; it is never a correctness claim.  Every
        # variant is gated in the un-ablated mode instead.
        pcc, relrms, bitexact = float("nan"), float("nan"), None
    elif variant.label == "base":
        baseline_out[(cid, fold_style)] = g
        bitexact = True
    elif (cid, fold_style) in baseline_out:
        bitexact = bool(torch.equal(g, baseline_out[(cid, fold_style)]))
    else:
        bitexact = None
    rec = dict(
        config=cid,
        variant=variant.label,
        group_size=gs,
        num_groups=ng,
        block_rows=block_rows,
        num_rows=num_rows,
        rd_tiles=rd_tiles,
        ablate=ablate,
        fold_style=fold_style,
        zero_r=variant.zero_r,
        split_mode=variant.split_mode,
        num=variant.num,
        den=variant.den,
        mcast_r=variant.mcast_r,
        split_root=variant.split_root,
        tok_round=variant.tok_round,
        pcc=pcc,
        rel_rms=relrms,
        bitexact=bitexact,
        l1_bytes=B.l1_bytes(gs, block_rows, variant),
    )
    records.append(rec)
    print(
        f"[gdn] {cid:20s} {variant.label:10s} abl={ablate} fold={fold_style} pcc={pcc:.6f} rel_rms={relrms:.5f} "
        f"bitexact={bitexact} l1={rec['l1_bytes']//1024}kB"
    )
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    ttnn.deallocate(load_t)
    if ablate:
        return  # perf-only mode: nothing to gate (see above)
    assert pcc >= PCC_GATE, f"{cid}/{variant.label}: pcc {pcc} < {PCC_GATE}"
    assert relrms <= RELRMS_GATE, f"{cid}/{variant.label}: rel-RMS {relrms} > {RELRMS_GATE}"
    assert bitexact is not False, (
        f"{cid}/{variant.label}: output is NOT bit-identical to `base`.  This idea only "
        f"changes which RISC issues a write, so this is a RACE (torn partial / lost or "
        f"early semaphore inc), not a precision effect."
    )


WARMUP_CFG = ("warmup", 4, 1, 4, 1, 1, 0)


def _drive(device, configs, menu, tag, *, modes=(0,), fold_styles=(1,), append=False):
    records = []
    baseline_out = {}
    # One throwaway program FIRST so no measured variant is this process's first device
    # launch (dispatch / L1 first-touch would otherwise land on it).  Its row is in the
    # manifest and is ignored by the reader.
    warm = []
    _run_one(device, WARMUP_CFG, B.BASELINE, warm, {})
    menu = _filtered(menu)
    try:
        for cfg in configs:
            for fold_style in fold_styles:
                for ablate in modes:
                    for variant in menu:
                        _run_one(device, cfg, variant, records, baseline_out, ablate=ablate, fold_style=fold_style)
    finally:
        mode = "a" if (append or os.environ.get("GDN_APPEND") == "1") else "w"
        with open(MANIFEST, mode) as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[gdn] manifest ({tag}, {len(records)} runs) -> {MANIFEST}")


def test_focus_menu(device):
    """Correctness + the un-ablated number: the WHOLE variant menu on the focus geometry."""
    _drive(device, [FOCUS], B.FULL_MENU, "focus_menu")


def test_focus_folds(device):
    """The DECISION run: the load-bearing options x {FULL, fold-ablated} x
    {the op's CURRENT fused D22 root chain, the pre-Perf-2 streaming D16 chain}.  A transport
    idea only cashes out if the transport is not already hidden behind the root's fold, and
    which fold the baseline runs decides that -- so both are measured, not argued."""
    _drive(
        device,
        [FOCUS],
        _menu(["base", "z", "sf", "sfm", "mc", "mcs", "sfm_mc", "sfm_mcs"]),
        "focus_folds",
        modes=(0, 1),
        fold_styles=(1, 0),
    )


def test_focus_ablated(device):
    """THE HEADLINE.  Same menu, fold PAYLOAD stubbed (all CB handshakes and trip counts
    kept), so the number the variants move is the combine's TRANSPORT + SYNC rather than the
    root's serial fold.  Never a correctness claim -- see _run_one."""
    _drive(device, [FOCUS], B.FULL_MENU, "focus_ablated", modes=(1,))


def test_focus_repeat(device):
    """3 fresh ABLATED programs per point, so the noise-band calls are a median."""
    records = []
    baseline_out = {}
    warm = []
    _run_one(device, WARMUP_CFG, B.BASELINE, warm, {})
    menu = _filtered(_menu(REPEAT_MENU))
    try:
        for _ in range(3):
            for variant in menu:
                _run_one(
                    device,
                    FOCUS,
                    variant,
                    records,
                    baseline_out,
                    ablate=int(os.environ.get("GDN_ABLATE", "1")),
                )
    finally:
        with open(MANIFEST, "w") as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[gdn] manifest (focus_repeat, {len(records)} runs) -> {MANIFEST}")


def test_sweep_br(device):
    """Correctness (un-ablated) AND the transport number (ablated), per regime."""
    _drive(device, SWEEP_BR, B.SWEEP_MENU, "sweep_br", modes=(0, 1))


def test_sweep_g(device):
    """Correctness (un-ablated) AND the transport number (ablated), per regime."""
    _drive(device, SWEEP_G, B.SWEEP_MENU, "sweep_g", modes=(0, 1))


def test_sweep_decode(device):
    """Correctness (un-ablated) AND the transport number (ablated), per regime."""
    _drive(device, SWEEP_DECODE, B.SWEEP_MENU, "sweep_decode", modes=(0, 1))


def test_load(device):
    """The carve-out probe: NoC0 busy with a reader-fed placement's own traffic."""
    _drive(device, SWEEP_LOAD, _menu(LOAD_MENU), "load", modes=(0, 1))


@pytest.mark.parametrize("cid", [c[0] for c in ALL_CONFIGS])
def test_single(device, cid):
    """One config, the whole menu -- for bring-up / --dev triage."""
    cfg = next(c for c in ALL_CONFIGS if c[0] == cid)
    _drive(device, [cfg], B.FULL_MENU, cid)
