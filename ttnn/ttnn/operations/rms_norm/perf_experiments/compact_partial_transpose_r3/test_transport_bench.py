# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""BENCH B driver: the WHOLE cross-core combine, FLAT vs COMPACT.

Correctness is the ONLY pass/fail; every variant's device kernel duration is read from the
profiler CSV afterwards by read_transport.py (this file only emits the run manifest, in execution
order).

    correctness only:  scripts/run_safe_pytest.sh --run-all <this file> -k sweep
    measured:          scripts/run_safe_pytest.sh --profile <this file> -k sweep
                       python3 <this dir>/read_transport.py <ops_perf_results_*.csv>
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent

# Load BY PATH: ttnn/ttnn/operations/__init__.py walk_packages()es and EXECUTES every reachable
# module at `import ttnn` (perf_experiments/README.md).
_spec = importlib.util.spec_from_file_location("_cpt3_transport", HERE / "transport_bench.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

MANIFEST = HERE / "transport_manifest.jsonl"
PCC_GATE = 0.9995
RELRMS_GATE = 0.04
W_LOGICAL = 1024
EPS = 1e-5


def _compute_config():
    """The op's PINNED precision contract -- FIXED, identical for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _pcc(a, b):
    import torch

    a = a.flatten().double()
    b = b.flatten().double()
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0.0 else float((a @ b).item() / denom)


def _eff_pcc(got, ref):
    """PCC the way the OP is gated: the stat is ~1.0 everywhere by construction, so a raw PCC on
    it is variance-starved.  Multiply both by the same fixed N(0,1) draw (i.e. score `x * 1/rms`,
    which is what the op's soft gate actually sees)."""
    import torch

    x = torch.randn(got.numel(), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return _pcc(x * got.double().flatten(), x * ref.double().flatten())


def _rel_rms(got, ref):
    return float((got.double() - ref.double()).pow(2).mean().sqrt() / ref.double().pow(2).mean().sqrt())


# (id, group_size, num_groups, box_w, block_rows, num_rows)
CONFIGS = [
    # THE FOCUS SHAPE: (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8 -> 8 groups of 8,
    # 32 tile-rows per core, BLOCK_ROWS = 8 -> 4 combine rounds.
    ("focus_g8_br8_r32", 8, 8, None, 8, 32),
    # THE num_blocks LADDER at the focus geometry: does one big round beat four small ones once
    # the gather ring stops scaling with BLOCK_ROWS?  (The op's D25 pipeline is VOID at 1 round.)
    ("g8_br2_r32", 8, 8, None, 2, 32),
    ("g8_br4_r32", 8, 8, None, 4, 32),
    ("g8_br16_r32", 8, 8, None, 16, 32),
    ("g8_br32_r32", 8, 8, None, 32, 32),
    # BLOCK_ROWS = 1: the decode / width-shard profile.  The mechanism is a NO-OP here, so this
    # is the FLAT-result cell, and flat is IN-DOMAIN, not an exception.
    ("g8_br1_r1", 8, 8, None, 1, 1),
    # GROUP_SIZE sweep.  9 is ODD -> GATHER_SLOTS = 10 with a boot-zeroed pad slot.
    ("g4_br8_r32", 4, 8, None, 8, 32),
    ("g4_br1_r1", 4, 8, None, 1, 1),
    ("g9_br8_r8", 9, 1, 8, 8, 8),
    ("g9_br1_r1", 9, 1, 8, 1, 1),
    ("g28_br8_r8", 28, 1, 8, 8, 8),
    ("g28_br1_r1", 28, 1, 8, 1, 1),
    ("g32_br8_r8", 32, 1, 8, 8, 8),
    ("g32_br1_r1", 32, 1, 8, 1, 1),
    # RAGGED: num_rows NOT a multiple of BLOCK_ROWS, so the LAST round is short.  A compact tile
    # then uses only columns 0..rows-1 and the rest stay exactly 0, and the widened finalize scope
    # is a superset -- so this should just work, and "should just work" is not a measurement.
    ("g8_br8_r20_ragged", 8, 8, None, 8, 20),
    ("g8_br3_r8_ragged", 8, 8, None, 3, 8),
]

VARIANTS = (("flat", B.V_FLAT), ("compact", B.V_COMPACT))
WARMUP = ("warmup", 4, 1, 4, 1, 1)


def _make_tensors(device, geo, num_rows, block_rows, seed):
    import torch

    ncores = len(geo.cores)
    g = torch.Generator().manual_seed(seed)
    mean = W_LOGICAL / geo.group_size
    # A member's partial: sum(x^2) over its W/GROUP_SIZE-wide slice of a W = 1024 row of N(0,1)
    # activations, so the group total averages W and the rsqrt argument sits at ~1.0 -- where the
    # real op runs it.  32 DISTINCT values per tile-row (a real REDUCE_ROW column vector).
    vals = (mean + (mean / 8.0) * torch.randn(ncores, num_rows, B.TILE, generator=g)).clamp_min(mean / 4)

    x_t = torch.zeros(ncores, num_rows * B.TILE, B.TILE, dtype=torch.float32)
    for r in range(num_rows):
        x_t[:, r * B.TILE : (r + 1) * B.TILE, 0] = vals[:, r, :]
        # Columns 1..15 hold finite GARBAGE, as the op's reduce leaves them.  Both variants must
        # ignore it: FLAT's add_tiles never mixes columns, and COMPACT's pack matmul multiplies
        # them by an exact 0 -- which is only safe because they are FINITE (inf*0 = NaN).
        x_t[:, r * B.TILE : (r + 1) * B.TILE, 1:16] = mean * torch.rand(ncores, B.TILE, 15, generator=g)

    # The one-hot bank, IDENTICAL on every core: E_r[0][r] = 1.  srcB `transpose` reads it as
    # E_r^T == F_r, so ONE bank of block_rows pages serves both permutation directions.
    bank_one = torch.zeros(block_rows * B.TILE, B.TILE, dtype=torch.float32)
    for r in range(block_rows):
        bank_one[r * B.TILE + 0, r] = 1.0
    bank_t = bank_one.unsqueeze(0).expand(ncores, -1, -1).contiguous()

    def dev(t, tiles_per_core, fill=None):
        src = t if fill is None else torch.full_like(t, fill)
        mc = ttnn.create_sharded_memory_config(
            shape=(tiles_per_core * B.TILE, B.TILE),
            core_grid=geo.core_range_set,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(
            src.reshape(1, 1, -1, B.TILE),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )

    return dev(x_t, num_rows), dev(bank_t, block_rows), dev(x_t, num_rows, fill=-7.0), vals


def _reference(geo, vals):
    """ref[core_index, row, i] = the finalized stat of the group that core belongs to."""
    import torch

    index_of = {(c.x, c.y): i for i, c in enumerate(geo.cores)}
    ref = torch.zeros_like(vals, dtype=torch.float64)
    active = set()
    for group in geo.groups:
        idxs = [index_of[(c.x, c.y)] for c in group]
        total = vals[idxs].double().sum(0)  # [rows, 32]
        stat = torch.rsqrt(total * (1.0 / W_LOGICAL) + EPS)
        for i in idxs:
            ref[i] = stat
            active.add(i)
    return ref, active


def _run_one(device, cfg, variant, records, *, fin=None, dest_batch=4, label=None):
    cid, gs, ng, box_w, block_rows, num_rows = cfg
    vname, vcode = variant
    label = label or vname
    geo = B.build_geometry(device, group_size=gs, num_groups=ng, box_w=box_w)
    x, bank, out, vals = _make_tensors(device, geo, num_rows, block_rows, seed=1234)
    n_wr, n_by = B.gather_transfers(gs, block_rows, vcode)
    rec = dict(
        config=cid,
        variant=label,
        group_size=gs,
        num_groups=ng,
        block_rows=block_rows,
        num_rows=num_rows,
        num_blocks=-(-num_rows // block_rows),
        l1_bytes=B.l1_bytes(gs, block_rows, vcode),
        gather_writes=n_wr,
        gather_bytes=n_by,
    )
    try:
        prog = B.build_program(
            device,
            x,
            bank,
            out,
            geo,
            variant=vcode,
            block_rows=block_rows,
            num_rows=num_rows,
            inv_w=1.0 / W_LOGICAL,
            eps=EPS,
            fin=fin,
            dest_batch=dest_batch,
            compute_config=_compute_config(),
        )
        res = ttnn.generic_op([x, bank, out], prog)
    except RuntimeError as e:
        if "L1" not in str(e) and "allocate" not in str(e).lower():
            raise
        # A variant may simply not FIT: FLAT's landing ring is GATHER_SLOTS * BLOCK_ROWS fp32
        # tiles, which at BLOCK_ROWS = 32 exceeds a core's L1 on its own.  That is a load-bearing
        # RESULT (it is exactly the L1 term the compact layout deletes), not a bench bug.
        print(f"[benchB] {cid:20s} {label:10s} L1_OOM ({rec['l1_bytes']//1024} kB of combine CBs)")
        rec["l1_oom"] = True
        records.append(rec)
        ttnn.deallocate(x)
        ttnn.deallocate(bank)
        ttnn.deallocate(out)
        return
    got = ttnn.to_torch(res).reshape(len(geo.cores), num_rows * B.TILE, B.TILE)
    ref, active = _reference(geo, vals)
    act = sorted(active)
    # COLUMN 0 only -- the single column the op's consumer (pass B's mul<BroadcastDim::Col>)
    # reads, and the one column both variants are contractually required to define.  Identical
    # lane set for both, so the gate compares like with like.
    g = got[act][:, :, 0].reshape(len(act), num_rows, B.TILE)
    r = ref[act]
    rec["pcc"] = _eff_pcc(g, r)
    rec["rel_rms"] = _rel_rms(g, r)
    records.append(rec)
    print(
        f"[benchB] {cid:20s} {label:10s} pcc={rec['pcc']:.7f} rel_rms={rec['rel_rms']:.6f} "
        f"l1={rec['l1_bytes']//1024}kB gather={n_wr}w/{n_by}B"
    )
    ttnn.deallocate(x)
    ttnn.deallocate(bank)
    ttnn.deallocate(out)
    assert rec["pcc"] >= PCC_GATE, f"{cid}/{label}: pcc {rec['pcc']} < {PCC_GATE}"
    assert rec["rel_rms"] <= RELRMS_GATE, f"{cid}/{label}: rel-RMS {rec['rel_rms']} > {RELRMS_GATE}"


def _drive(device, configs, tag, extra=()):
    records = []
    warm = []
    # One throwaway program FIRST, so no measured variant is this process's first device launch.
    _run_one(device, WARMUP, VARIANTS[0], warm)
    try:
        for cfg in configs:
            for variant in VARIANTS:
                _run_one(device, cfg, variant, records)
        for cfg, variant, kw, label in extra:
            _run_one(device, cfg, variant, records, label=label, **kw)
    finally:
        with open(MANIFEST, "w") as f:
            for r in warm + records:
                f.write(json.dumps(r) + "\n")
        print(f"[benchB] manifest ({tag}, {len(records)} runs) -> {MANIFEST}")


# OPTION probes at the focus geometry, on top of the head-to-head: the finalize-scope widening
# (which the compact layout forces) and the un-pack's DEST batching.
FOCUS = CONFIGS[0]
BR32 = CONFIGS[4]
EXTRA = [
    (FOCUS, ("compact", B.V_COMPACT), dict(fin=B.FIN_RC), "compact_finRC"),
    (FOCUS, ("compact", B.V_COMPACT), dict(dest_batch=8), "compact_batch8"),
    (FOCUS, ("compact", B.V_COMPACT), dict(dest_batch=2), "compact_batch2"),
    (BR32, ("compact", B.V_COMPACT), dict(dest_batch=8), "compact_batch8"),
]


def test_sweep(device):
    _drive(device, CONFIGS, "sweep", extra=EXTRA)


@pytest.mark.parametrize("cid", [c[0] for c in CONFIGS])
def test_single(device, cid):
    """One config, both variants -- for bring-up under --dev."""
    cfg = next(c for c in CONFIGS if c[0] == cid)
    _drive(device, [cfg], cid)
