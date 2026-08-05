# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off runner for `root_rotation`.

    scripts/run_safe_pytest.sh --profile --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/root_rotation/test_root_rotation.py \
        -k focus
    python3 .../root_rotation/read_results.py

Correctness is the only pass/fail; perf is recorded, never asserted.  The gate is
BIT-EXACTNESS against the `fixed` variant at the same BLOCK_ROWS -- rotation moves
WHICH CORE folds a round, never the fold's order/DEST walk/finalize, so anything
other than an identical output is a bug -- plus pcc AND rel-RMS against a torch
fp32 reference (pcc alone hides a uniform scale error; the op has been bitten by
exactly that on this code path).

Every variant of a case runs in ONE profiled process, one program per variant, and
the test writes `manifests/<case>.jsonl` (one line per launch, in launch order) so
read_results.py can join to the profiler CSV's `DEVICE KERNEL DURATION [ns]` rows.

Env knobs:
    RMS_ROT_VARIANTS=0,1,3   run exactly these variants (overrides the per-case set)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd

from . import bench


# `torch` is imported LAZILY (see bench.py / perf_experiments/README.md).
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


HERE = Path(__file__).parent
# ONE manifest per case, TRUNCATED by the test that owns it, so a re-run (and the
# --profile wrapper's re-invocation of pytest) can never leave stale launch records
# that would slide the profiler-CSV join.
MANIFEST_DIR = HERE / "manifests"

_ML = ttnn.TensorMemoryLayout

# case -> (shape, [shard_shape, grid] | None, memory_layout, block_rows_cap[, grid_w])
#
# `block_rows_cap` is the ONLY synthetic knob: it lowers the descriptor's own
# L1_SAFETY_FRACTION until its own L1 solve picks BLOCK_ROWS <= cap, which is how the
# num_blocks axis is swept without changing the shard geometry.  None = the op's real
# solve.  `grid_w` sets the descriptor's GRID_W override (cores per width group) for
# the interleaved plan, whose group size is otherwise auto-chosen.
CASES = {
    # ---- a SMALL first launch: 16 cores, GS 4, several rounds.  Run this alone
    #      before anything else -- rotation's failure mode is a semaphore/mcast
    #      deadlock, and a hang on 16 cores triages faster than one on 64.
    "smoke": ((1, 1, 512, 256), ([128, 64], (4, 4)), _ML.BLOCK_SHARDED, 1),
    # ---- the FOCUS shape: 64 cores, 8 groups of 8, 4 combine rounds -----------
    "focus": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    # ---- num_blocks axis at GROUP_SIZE 8 -------------------------------------
    "nb1": ((1, 1, 2560, 1024), ([320, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    "nb2": ((1, 1, 5120, 1024), ([640, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    "nb8": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 4),
    # num_blocks (16) > GROUP_SIZE (8): the rotation WRAPS, so every core roots twice.
    "nb16": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 2),
    # ---- GROUP_SIZE axis -----------------------------------------------------
    # GS 4 with num_blocks 4 (one full cycle) and with num_blocks 8 (two cycles).
    "gs4": ((1, 1, 4096, 512), ([512, 128], (4, 8)), _ML.BLOCK_SHARDED, 4),
    "gs4_nb8": ((1, 1, 4096, 512), ([512, 128], (4, 8)), _ML.BLOCK_SHARDED, 2),
    # The op's own pinned decode geometries: ONE round, so there is nothing to
    # rotate.  Expected FLAT -- which is IN the domain, not an exception.
    "gs32_nb1": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, None),
    "gs28_nb1": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, None),
    "gs9_nb1": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, None),
    # Multi-round at a WIDE group, on the PACKED single-group topology (Mcast2D over
    # a bounding box).  num_blocks < GROUP_SIZE here, so only some cores ever root.
    "gs32_multi": ((1, 1, 2048, 1024), ([2048, 32], (8, 4)), _ML.WIDTH_SHARDED, None),
    "gs16_multi": ((1, 1, 4096, 1024), ([4096, 64], (8, 2)), _ML.WIDTH_SHARDED, None),
    # ---- the NON-NATIVE input path: an INTERLEAVED width split (cb_input_tiles is
    #      reader-fed, not backed on a resident shard).
    "ilv_gw8": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, None, 8),
}
GRID_W = {k: v[4] for k, v in CASES.items() if len(v) > 4}

B = bench
# The FULL menu (every lever alone and composed, plus the zeroing ablation pair)
# only where the imbalance is largest; elsewhere the baseline + the two rotation
# placements, because every extra program costs a JIT build.
FULL_CASES = {"focus"}
FULL_SET = (
    B.ROTATE,  # the candidate
    B.ROTATE | B.ZDEFER,  # + deferred gather zeroing        (measured WORSE)
    B.ROTATE | B.DIAG,  # + per-grid-row rotation phase    (measured FLAT)
    B.ROTATE | B.STILL,  # ATTRIBUTION: machinery, no move
    B.MIDROOT,  # ATTRIBUTION: full-line EXCLUDE rect, no move
    B.NOZERO,  # ABLATION pair: the gather zeroing's own cost ...
    B.ROTATE | B.NOZERO,  # ... on the fixed and on the rotating placement
)
DEFAULT_SET = (B.ROTATE,)

# ABLATION variants: deliberately INCORRECT wherever the gather ships fewer than 4
# faces (undefined L1 faces reach the fold).  Recorded, never asserted, and never a
# graduatable option -- they exist only to make the zeroing's cost separable from
# rotation's.
ABLATIONS = B.NOZERO


def _mk_tensors(device, shape, shard, layout):
    from eval.sharding import shard_config

    torch.manual_seed(42)
    W = shape[-1]
    mc = (
        ttnn.DRAM_MEMORY_CONFIG
        if shard is None
        else shard_config(shard[0], shard[1], layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
    x_t = torch.randn(shape, dtype=torch.bfloat16)
    g_t = torch.randn(W, dtype=torch.bfloat16)
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    g = ttnn.from_torch(g_t.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x_t, g_t, x, g, mc


def _resolve_safety_fraction(x, out, g, cfg, cap):
    """Lowest-perturbation L1_SAFETY_FRACTION whose own solve gives BLOCK_ROWS <= cap."""
    if cap is None:
        return rpd.L1_SAFETY_FRACTION
    orig = rpd.L1_SAFETY_FRACTION
    try:
        f = orig
        while f > 0.02:
            rpd.L1_SAFETY_FRACTION = f
            pd = rpd.create_program_descriptor(x, out, gamma=g, epsilon=1e-6, compute_kernel_config=cfg)
            if pd.kernels[2].compile_time_args[3] <= cap:
                return f
            f = round(f - 0.01, 4)
        raise AssertionError(f"no L1_SAFETY_FRACTION gives BLOCK_ROWS <= {cap}")
    finally:
        rpd.L1_SAFETY_FRACTION = orig


@pytest.mark.parametrize("case", list(CASES))
def test_root_rotation(device, case):
    shape, shard, layout, br_cap = CASES[case][:4]
    cfg = bench._perf_config()
    # The descriptor's own GRID_W override.  Restored in the finally so one case can
    # never leak its plan into the next.
    grid_w_orig = rpd.GRID_W
    rpd.GRID_W = GRID_W.get(case, rpd.GRID_W)
    try:
        _body(device, case, shape, shard, layout, br_cap, 1e-6, cfg)
    finally:
        rpd.GRID_W = grid_w_orig


def _body(device, case, shape, shard, layout, br_cap, eps, cfg):
    x_t, g_t, x, g, mc = _mk_tensors(device, shape, shard, layout)
    ref = bench.torch_reference(x_t, g_t, eps)

    # ONE output tensor for every variant: a second resident shard per variant would
    # eat the L1 the CBs need (the shards and the CB arena share L1).
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    frac = _resolve_safety_fraction(x, out, g, cfg, br_cap)
    results = []
    outs = []

    def one(v):
        orig = rpd.L1_SAFETY_FRACTION
        rpd.L1_SAFETY_FRACTION = frac
        try:
            y, info = bench.run(device, x, out, g, variant=v, epsilon=eps, compute_config=cfg)
        finally:
            rpd.L1_SAFETY_FRACTION = orig
        got = ttnn.to_torch(y)
        rec = dict(
            case=case,
            # The shape/layout ride the manifest so read_results.py can VERIFY the
            # positional join against the CSV's own INPUT_0_W column.  generated/ is
            # shared with the sibling benches running in this same clone, so "the newest
            # ops_perf_results CSV" is NOT reliably this run's -- one such collision was
            # caught here by the row-count assert.
            w=int(shape[-1]),
            h=int(shape[-2]),
            variant=v,
            name=bench.VARIANTS[v],
            pcc=bench.pcc(got, ref),
            rel_rms=bench.rel_rms(got, ref),
            bit_exact=None,  # filled in the post-pass
            ablation=bool(v & ABLATIONS),
            safety_fraction=frac,
            **info,
        )
        results.append(rec)
        outs.append((rec, got.clone()))
        print(
            f"  [{case}] v{v:>2d} {rec['name']:<20s} pcc={rec['pcc']:.6f} relrms={rec['rel_rms']:.2e} "
            f"BR={info['block_rows']} nblk={info['num_blocks']} GS={info['group_size']} "
            f"rootcores={info['root_cores']} diag={info['diag_applied']} +L1={info['extra_l1_bytes']}B"
        )
        return rec

    want = os.environ.get("RMS_ROT_VARIANTS")
    if want:
        variants = [int(t) for t in want.split(",")]
    else:
        # -1 (`pure_op`) FIRST: the op's own descriptor, untouched, so every table
        # carries its own calibration of `fixed` against the real op.
        variants = [bench.PUREOP, 0] + list(FULL_SET if case in FULL_CASES else DEFAULT_SET)
    for v in variants:
        one(v)

    # ---- BIT-EXACTNESS post-pass -------------------------------------------
    base_t = next((t for r, t in outs if r["variant"] == 0), None)
    for rec, t in outs:
        if base_t is not None:
            rec["bit_exact"] = bool(torch.equal(t, base_t))

    MANIFEST_DIR.mkdir(exist_ok=True)
    with (MANIFEST_DIR / f"{case}.jsonl").open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    # Correctness gate -- the ONLY pass/fail.  Ablation variants are RECORDED (they are
    # deliberately incorrect wherever GATHER_FACES < 4); every real variant must be
    # bit-identical to `fixed` and must pass both pcc and rel-RMS.
    for rec in results:
        if rec["ablation"]:
            print(
                f"  [{case}] ABLATION (not a candidate) {rec['name']}: pcc {rec['pcc']:.6f} "
                f"relrms {rec['rel_rms']:.2e} bit_exact={rec['bit_exact']}"
            )
            continue
        assert rec["pcc"] > 0.9995, f"{case}/{rec['name']}: pcc {rec['pcc']}"
        assert rec["rel_rms"] < 0.04, f"{case}/{rec['name']}: rel-RMS {rec['rel_rms']}"
        assert rec["bit_exact"] is not False, (
            f"{case}/{rec['name']}: output differs from the `fixed` variant -- rotation moved "
            f"WHICH CORE folds, so it must not change the ARITHMETIC"
        )
