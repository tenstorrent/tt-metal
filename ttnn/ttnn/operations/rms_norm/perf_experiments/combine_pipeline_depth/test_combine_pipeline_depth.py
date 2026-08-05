# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off runner for `combine_pipeline_depth`.

    scripts/run_safe_pytest.sh --profile --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/combine_pipeline_depth/test_combine_pipeline_depth.py \
        -k focus
    python3 .../combine_pipeline_depth/read_results.py

Correctness is the only pass/fail; perf is recorded, never asserted.  The gate is
BIT-EXACTNESS against the serial variant at the same BLOCK_ROWS -- the pipeline
reorders WHEN work is issued, never WHAT is computed, so anything other than an
identical output is a bug -- plus pcc against a torch fp32 reference.

Every variant of a case runs in ONE profiled process, one program per variant, and
the test writes `manifests/<case>.jsonl` (one line per launch, in launch order) so
read_results.py can join to the profiler CSV's `DEVICE KERNEL DURATION [ns]` rows.
A variant that does not FIT L1 is recorded as such and skipped: that is the
BLOCK_ROWS consequence of a deeper combine buffer, which is a measurement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as rpd

from . import bench


# `torch` is imported LAZILY here, on first attribute access, rather than at module scope.
# `ttnn/ttnn/operations/__init__.py` runs pkgutil.walk_packages over the operations tree, and
# the repo's `check-torch-imports-in-ttnn` pre-commit hook forbids a global torch import
# anywhere under ttnn/ for exactly that reason.  See perf_experiments/README.md -- an
# __init__.py in that directory once broke `import ttnn` repo-wide twice in one round.
# Every `torch.<attr>` use below is unchanged; the proxy just defers the import.
class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()


HERE = Path(__file__).parent
# ONE manifest per case, TRUNCATED by the test that owns it, so a re-run (and the
# --profile wrapper's re-invocation of pytest) can never leave stale launch
# records that would slide the profiler-CSV join.
MANIFEST_DIR = HERE / "manifests"

_ML = ttnn.TensorMemoryLayout

# (shape, [shard_shape, grid], memory_layout, block_rows_cap)
#
# `block_rows_cap` is the ONLY synthetic knob: it lowers the descriptor's own
# L1_SAFETY_FRACTION until its own L1 solve picks BLOCK_ROWS <= cap, which is how
# the num_blocks axis is swept without changing the shard geometry.  None = the
# op's real solve.
CASES = {
    # ---- the FOCUS shape: 64 cores, 8 groups of 8, 4 combine rounds ----------
    "focus": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    # ---- num_blocks axis at GROUP_SIZE 8 ------------------------------------
    "nb1": ((1, 1, 2560, 1024), ([320, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    "nb2": ((1, 1, 5120, 1024), ([640, 128], (8, 8)), _ML.BLOCK_SHARDED, None),
    "nb8": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 4),
    "nb16": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, 2),
    # ---- GROUP_SIZE axis ----------------------------------------------------
    "gs32_nb1": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, None),
    "gs28_nb1": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, None),
    "gs32_multi": ((1, 1, 2048, 1024), ([2048, 32], (8, 4)), _ML.WIDTH_SHARDED, None),
    "gs16_multi": ((1, 1, 4096, 1024), ([4096, 64], (8, 2)), _ML.WIDTH_SHARDED, None),
    # ---- the NON-NATIVE input path: an INTERLEAVED width split, where
    #      cb_input_tiles is reader-fed at CB_X_DEPTH == 2 rather than backed on a
    #      resident shard.  This is the expressibility question for the pipeline:
    #      pass A one block ahead reads TWO blocks of cb_input_tiles.
    "ilv_gw8": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, None, 8),
}
# Cases whose plan needs the descriptor's GRID_W override (cores per width group).
GRID_W = {k: v[4] for k, v in CASES.items() if len(v) > 4}

# The RING2 lever (and its BLOCK_ROWS search) is only explored where the fan-in is
# largest and the L1 headroom smallest; elsewhere it is dominated by the cheaper
# levers, and every extra program costs a JIT build.
FULL_CASES = {"focus", "gs32_multi"}

# MEASURED DOMAIN EXCEPTION, recorded rather than asserted.
#
# `pipe` reads a TWO-ROW-BLOCK window of cb_input_tiles at a tile OFFSET from the
# CB front, and a tile offset cannot cross the ring wrap.  Where cb_input_tiles is
# backed on the resident input shard (every SHARDED combine plan) the ring is the
# whole per-core assignment and the front never wraps -- bit-exact everywhere.  On
# the INTERLEAVED width split it is a reader-fed CB_X_DEPTH == 2 ring, the window
# straddles the wrap once per N rounds, and the result is WRONG (pcc 0.9802).  The
# `xdeep` bit is the fix and its price; both are run here so the exception is
# measured with its remedy rather than asserted away.
EXPECT_INCORRECT = {"ilv_gw8": bench.PIPE_A}  # case -> bits that need `xdeep`
XDEEP_CASES = {"ilv_gw8"}


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
def test_combine_pipeline_depth(device, case):
    shape, shard, layout, br_cap = CASES[case][:4]

    eps = 1e-6
    cfg = bench._perf_config()
    # The descriptor's own GRID_W override (cores per width group).  Restored in the
    # finally below so one case can never leak its plan into the next.
    grid_w_orig = rpd.GRID_W
    rpd.GRID_W = GRID_W.get(case, rpd.GRID_W)
    try:
        _body(device, case, shape, shard, layout, br_cap, eps, cfg)
    finally:
        rpd.GRID_W = grid_w_orig


def _body(device, case, shape, shard, layout, br_cap, eps, cfg):
    x_t, g_t, x, g, mc = _mk_tensors(device, shape, shard, layout)
    ref = bench.torch_reference(x_t, g_t, eps)

    # ONE output tensor for every variant: a second resident shard per variant
    # would eat the L1 the CBs need (the shards and the CB arena share L1).
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    base_frac = _resolve_safety_fraction(x, out, g, cfg, br_cap)
    results = []
    serial_out = []  # [(block_rows, torch tensor)] for the bit-exactness gate

    def one(v, frac, note=""):
        """Run one variant at one L1 budget.  Returns None if it does not FIT L1 --
        which is a MEASUREMENT (the deeper combine ring's BLOCK_ROWS consequence),
        not an error."""
        orig = rpd.L1_SAFETY_FRACTION
        rpd.L1_SAFETY_FRACTION = frac
        try:
            y, info = bench.run(x, out, g, variant=v, epsilon=eps, compute_config=cfg)
        except RuntimeError as e:
            if "clash with L1" not in str(e) and "circular buffer" not in str(e):
                raise
            print(f"  [{case}] variant {v} {bench.VARIANTS[v]:<12s} DOES NOT FIT L1 at frac={frac}")
            return None
        finally:
            rpd.L1_SAFETY_FRACTION = orig
        got = ttnn.to_torch(y)
        p = bench.pcc(got, ref)
        rec = dict(
            bit_exact=None,  # filled in the post-pass below
            case=case,
            variant=v,
            name=bench.VARIANTS[v] + note,
            pcc=p,
            block_rows=info["block_rows"],
            group_size=info["group_size"],
            num_blocks=info["num_blocks"],
            rows_per_core=info["rows_per_core"],
            extra_l1_bytes=info["extra_l1_bytes"],
            safety_fraction=frac,
        )
        results.append(rec)
        serial_out.append((rec, got.clone()))
        print(
            f"  [{case}] variant {v:>2d} {bench.VARIANTS[v]+note:<24s} pcc={p:.6f} "
            f"BR={info['block_rows']} nblk={rec['num_blocks']} GS={info['group_size']} "
            f"+L1={info['extra_l1_bytes']}B"
        )
        return rec

    # --- everything the op's own L1 solve already affords ---------------------
    # Order matters only for the profiler-CSV join; the manifest records it.
    B = bench
    base = one(0, base_frac)
    op_br = base["block_rows"]
    # RMS_PIPE_VARIANTS=0,11 narrows the set -- used for the MECHANISM check, where
    # only two launches may share one zone CSV.
    only = os.environ.get("RMS_PIPE_VARIANTS")
    if only:
        for v in [int(t) for t in only.split(",") if int(t) != 0]:
            one(v, base_frac)
        MANIFEST_DIR.mkdir(exist_ok=True)
        with (MANIFEST_DIR / f"{case}.jsonl").open("w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        return
    for v in (
        B.MCAST_EARLY,  # 1   zero extra L1
        B.PIPE_A,  # 2   zero extra L1
        B.PIPE_A | B.MCAST_EARLY,  # 3   zero extra L1
        B.PIPE_A | B.HANDOFF2,  # 10  +1 fp32 page / tile-row
        B.PIPE_A | B.HANDOFF2 | B.MCAST_EARLY,  # 11  <- the recommended option
    ):
        one(v, base_frac)

    # The reader-fed input CB needs `xdeep` for the pipeline to be CORRECT at all;
    # run it with its remedy so the exception carries its price.
    if case in XDEEP_CASES:
        one(B.PIPE_A | B.XDEEP, base_frac)
        one(B.PIPE_A | B.HANDOFF2 | B.MCAST_EARLY | B.XDEEP, base_frac)

    # --- RING2 buys GROUP_SIZE extra fp32 pages per tile-row.  Find the LARGEST
    #     BLOCK_ROWS at which the descriptor's own solve still fits (lowering only
    #     its L1 budget), then run the SERIAL baseline and the no-ring pipeline
    #     there too, so the ring's cost and the BLOCK_ROWS change stay separable.
    if case in FULL_CASES:
        ring = B.PIPE_A | B.RING2 | B.HANDOFF2
        br = op_br
        pipe_frac = None
        while br >= 1:
            frac = base_frac if br == op_br else _resolve_safety_fraction(x, out, g, cfg, br)
            if one(ring, frac) is not None:
                pipe_frac = frac
                break
            br -= 1
        if pipe_frac is not None:
            one(ring | B.MCAST_EARLY, pipe_frac)
            if br != op_br:
                one(0, pipe_frac, note="@ringBR")
                one(B.PIPE_A | B.HANDOFF2 | B.MCAST_EARLY, pipe_frac, note="@ringBR")

    # ---- BIT-EXACTNESS post-pass -------------------------------------------
    # The pipeline reorders WHEN work is issued, never WHAT is computed, so at an
    # unchanged BLOCK_ROWS every variant's output must be BIT-IDENTICAL to the
    # serial one.  (A different BLOCK_ROWS only regroups rows, so it is compared
    # against torch by pcc instead.)
    serial_by_br = {r["block_rows"]: t for r, t in serial_out if r["variant"] == 0}
    for rec, t in serial_out:
        ref_t = serial_by_br.get(rec["block_rows"])
        if ref_t is not None:
            rec["bit_exact"] = bool(torch.equal(t, ref_t))
        print(f"  [{case}] {rec['name']:<24s} BR={rec['block_rows']} bit_exact={rec['bit_exact']}")

    known = EXPECT_INCORRECT.get(case)
    for rec in results:
        v = rec["variant"]
        rec["expected_incorrect"] = bool(known is not None and (v & known) == known and not (v & bench.XDEEP))

    MANIFEST_DIR.mkdir(exist_ok=True)
    with (MANIFEST_DIR / f"{case}.jsonl").open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    # Correctness gate -- the ONLY pass/fail.  A variant listed in EXPECT_INCORRECT
    # is a RECORDED domain exception (see the comment there), so it is reported, not
    # asserted; anything else that is not bit-identical to the serial run at the same
    # BLOCK_ROWS is a bug in this bench and fails the test.
    for rec in results:
        if rec["expected_incorrect"]:
            print(
                f"  [{case}] RECORDED EXCEPTION {rec['name']}: pcc {rec['pcc']:.6f} "
                f"bit_exact={rec['bit_exact']} (needs `xdeep`)"
            )
            continue
        assert rec["pcc"] > 0.999, f"{case}/{rec['name']}: pcc {rec['pcc']}"
        assert rec["bit_exact"] is not False, (
            f"{case}/{rec['name']}: output differs from the serial variant at the same "
            f"BLOCK_ROWS -- the reorder changed the ARITHMETIC, not just the schedule"
        )
