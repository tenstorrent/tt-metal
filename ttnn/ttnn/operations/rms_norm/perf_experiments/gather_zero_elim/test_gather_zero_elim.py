# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gather_zero_elim -- is rms_norm's gather boot-zeroing needed AT ALL?

TWO benches, one question each.

  poison_bench   SAFETY, the pass/fail.  Seeds the faces the gather never ships with
                 nine catastrophic patterns (1e30, -1e30, NaN, +-Inf, subnormals, a
                 mixed pattern that makes the fold evaluate Inf + -Inf, and a stale-L1
                 lookalike) and runs the op's REAL fold + finalize + pass-B consumer.
                 A poisoned seed's output is compared BIT-EXACTLY against the same run
                 with those lanes zeroed (== what the boot achieves) and against an
                 fp64 reference on pcc AND rel-RMS.  The odd-GROUP_SIZE PAD page is a
                 SEPARATE axis: it is folded WHOLE, so it may still need zeroing even
                 if faces 1/3 do not.

  zero_bench     PRICE.  The boot's own ns, and every alternative to deleting it
                 (pad-only carve-out, a half-transaction scratch form, whole-CB, and
                 the same work issued from the idle reader/NoC0).  Byte-exact
                 correctness gate on which bytes each scheme actually zeroes.

Run (foreground, one fresh-cache profiled run per launch, NO trial loop):

  source python_env/bin/activate ; unset TT_METAL_DPRINT_CORES
  scripts/run_safe_pytest.sh --run-all --profile \
      ttnn/ttnn/operations/rms_norm/perf_experiments/gather_zero_elim/test_gather_zero_elim.py

then join the profiler CSV with this run's launch log:

  python3 ttnn/ttnn/operations/rms_norm/perf_experiments/gather_zero_elim/report.py

Every launch is ONE `ttnn.generic_op`, so the CSV rows and launches.jsonl lines are 1:1
in order.  A failing gate still logs its line (the metrics ARE the deliverable), so the
run uses --run-all.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"gze_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


poison = _load("poison_bench")
zero = _load("zero_bench")

# The op's own soft gates for the focus case.
PCC_GATE = 0.9995
RELRMS_GATE = 0.04

# One JSONL line per `ttnn.generic_op` launch, in launch order, so report.py can join it
# with the profiler CSV positionally.  The suite is split across two profiled invocations
# (the poison half and the price half) to keep each run's wall time sane, so the log name
# is a parameter rather than a constant -- report.py takes the matching pair.
LOG = HERE / os.environ.get("GZE_LOG", "launches.jsonl")

WT = int(os.environ.get("GZE_WT", "4"))  # the focus shard's width in tiles ([1024,128] -> 4)

# ---------------------------------------------------------------------------
# geometries
# ---------------------------------------------------------------------------
# FOCUS: (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8 -> GROUP_SIZE 8, BLOCK_ROWS 8.
FOCUS_GEOM = (8, 8)
# The op's other live combine geometries, plus the ODD one (GROUP_SIZE 9 == the
# `wshard_w2304_9c` profile, the only shape with a pad slot).
LIVE_GEOMS = [(8, 8), (32, 1), (28, 1), (9, 1)]

# DOMAIN sweep, capped at 300 fp32-page equivalents (~1.2 MB) of resident L1 -- the same
# budget the op's own BLOCK_ROWS solve respects, so an excluded corner is not a reachable
# op configuration rather than an untested one.
_SWEEP_G = (4, 8, 9, 28, 32)
_SWEEP_ROWS = (1, 8, 32)
PAGE_CAP = 300
POISON_SWEEP = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if poison.l1_fp32_pages(g, r, WT) <= PAGE_CAP]
# zero_bench pins only the gather CB, so it reaches every corner of the sweep.
ZERO_SWEEP = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if zero.gather_slots(g) * r <= 320]


def _record(line, tag):
    line = dict(line)
    line["tag"] = tag
    with LOG.open("a") as f:
        f.write(json.dumps(line) + "\n")
    print("LAUNCH " + json.dumps(line))
    return line


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    if LOG.exists():
        LOG.unlink()
    yield


# ===========================================================================
# SAFETY: the faces the gather never ships
# ===========================================================================
# The reference run for a geometry, cached so the bit-exact comparison never re-launches.
_REF = {}


def _reference_run(device, group_size, rows, wt):
    key = (group_size, rows, wt)
    if key not in _REF:
        res, stat, out = poison.run_seed(device, "zero", "zero", group_size, rows, wt=wt)
        _REF[key] = (stat, out)
        line = _record({**res, "bench": "poison", "bit_equal_to_zeroed": True}, f"poison_ref_g{group_size}_r{rows}")
        assert line["pcc_out"] >= PCC_GATE and line["rel_rms_out"] <= RELRMS_GATE, (
            f"the ZEROED reference itself misses the op's gates at g={group_size} rows={rows}: "
            f"pcc={line['pcc_out']} rel_rms={line['rel_rms_out']} -- the bench is wrong, not the op"
        )
    return _REF[key]


def _run_poison(device, face_seed, pad_seed, group_size, rows, tag, wt=WT):
    import torch

    ref_stat, ref_out = _reference_run(device, group_size, rows, wt)
    res, stat, out = poison.run_seed(device, face_seed, pad_seed, group_size, rows, wt=wt)
    res["bench"] = "poison"
    # The STRONGEST statement available: identical BITS to the boot-zeroed run.
    res["bit_equal_to_zeroed"] = bool(torch.equal(out, ref_out))
    res["stat_col0_bit_equal"] = bool(torch.equal(stat[:, 0], ref_stat[:, 0]))
    line = _record(res, tag)
    if (face_seed, pad_seed) == poison.REFERENCE:
        return line
    # Correctness is the only pass/fail.  A seed that breaks the op's soft gates means
    # the boot-zeroing of THAT region is REQUIRED -- reported, not swallowed.
    assert line["pcc_out"] >= PCC_GATE and line["rel_rms_out"] <= RELRMS_GATE, (
        f"face_seed={face_seed} pad_seed={pad_seed} g={group_size} rows={rows}: "
        f"pcc_out={line['pcc_out']} rel_rms_out={line['rel_rms_out']} "
        f"(stat col0 non-finite {line['stat_col0_nonfinite']:.3f}, out non-finite {line['out_nonfinite']:.3f}) "
        "-- this region's boot-zeroing IS load-bearing"
    )
    return line


@pytest.mark.parametrize("face_seed", poison.FACE_SEEDS)
def test_poison_faces_focus(device, face_seed):
    """FOCUS shape, every catastrophic seed in the unshipped faces (GROUP_SIZE 8: no pad)."""
    _run_poison(device, face_seed, "zero", *FOCUS_GEOM, tag="poison_faces_focus")


@pytest.mark.parametrize("group_size,rows", POISON_SWEEP)
def test_poison_faces_sweep(device, group_size, rows):
    """DOMAIN: the worst seed (`mixed` -- makes the fold evaluate Inf + -Inf) everywhere."""
    _run_poison(device, "mixed", "zero", group_size, rows, tag="poison_faces_sweep")


@pytest.mark.parametrize("face_seed", ("nan", "big_pos", "mixed"))
def test_poison_faces_wide(device, face_seed):
    """DOMAIN: a WIDER per-core slice (WT 8) -- pass B walks more tiles per stat tile."""
    _run_poison(device, face_seed, "zero", 8, 4, tag="poison_faces_wt8", wt=8)


@pytest.mark.parametrize("pad_seed", [s for s in poison.PAD_SEEDS if s != "zero"])
@pytest.mark.parametrize("group_size,rows", [(9, 1), (9, 8)])
def test_poison_pad(device, pad_seed, group_size, rows):
    """The ODD-GROUP_SIZE PAD PAGE -- a DIFFERENT question from faces 1/3.

    A pad page is folded WHOLE (it pairs against the odd contributor), so its faces 0/2
    land in column 0 and must be an exact +0.0.  `faces13` poisons ONLY the pad page's
    columns 16..31, i.e. asks whether the pad needs its unshipped faces zeroed too or
    only its shipped ones.  A failure here is a legitimate carve-out, not a bug.
    """
    _run_poison(device, "zero", pad_seed, group_size, rows, tag="poison_pad")


# ===========================================================================
# PRICE: the boot's own ns, and the alternatives
# ===========================================================================


def _run_zero(device, variant, group_size, rows, gather_faces, tag):
    res = zero.run_variant(device, variant, group_size, rows, gather_faces=gather_faces)
    res["bench"] = "zero"
    line = _record(res, tag)
    assert line["byte_exact"], (
        f"{variant} g={group_size} rows={rows} faces={gather_faces}: {line['mismatched_elems']} elements "
        "differ from the exact zero/intact pattern this scheme promises"
    )
    return line


@pytest.mark.parametrize("variant", zero.VARIANTS)
def test_zero_focus(device, variant):
    """FOCUS shape (GROUP_SIZE 8, BLOCK_ROWS 8, GATHER_FACES 2 -> 64 pages, 128 face zeros)."""
    _run_zero(device, variant, *FOCUS_GEOM, gather_faces=2, tag="zero_focus")


@pytest.mark.parametrize("gather_faces", (2, 3, 4))
@pytest.mark.parametrize("variant", ("none", "faces", "scratch", "faces_r"))
def test_zero_faces(device, variant, gather_faces):
    """DOMAIN over GATHER_FACES.  At 4 nothing is unshipped, so at an EVEN GROUP_SIZE the
    whole stage is inert BY CONSTRUCTION -- that is the third route to eliminating it."""
    _run_zero(device, variant, *FOCUS_GEOM, gather_faces=gather_faces, tag="zero_faces")


@pytest.mark.parametrize("group_size,rows", ZERO_SWEEP)
@pytest.mark.parametrize("variant", ("none", "faces", "pad_only", "scratch"))
def test_zero_sweep(device, variant, group_size, rows):
    """DOMAIN: GROUP_SIZE x BLOCK_ROWS.  GROUP_SIZE 9 is the odd one (pad slots)."""
    _run_zero(device, variant, group_size, rows, gather_faces=2, tag="zero_sweep")


@pytest.mark.parametrize("variant", zero.VARIANTS)
@pytest.mark.parametrize("group_size,rows", LIVE_GEOMS)
def test_zero_live(device, variant, group_size, rows):
    """The op's live combine geometries, full menu."""
    _run_zero(device, variant, group_size, rows, gather_faces=2, tag="zero_live")


@pytest.mark.parametrize("group_size,rows", [(9, 1), (9, 8), (9, 32)])
@pytest.mark.parametrize("variant", ("none", "faces", "pad_only", "pad_faces02"))
def test_zero_pad(device, variant, group_size, rows):
    """The ODD-GROUP_SIZE pad, priced three ways.

    `pad_faces02` is the MINIMUM the poison bench licenses: it zeroes only the pad
    page's SHIPPED faces (0 and 2), because a pad page with faces 1/3 poisoned and
    faces 0/2 zero measured BIT-IDENTICAL (`pad_seed=faces13`).
    """
    _run_zero(device, variant, group_size, rows, gather_faces=2, tag="zero_pad")
