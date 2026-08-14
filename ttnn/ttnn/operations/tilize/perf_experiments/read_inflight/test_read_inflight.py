# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight — isolated bake-off for BYTES IN FLIGHT PER READ BARRIER.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight/test_read_inflight.py

Correctness is the ONLY pass/fail and tilize is a PERMUTATION, so the bar is
bit-exact (`torch.equal`). Perf is measured and printed, never asserted.

One fresh-cache warm launch (compile + program cache) then ONE measured launch
per arm — device kernel duration has no warm-up transient.

MEASURED, Wormhole B0 (n150), bf16 unless stated. `DEVICE KERNEL DURATION [ns]`.
Every arm below was bit-exact; the repeated arms are 3-run medians (spread<0.5%).

  arm                 focus(8blk)  tall(32blk)  small(4blk)  interleaved(64c)
  baseline_helper        14758        55060         7567          86896
  raw_nt1_d2             14946        56459         7493          87203
  nt2_d2                 14100        50978         7064          88156
  nt4_d2                 14180        48440         6943          88475
  nt8_d2                 15805        48860           --             --
  nt1_d3 (depth only)    15024        56094         7531          85793
  nt1_d8 (depth only)    15043        55920         7470          88104
  trid_nt1_d2 (B8)       14270        53561         6608          87228
  ahead1_nt1_d3          12410        44376         6386          85917   <-- rec.
  ahead1_nt2_d3          13000        43680         6442          86799
  ahead2_nt1_d4          12812        44631         6465          86349
  ahead4_nt1_d6          13671        45474           --          (n/a)

  oneblock (1 blk/core): baseline 3102, ahead1_nt1_d3 3034  (flat, bit-exact)
  fp32 focus:            baseline 17235, ahead1_nt1_d3 15211 (1.13x)
  fp32 small:            baseline  7761, ahead1_nt1_d3  6683 (1.16x)
  fp32 focus trid_nt1_d2 (the op's own two-slot B8 lever): 22865 -- REGRESSION

The read-floor probe (`test_probe_readfloor.py`, reader only, no consumer,
128 KB/core over 8 cores) is what explains the shape of this table:
  512 B transfers, barrier every {32,64,128,256} pages -> 14709/13144/12533/12107
  same 128 KB as {512,1024,2048,4096} B pages, one barrier -> 12089/6383/5627/6115
  512 B / 32-page cadence on {8,32,64} cores -> 14709/21069/41932 ns
    (aggregate 71 / 199 / 200 GB/s -- the fabric saturates by 32 cores)
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py
# — the shipped package must not drag torch in). These perf-experiment benches DO need it
# for their bit-exact oracle, so the import is done inside a function scope and published
# under the module-global name, which keeps every `torch.` use below unchanged.
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.read_inflight import descriptor as D

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# ---------------------------------------------------------------------------
# Regimes. `focus` is the perf-flagged plan; the other two are the domain sweep.
REGIMES = {
    # the DRAM -> local-L1-shard crossover the idea targets (8 blocks/core)
    "focus": dict(shape=[1, 1, 2048, 256], num_cores=8),
    # small crossover: per-core-overhead regime, 128 B per transfer (4 blocks/core)
    "small": dict(shape=[1, 1, 512, 64], num_cores=4),
    # many cores already saturating DRAM (64-core interleaved DRAM -> DRAM)
    "interleaved": dict(shape=[1, 1, 2048, 2048], grid=(8, 8), n_chunks=4),
    # SAME crossover topology, 4x taller shard (32 blocks/core). The focus plan's
    # 8 blocks/core is short enough that one group's compute tail is a visible
    # fraction of the wall; this regime is what separates "the lever is small"
    # from "the focus plan is tail-dominated".
    "tall": dict(shape=[1, 1, 8192, 256], num_cores=8),
    # ONE block per core — the issue-ahead window cannot fill. Expected FLAT
    # (the schedule degenerates to the baseline's), and it must stay bit-exact.
    "oneblock": dict(shape=[1, 1, 256, 256], num_cores=8),
}


def _height_shard(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2] // num_cores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _make(device, regime, dtype):
    cfg = REGIMES[regime]
    shape = cfg["shape"]
    torch_in = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    tt_in = ttnn.from_torch(
        torch_in, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if "num_cores" in cfg:
        mem = _height_shard(shape, cfg["num_cores"])
        spec = ttnn.TensorSpec(
            ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, mem.memory_layout, mem.shard_spec, mem.buffer_type
        )
        plan = D.plan_height_sharded(shape, dtype, cfg["num_cores"])
    else:
        spec = ttnn.TensorSpec(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.DRAM)
        plan = D.plan_interleaved(shape, dtype, cfg["grid"], cfg["n_chunks"])
    tt_out = ttnn.allocate_tensor_on_device(spec, device)
    return torch_in, tt_in, tt_out, plan


def _run_arm(device, regime, dtype, label, *, variant, nt_blk, cb_depth, ahead=1):
    torch_in, tt_in, tt_out, plan = _make(device, regime, dtype)
    desc = D.build(tt_in, tt_out, plan, variant=variant, nt_blk=nt_blk, cb_depth=cb_depth, ahead=ahead)

    ttnn.generic_op([tt_in, tt_out], desc)  # warm: compile + program cache
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window

    out = ttnn.generic_op([tt_in, tt_out], desc)  # the ONE measured launch
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    got = ttnn.to_torch(out)
    exact = torch.equal(got, torch_in)
    l1_in = plan.in_cb_bytes(nt_blk, cb_depth)
    l1_out = plan.out_cb_bytes(2)
    logger.info(
        f"READ_INFLIGHT {regime}/{label}: ns={ns} bit_exact={exact} "
        f"in_cb={l1_in}B out_cb={l1_out}B total_cb={l1_in + l1_out}B "
        f"(budget {D.CB_L1_BUDGET}B) blocks/core={plan.blocks_per_core} row_bytes={plan.row_bytes}"
    )
    assert exact, f"{regime}/{label}: NOT bit-exact — arm disqualified"
    assert l1_in + l1_out <= D.CB_L1_BUDGET, f"{regime}/{label}: over the op's CB L1 budget"
    assert ns is not None, "profiler produced no data"
    return ns


# ---------------------------------------------------------------------------
# ARMS. `baseline` is the op's current approach for this part, verbatim.
# `raw_nt1` is the control that separates "raw vs helper" from "NT_BLK effect".
ARMS = {
    "baseline_helper": dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2),
    "raw_nt1_d2": dict(variant=D.VARIANT_RAW, nt_blk=1, cb_depth=2),
    "nt2_d2": dict(variant=D.VARIANT_RAW, nt_blk=2, cb_depth=2),
    "nt4_d2": dict(variant=D.VARIANT_RAW, nt_blk=4, cb_depth=2),
    "nt8_d2": dict(variant=D.VARIANT_RAW, nt_blk=8, cb_depth=2),
    "nt1_d3": dict(variant=D.VARIANT_RAW, nt_blk=1, cb_depth=3),
    "nt1_d4": dict(variant=D.VARIANT_RAW, nt_blk=1, cb_depth=4),
    "nt1_d8": dict(variant=D.VARIANT_RAW, nt_blk=1, cb_depth=8),
    "trid_nt1_d2": dict(variant=D.VARIANT_TRID, nt_blk=1, cb_depth=2),
    "trid_nt2_d2": dict(variant=D.VARIANT_TRID, nt_blk=2, cb_depth=2),
    "trid_nt4_d2": dict(variant=D.VARIANT_TRID, nt_blk=4, cb_depth=2),
    # fusion: grouping + depth
    "nt2_d4": dict(variant=D.VARIANT_RAW, nt_blk=2, cb_depth=4),
    "nt4_d4": dict(variant=D.VARIANT_RAW, nt_blk=4, cb_depth=4),
    # FUSION (1)+(2)+(3): NT_BLK grouping + issue-ahead + a CB deep enough to
    # hold every outstanding group, so in-flight depth is bought WITHOUT giving
    # up push granularity (what caps the two-slot B8 form).
    "ahead1_nt1_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1),
    "ahead2_nt1_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=2),
    "ahead3_nt1_d5": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=5, ahead=3),
    "ahead4_nt1_d6": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=6, ahead=4),
    "ahead1_nt2_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=2, cb_depth=3, ahead=1),
    "ahead2_nt2_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=2, cb_depth=4, ahead=2),
    "ahead3_nt2_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=2, cb_depth=4, ahead=3),
    "ahead1_nt4_d3": dict(variant=D.VARIANT_AHEAD, nt_blk=4, cb_depth=4, ahead=1),
    # `cb_depth - (ahead + 1)` is the SLACK: how many finished groups compute may
    # still be sitting on when the reader wants to issue the next one. With zero
    # slack the reader has to wait for a full drain, which is what caps the
    # two-slot B8 form; these arms sweep the slack separately from `ahead`.
    "ahead1_nt1_d4": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=1),
    "ahead2_nt1_d6": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=6, ahead=2),
    "ahead3_nt1_d8": dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=8, ahead=3),
}


def _blocks_per_core(regime):
    cfg = REGIMES[regime]
    if "num_cores" in cfg:
        return cfg["shape"][-2] // cfg["num_cores"] // D.TILE_H
    return (cfg["shape"][-2] // D.TILE_H) * cfg["n_chunks"] // (cfg["grid"][0] * cfg["grid"][1])


def _skip_if_impossible(regime, arm):
    per_core = _blocks_per_core(regime)
    if per_core % arm["nt_blk"]:
        pytest.skip(f"nt_blk={arm['nt_blk']} does not divide {per_core} blocks/core")
    if arm["variant"] == D.VARIANT_TRID and per_core // arm["nt_blk"] < 2:
        pytest.skip("the two-slot B8 form needs at least two groups")


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("regime", list(REGIMES))
def test_arm(device, regime, arm):
    _skip_if_impossible(regime, ARMS[arm])
    _run_arm(device, regime, ttnn.bfloat16, arm, **ARMS[arm])


# --- domain: dtype ---------------------------------------------------------
# fp32 doubles the transfer size (1024 B per stick on `focus`) and runs under
# the op's fp32 exactness contract (fp32 DEST + lossless unpack), which is
# applied IDENTICALLY to every arm. Bit-exactness is still the pass bar.
_FP32_ARMS = ["baseline_helper", "raw_nt1_d2", "nt2_d2", "trid_nt1_d2", "ahead1_nt1_d3", "ahead1_nt1_d4"]


@pytest.mark.parametrize("arm", _FP32_ARMS)
@pytest.mark.parametrize("regime", ["focus", "small"])
def test_arm_fp32(device, regime, arm):
    _skip_if_impossible(regime, ARMS[arm])
    _run_arm(device, regime, ttnn.float32, f"fp32/{arm}", **ARMS[arm])


# --- repeat the headline pair, 3x, so the win/null call is not on noise -----
@pytest.mark.parametrize("rep", [0, 1, 2])
@pytest.mark.parametrize("arm", ["baseline_helper", "ahead1_nt1_d3", "ahead1_nt2_d3"])
@pytest.mark.parametrize("regime", ["focus", "tall", "interleaved"])
def test_repeat(device, regime, arm, rep):
    _skip_if_impossible(regime, ARMS[arm])
    _run_arm(device, regime, ttnn.bfloat16, f"rep{rep}/{arm}", **ARMS[arm])
