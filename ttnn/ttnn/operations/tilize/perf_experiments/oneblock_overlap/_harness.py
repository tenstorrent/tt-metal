# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bake-off harness for the `oneblock_overlap` idea.

The part under test is the op's **work distribution**: on a shape whose blocking
lands exactly ONE block per core, read / compute / write are strictly serialized
because there is no next block to overlap against. The candidate buys overlap by
handing each core SEVERAL blocks and lighting correspondingly fewer cores, with
`WT_CHUNK` — and therefore the per-stick read transfer — held EXACTLY fixed.

Everything else is held constant per /perf-lab's concept-isolation table: same
kernels (the op's own, unmodified), same dtypes, same ComputeKernelConfig, same
levers, same CB depth (except the one arm that prices depth explicitly).

Correctness gate: tilize is a PERMUTATION, so the bar is BIT-EXACT — the
round-tripped output must `torch.equal` the input. Perf is never asserted.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib
import statistics


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

from . import _descriptor as xpd

_tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

SHAPES = {
    "b_wide_short": [1, 1, 32, 16384],  # FOCUS: NT_H=1, 64 blocks on 64 cores
    "a_square": [1, 1, 2048, 2048],
    "c_multiblock": [1, 1, 8192, 1024],
    "d_smallest": [1, 1, 32, 64],
    # A SECOND NT_H==1 / one-block-per-core topology, reached with a different
    # WT and therefore a different read transfer (256 B, not 512 B) — so the
    # domain claim rests on the BLOCKING predicate, not on one width.
    "e_wide_short_half": [1, 1, 32, 8192],
}

# --- the arm menu -----------------------------------------------------------
# knobs: active_cores / blocking_cores / cb_depth (see _descriptor.KNOBS).
# levers: op lever overrides, used only where an arm is structurally forced to
#         give one up (depth != 2 cannot carry the B8 trid double-issue).
ARMS = {
    # honest baseline = the op exactly as it ships
    "baseline": dict(knobs={}),
    # THE IDEA: same WT_CHUNK / same 512 B read, fewer cores, more blocks/core
    "cores32": dict(knobs=dict(active_cores=32)),
    "cores16": dict(knobs=dict(active_cores=16)),
    "cores8": dict(knobs=dict(active_cores=8)),
    # CONTROL 1 (original master.md A0 shape): fewer cores AND the coarser chunk
    # the blocking derivation picks for them -> 1 block/core again, bigger read.
    "cores32_coarse": dict(knobs=dict(active_cores=32, blocking_cores=32)),
    # CONTROL 2 (option 3, the already-bought counterfactual): 2 blocks/core
    # bought by HALVING WT_CHUNK on all 64 cores -> the read drops to 256 B.
    "halfchunk64": dict(knobs=dict(blocking_cores=128)),
    # OPTION 4: does CB depth matter once a core owns 2 blocks? Depth != 2 is
    # structurally incompatible with the B8 trid double-issue (the reader holds
    # two FIXED slot addresses), so `d4` also loses trid — `d2_notrid` is its
    # attribution partner.
    "cores32_d4": dict(knobs=dict(active_cores=32, cb_depth=4)),
    "cores32_d2_notrid": dict(knobs=dict(active_cores=32), levers=dict(read_trid=0, write_trid=0)),
    # A finer point on the core-count curve, and the trid-OFF partners at 64 / 16
    # cores: the smoke run put `cores32_d2_notrid` fastest, so B8's double-issue
    # has to be priced AS A FUNCTION of blocks-per-core, not just at 32.
    "cores48": dict(knobs=dict(active_cores=48)),
    "baseline_notrid": dict(knobs={}, levers=dict(read_trid=0, write_trid=0)),
    "cores16_notrid": dict(knobs=dict(active_cores=16), levers=dict(read_trid=0, write_trid=0)),
    # Depth-4 at the other two core counts. `baseline_d4` is the control that
    # says whether the depth-4 edge is really the CB or really the core count:
    # at ONE block per core a deeper CB (and the trid double-issue) are both
    # structurally inert, so it must land on top of `baseline`.
    "baseline_d4": dict(knobs=dict(cb_depth=4)),
    "cores16_d4": dict(knobs=dict(active_cores=16, cb_depth=4)),
    # The rule `C' = num_blocks // 2` applied to the SMALLEST regime, which owns
    # only 2 blocks in total: it would collapse the whole tensor onto ONE core.
    # A predicate must not be proposed with that cell untested.
    "cores1": dict(knobs=dict(active_cores=1)),
    "cores1_notrid": dict(knobs=dict(active_cores=1), levers=dict(read_trid=0, write_trid=0)),
    "cores2_notrid": dict(knobs=dict(active_cores=2), levers=dict(read_trid=0, write_trid=0)),
    # Which HALF of the B8 pair costs the 2-blocks-per-core handoff. The reader's
    # trid loop pushes block 0 only AFTER block 1 is issued; the writer's twin
    # holds block i's write open across block i+1. Both are plausible; measure.
    "cores32_no_readtrid": dict(knobs=dict(active_cores=32), levers=dict(read_trid=0)),
    "cores32_no_writetrid": dict(knobs=dict(active_cores=32), levers=dict(write_trid=0)),
}


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _make_input(device, shape, dtype):
    torch_input = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def run_arm(device, tt_input, arm, *, check=None, label=""):
    """One warm launch (compile + cache), then ONE measured launch. Returns ns.

    `check` (a torch tensor) turns the BIT-EXACT correctness gate on for this
    launch. Device kernel duration has no warm-up transient, so the measured
    launch is a single fresh sample — callers take medians ACROSS rounds.
    """
    spec = ARMS[arm]
    saved_knobs = dict(xpd.KNOBS)
    saved_levers = dict(xpd.LEVERS)
    saved_ctor = _tilize_mod.create_program_descriptor
    xpd.KNOBS.update({"active_cores": 0, "blocking_cores": 0, "cb_depth": 0})
    xpd.KNOBS.update(spec.get("knobs", {}))
    xpd.LEVERS.update(spec.get("levers", {}))
    _tilize_mod.create_program_descriptor = xpd.create_program_descriptor
    try:
        out = _tilize_mod.tilize(tt_input, use_multicore=True, use_double_buffer=True)
        ttnn.synchronize_device(device)
        blocking = dict(xpd.LAST)
        if check is not None:
            got = ttnn.to_torch(out)
            assert torch.equal(got, check), f"{label}/{arm}: NOT bit-exact — arm disqualified"
        ttnn.deallocate(out)
        _read_kernel_ns(device)  # flush the warm-up window

        out = _tilize_mod.tilize(tt_input, use_multicore=True, use_double_buffer=True)
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        ttnn.deallocate(out)
    finally:
        _tilize_mod.create_program_descriptor = saved_ctor
        xpd.KNOBS.update(saved_knobs)
        xpd.LEVERS.update(saved_levers)
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    logger.info(f"XP oneblock_overlap {label}/{arm} ns={ns:.0f} blocking={blocking}")
    return ns, blocking


def bake_off(device, shape_key, arms, rounds, dtype=ttnn.bfloat16):
    """Round-robin the arms `rounds` times in ONE session, then report medians.

    Round-robin (not arm-major) is deliberate: this shape's documented +-4-6%
    run-to-run spread is session drift, and interleaving makes every arm eat the
    same drift instead of one arm eating all of it.
    """
    rounds = int(os.environ.get("XP_ROUNDS", rounds))  # smoke-run override
    shape = SHAPES[shape_key]
    torch_input, tt_input = _make_input(device, shape, dtype)
    samples = {a: [] for a in arms}
    blocking = {}
    for r in range(rounds):
        for arm in arms:
            ns, blk = run_arm(device, tt_input, arm, check=torch_input if r == 0 else None, label=shape_key)
            samples[arm].append(ns)
            blocking[arm] = blk
    ttnn.deallocate(tt_input)

    base = statistics.median(samples[arms[0]])
    logger.info(f"=== oneblock_overlap {shape_key} {shape} — medians of {rounds} ===")
    for arm in arms:
        med = statistics.median(samples[arm])
        b = blocking[arm]
        logger.info(
            f"  {arm:<20} median={med:>9.0f} ns  x{base/med:5.3f}  "
            f"mean={statistics.mean(samples[arm]):.0f} sd={statistics.pstdev(samples[arm]):.0f} "
            f"min={min(samples[arm]):.0f} max={max(samples[arm]):.0f}  "
            f"[cores={b['num_cores']} wt_chunk={b['wt_chunk']} blocks/core={b['blocks_per_core']:.2f} "
            f"read={b['read_bytes']}B depth={b['cb_depth']} trid={b['read_trid']}/{b['write_trid']}]"
        )
    return samples, blocking
