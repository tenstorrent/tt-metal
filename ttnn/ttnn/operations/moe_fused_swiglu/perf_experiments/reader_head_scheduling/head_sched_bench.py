# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED BAKE-OFF (blocking-perf-part-optimizer) — moe_fused_swiglu's READER SERIAL HEAD.

Assigned idea: break the head-of-line blocking at the top of the reader's per-M-block chain, so `x`
reaches every core's gate/up matmul sooner. The head is, in order:

    1. `cb_reserve_back(cb_w_gate)` + ISSUE `kr` coalesced bank-run reads of this core's 79.5 KB
       W_gate K-block                                          (zone `reader_wg_issue`)
    2. `cb_reserve_back(cb_x_tiles)` + the x staging: 32 row-major stick reads of `kr*64` B,
       `noc_async_read_barrier()`, push `cb_x_in`, wait compute's fused tilize, L1 self-copy into
       the resident slot                                        (zone `reader_xstage`)
    3. the x row-multicast rounds                               (zone `reader_xmcast`)

Step 2's barrier is ALL-OR-NOTHING, so it drains step 1's whole weight prefetch: `cb_x_in` — and
therefore compute's fused tilize, and therefore the row-multicast rendezvous — cannot start until
the 79.5 KB weight block has landed.

WHY THIS IS A WHOLE-OP CLONE AND NOT A MICRO-BENCH
--------------------------------------------------
Round 1's single-core `xstage_coalesce` bench measured this stage at ~1.3 us against 15.2 us in the
op, a 12x understatement, because the effect being measured IS 110-core / 8-bank DRAM contention.
So this bake-off imports the op's OWN `create_program_descriptor`, re-points its `KERNEL_DIR` at
`perf_experiments/reader_head_scheduling/kernels/` (byte-for-byte copies, with only the reader's
serial head parameterised by the `HEAD_VARIANT` define) and measures the WHOLE op on the real
110-core grid at the exact focus shape. The real op's files are never touched.

VARIANTS (see the banner in kernels/moe_fused_swiglu_reader.cpp for the full rationale)
    0 baseline            verbatim shipped order + plain all-or-nothing barriers
    1 wg_after_prologue   W_gate issued after the whole x prologue (the historical control)
    2 x_trid              baseline order, x reads on their own transaction id
    3 x_first_trid        x issued first, W_gate ~32 instructions later, x drained on its own trid
    4 wg_split_trid       W_gate rows [0, WG_HEAD_ROWS) first, then x, then the W_gate tail
    5 x_first_notrid      variant 3's order with PLAIN barriers (isolates order from drain)

PRECISION CONTRACT: frozen and NOT a lever. Every variant runs `default_compute_kernel_config()`
verbatim (LoFi / approx / no fp32 DEST / half sync / bfp8_pack_precise) and the same dtypes. The
change is a re-ORDER and a re-BARRIER of identical NoC transactions over identical bytes into
identical L1 addresses, so the output must be BIT-IDENTICAL to the baseline's — which this bench
ASSERTS (`torch.equal`) rather than merely PCC-checking.

This module is a plain library (NOT collected by pytest — pytest's --import-mode=importlib cannot
safely collect a test_*.py living inside `ttnn/ttnn/operations/...`; it derives a dotted module path
starting with "ttnn" and re-executes ttnn/ttnn/__init__.py under a second qualified name, crashing on
duplicate C++ op registration). The pytest entry point that imports this module lives at
tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_reader_head_scheduling.py
(uniquely named for this idea so it cannot collide with a sibling part-optimizer's probe).
"""

import os

# Enable the on-device profiler IN-PROCESS (must be set before the device opens). The pytest entry
# point imports this module before the `device` fixture runs, so module-import time is early enough.
#
# Two modes, deliberately kept apart:
#   * plain `run_safe_pytest.sh`  -> we own the profiler; the C++ post-processor fills
#     `program_analyses_results` so `read_kernel_ns()` returns the whole-op duration IN PROCESS.
#     This is the mode that produces the bake-off's ns numbers.
#   * `run_safe_pytest.sh --profile` -> the tracy wrapper already set TT_METAL_DEVICE_PROFILER, and
#     it owns the device log. We must NOT also switch the C++ post-processor on, or it consumes the
#     log tracy is about to report from. This is the mode that produces the per-STAGE zone data.
#   * `MOE_HEAD_NO_PROFILER=1` -> no profiler at all. Required for a `--dev` (watcher) pass: zones
#     share SRAM with the Watcher and DPRINT and cannot be enabled together. In that mode the ns
#     column reads `n/a` and only the correctness gate (bit-identity vs the baseline) runs.
IN_PROCESS_PROFILER = ("TT_METAL_DEVICE_PROFILER" not in os.environ) and not os.environ.get("MOE_HEAD_NO_PROFILER")
if IN_PROCESS_PROFILER:
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"

from contextlib import contextmanager
from pathlib import Path

# NOTE: `torch` is imported LAZILY. `scripts/validate_no_global_torch_imports.py` forbids a
# module-level torch import anywhere under `ttnn/ttnn/`, and these perf-experiment benches obey the
# same rule.
import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu as _op
from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu_program_descriptor as _pd

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 2048
BFP4_TILE = 576
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

VARIANTS = {
    0: "baseline",
    1: "wg_after_prologue",
    2: "x_trid",
    3: "x_first_trid",
    4: "wg_split_trid",
    5: "x_first_notrid",
}

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


# ---------------------------------------------------------------------------
# Variant selection — the ONLY host-side hook. `KERNEL_DIR` is re-pointed at this experiment's
# kernel copies and the reader/writer descriptors get one extra `defines` pair. Nothing else about
# the op's descriptor (CB sizes, grid, semaphores, compute config, runtime args) is touched, so a
# variant differs from the shipped op by exactly the reader's head schedule.
# ---------------------------------------------------------------------------
@contextmanager
def variant_build(variant, wg_head_rows=4):
    orig_dir = _pd.KERNEL_DIR
    orig_kd = ttnn.KernelDescriptor
    extra = [("HEAD_VARIANT", str(int(variant))), ("WG_HEAD_ROWS", str(int(wg_head_rows)))]

    def patched(*args, **kwargs):
        src = str(kwargs.get("kernel_source", ""))
        if src.endswith("moe_fused_swiglu_reader.cpp") or src.endswith("moe_fused_swiglu_writer.cpp"):
            kwargs["defines"] = list(kwargs.get("defines") or []) + extra
        return orig_kd(*args, **kwargs)

    _pd.KERNEL_DIR = KERNEL_DIR
    ttnn.KernelDescriptor = patched
    try:
        yield
    finally:
        _pd.KERNEL_DIR = orig_dir
        ttnn.KernelDescriptor = orig_kd


# ---------------------------------------------------------------------------
# Inputs — byte-identical to tests/.../test_moe_fused_swiglu_r2_perf.py's `_build`, including the
# hostile sentinel in the phantom rows (hazard 3: rows [count, ceil_tile(count)) must never leak).
# ---------------------------------------------------------------------------
def build_inputs(device, emb, capacity, count, input_format, seed=42):
    import torch

    torch.manual_seed(seed)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, to_dev(counts), to_dev(idx)


def torch_reference(tt_x, tt_w, count):
    """Reference from the ROUND-TRIPPED device tensors, so the only error left is the kernel's own
    accumulation order — not the bfp4/bfp8 quantisation, which is a property of the inputs."""
    import torch

    x = ttnn.to_torch(tt_x).to(torch.float32)[0, 0, :count, :]
    wg, wu, wd = (ttnn.to_torch(w).to(torch.float32) for w in tt_w)
    h = torch.nn.functional.silu(x @ wg) * (x @ wu)
    return h @ wd


def pcc(a, b):
    import torch

    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if float(denom) > 0 else 0.0


# ---------------------------------------------------------------------------
# Measurement — /perf-measure discipline: ONE run per (variant, case). Device kernel time has no
# warm-up transient, so a trial loop would only re-measure the same number N times.
# ---------------------------------------------------------------------------
def read_kernel_ns(device):
    if not IN_PROCESS_PROFILER:
        return None  # the tracy wrapper owns the log in --profile mode; read the CSV instead
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


def run_and_measure(device, variant, tt_x, tt_w, tt_counts, tt_idx, wg_head_rows=4):
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # drain the pending profiler window
    with variant_build(variant, wg_head_rows):
        out = _op(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    ns = read_kernel_ns(device)
    if IN_PROCESS_PROFILER:
        assert ns is not None, "profiler produced no data (profiler-enabled build / env not set?)"
    return out, ns


# ---------------------------------------------------------------------------
# /perf-ceiling-dm — the W_gate head's transaction count and issue-rate bound, computed from the
# SAME arithmetic `BankRuns::run()` uses on device, so the count is derived, not guessed.
# ---------------------------------------------------------------------------
def bank_run_txn_count(j0, jend, slots, wrun):
    j, n = j0, 0
    while j < jend:
        r = jend - j
        edge = slots - (j % slots)
        r = min(r, edge, wrun)
        j += r
        n += 1
    return n


def wgate_head_transactions(emb, hgroups=11, kgroups=10, num_banks=8, wrun=8, hidden=HIDDEN):
    """Per-core and grid-wide W_gate transaction counts + bytes for one M-block."""
    emb_t, hid_t = emb // TILE, hidden // TILE
    hn_pad = (hid_t + hgroups - 1) // hgroups
    slots_h = hid_t // num_banks
    kr_pad = (emb_t + kgroups - 1) // kgroups
    # `_split`'s `base + (i < rem)` rule, verbatim: the first `rem` grid ROWS own `kr_pad` emb tiles
    # and the rest own one fewer.
    base, rem = emb_t // kgroups, emb_t % kgroups
    rows = [base + (1 if r < rem else 0) for r in range(kgroups)]
    cols = []
    for c in range(hgroups):
        hstart = c * hn_pad
        hn = min(hn_pad, hid_t - hstart)
        cols.append((hn, bank_run_txn_count(hstart, hstart + hn, slots_h, wrun)))
    per_core = {}
    grid_txn = grid_bytes = 0
    for r, kr in enumerate(rows):
        for c, (hn, tpk) in enumerate(cols):
            per_core[(c, r)] = (kr * tpk, kr * hn * BFP4_TILE)
            grid_txn += kr * tpk
            grid_bytes += kr * hn * BFP4_TILE
    return {
        "kr_pad": kr_pad,
        "hn_pad": hn_pad,
        "slots_h": slots_h,
        "rows_kr": rows,
        "cols_txn_per_krow": [t for _, t in cols],
        "per_core": per_core,
        "grid_txn": grid_txn,
        "grid_bytes": grid_bytes,
    }
