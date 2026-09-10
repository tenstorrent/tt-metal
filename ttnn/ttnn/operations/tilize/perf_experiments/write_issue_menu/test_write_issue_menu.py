# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness — idea `write_issue_menu`.

PERF EXPERIMENT ONLY. Does not touch the real op; see `bench.py` for how the
op's own reader/compute kernels and descriptor are reused unmodified and only
the writer kernel source is swapped.

Correctness gate (bit identity for EVERY mode x EVERY sweep shape) — run this
first, WITHOUT --profile:

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/write_issue_menu/test_write_issue_menu.py \
        -k all_modes_correctness

Perf, ONE mode per invocation (mirrors the op's TILIZE_ABLATE env convention):

    for M in baseline rot_none rot_bankunif flush_half barrier_half cmdbuf2 cmdbuf4 posted; do
        WRITE_ISSUE_MODE_NAME=$M scripts/run_safe_pytest.sh --profile \
            ttnn/ttnn/operations/tilize/perf_experiments/write_issue_menu/test_write_issue_menu.py \
            -k "focus"
    done

Each `focus` invocation emits, IN THIS ORDER, one profiler row per dispatch:
    rows 1-3   isolated WRITE STAGE  (TILIZE_ABLATE=reads,compute) x 3 reps
    rows 4-6   whole op              (no ablation)                 x 3 reps
Read `DEVICE KERNEL DURATION [ns]` from the newest
`generated/profiler/reports/*/ops_perf_results*.csv`, and the `writer_issue` /
`writer_barrier` zone means from
`ttnn/ttnn/operations/tilize/perf_experiments/zone_report.py`.

The domain sweep is a separate `-k sweep` invocation (whole op only, 1 rep per
shape, bit-identity checked on every one).
"""

import ttnn

from ttnn.operations.tilize.perf_experiments.write_issue_menu.bench import (
    current_mode_from_env,
    run_variant_from_tt,
)

FOCUS = (1, 1, 32, 16384)  # bw=8, R=1, C=512, 64 blocks / 64 cores, 1 batch of 8 writes per core

# The domain sweep the task brief names, focus first.
SWEEP = [
    (FOCUS, "focus"),
    ((1, 1, 32, 32768), "short_wide_2x"),
    ((1, 1, 1024, 1024), "square_1k"),
    ((1, 1, 2048, 2048), "square_2k"),
    ((1, 1, 16384, 32), "tall_narrow"),  # bw=1 -> NO column order to permute (the edge case)
    ((1, 1, 2048, 64), "tall_narrow_small"),
]


def _make_input(device, shape):
    # Function-local torch import: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import under `ttnn/ttnn/`.
    import torch

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def _run_one(device, tt_input, torch_input, label, mode_name, ablate_stages, check):
    import torch

    tt_output, plan = run_variant_from_tt(tt_input, mode_name, ablate_stages=ablate_stages)
    print(
        f"\n[write_issue_menu mode={mode_name} {label} ablate={ablate_stages or 'none'}] "
        f"{tuple(tt_input.shape)}: bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} "
        f"rows/blk={plan.tensor_row_blocks // plan.num_row_groups} "
        f"wrpb={plan.groups[0].write_rows_per_barrier if plan.groups else '?'} "
        f"blocks={plan.num_blocks_total} cores={len(plan.assignment)} "
        f"split_reader={plan.groups[0].split_reader if plan.groups else 0}"
    )
    if check:
        got = ttnn.to_torch(tt_output)
        # tilize is a pure re-lay of the same bytes -> bit identity, never PCC.
        assert torch.equal(got, torch_input), f"{mode_name}/{label}: NOT bit-identical"
    return plan


def test_focus(device):
    """Focus shape. 3 reps of the isolated write stage, then 3 of the whole op.

    The ablated reps write garbage BY DESIGN (reads + compute payloads removed,
    all synchronization kept), so only the un-ablated reps are correctness
    gated — the `all_modes_correctness` test is the real gate."""
    mode_name = current_mode_from_env()
    torch_input, tt_input = _make_input(device, FOCUS)
    for _ in range(3):
        plan = _run_one(device, tt_input, torch_input, "focus/WRITE-STAGE", mode_name, "reads,compute", check=False)
    for _ in range(3):
        _run_one(device, tt_input, torch_input, "focus/WHOLE-OP", mode_name, "", check=True)
    assert plan.tensor_row_blocks == 1 and plan.tensor_col_tiles == 512
    assert len(plan.assignment) == 64, "the flagged number is only meaningful at 64/64 cores"


def test_sweep(device):
    """Domain sweep, whole op, one dispatch per shape, all bit-identity gated."""
    mode_name = current_mode_from_env()
    for shape, label in SWEEP:
        torch_input, tt_input = _make_input(device, shape)
        _run_one(device, tt_input, torch_input, label, mode_name, "", check=True)


# `cmdbuf4` HUNG the device on the focus shape (dispatch timeout, first run of
# the correctness gate) and is therefore excluded from every default run here.
# Modes {0,2} (`cmdbuf2`) are safe and gated; extending the round-robin to NoC1
# command buffers 3 (BRISC_AT_CMD_BUF) and/or 1 (BRISC_RD_CMD_BUF) is not.
# See the kernel head for the diagnosis. Set WIM_CORRECTNESS_MODES to re-arm it
# deliberately.
DEFAULT_CORRECTNESS_MODES = [
    "baseline",
    "rot_none",
    "rot_bankunif",
    "flush_half",
    "barrier_half",
    "cmdbuf2",
    "posted",
    "flush_only",
]


def test_all_modes_correctness(device):
    """Non-profiled gate: every mode x every sweep shape, bit identity only.
    Nothing in the menu is quotable until this passes."""
    import os

    modes = os.environ.get("WIM_CORRECTNESS_MODES", ",".join(DEFAULT_CORRECTNESS_MODES)).split(",")
    for mode_name in modes:
        for shape, label in SWEEP:
            torch_input, tt_input = _make_input(device, shape)
            _run_one(device, tt_input, torch_input, label, mode_name, "", check=True)


# --- paired / interleaved A-B, the statistically honest form -----------------
# Run-to-run drift inside a single profiled invocation is ~+-5% on this shape
# (rep 3 is reliably slower than rep 1), which is the same size as the effects
# in this menu. One-mode-per-invocation therefore cannot separate them. This
# test builds a DIFFERENT descriptor per mode inside ONE process and dispatches
# them ROUND-ROBIN, so every mode sees the same drift and the comparison is
# paired. `WRITE_ISSUE_MODE` is only a kernel define, so nothing prevents
# several modes coexisting in one run.
AB_MODES = ["baseline", "rot_none", "rot_bankunif", "flush_half", "barrier_half", "cmdbuf2", "posted", "flush_only"]


def test_interleaved(device):
    """Round-robin dispatch of every mode, `WIM_AB_REPS` times, first with the
    read+compute payloads ablated (isolated write stage) then whole-op.

    Row order in the profiler CSV is exactly:
        for phase in [WRITE-STAGE, WHOLE-OP]: for rep in range(R): for mode in AB_MODES
    which `read_perf.py --ab` decodes."""
    import os

    modes = os.environ.get("WIM_AB_MODES", ",".join(AB_MODES)).split(",")
    reps = int(os.environ.get("WIM_AB_REPS", "5"))
    shape = tuple(int(x) for x in os.environ.get("WIM_AB_SHAPE", "1,1,32,16384").split(","))
    torch_input, tt_input = _make_input(device, shape)
    print(f"\n[write_issue_menu INTERLEAVED] shape={shape} reps={reps} modes={modes}")
    for phase, stages, check in (("WRITE-STAGE", "reads,compute", False), ("WHOLE-OP", "", True)):
        for rep in range(reps):
            for mode_name in modes:
                _run_one(device, tt_input, torch_input, f"{phase}/rep{rep}", mode_name, stages, check=check)
