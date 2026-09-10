# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness — idea `writer_bank_rotate`.

PERF EXPERIMENT ONLY. Does not touch the real op. See `bench.py` for how the
real op's own reader + compute kernels are reused unmodified and only the
writer kernel is swapped for `kernels/tilize_writer_bankrotate.cpp`.

Correctness run (bit identity, tilize is a pure re-lay):
    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/writer_bank_rotate/test_writer_bank_rotate.py

Perf run, ONE rotate mode per invocation (mirrors the op's own
TILIZE_ABLATE env-var-to-defines convention):
    for M in baseline rotate_chunk rotate_blockid rotate_bankexact rotate_rows_cols; do
        WRITER_ROTATE_MODE_NAME=$M scripts/run_safe_pytest.sh --profile \
            ttnn/ttnn/operations/tilize/perf_experiments/writer_bank_rotate/test_writer_bank_rotate.py
    done
Then read `DEVICE KERNEL DURATION [ns]` (whole-op) and
`DEVICE BRISC KERNEL DURATION [ns]` per row from the newest
`generated/profiler/reports/*/ops_perf_results*.csv`, and the per-core BRISC
KERNEL span mean/max from `perf_experiments/zone_report.py` (reads
`generated/profiler/.logs/profile_log_device.csv`, refreshed by the same run).
"""


import ttnn

from ttnn.operations.tilize.perf_experiments.writer_bank_rotate.bench import (
    current_mode_from_env,
    run_variant_from_tt,
)

# focus shape first (mandatory), then the domain sweep the task brief names.
CASES = [
    ((1, 1, 32, 16384), "attention_focus"),  # bw=8, 8k mod 12 in {0,4,8} -- predicted worst case
    ((1, 1, 32, 32768), "short_wide_wide"),  # bw=8 still (C=1024 -> num_w_chunks=128, 2 blocks/core)
    ((1, 1, 1024, 1024), "square_mid"),  # bw=8, R=32
    ((1, 1, 16384, 32), "tall_narrow"),  # bw=1 -- no column rotation possible, split-reader active
    ((1, 1, 32, 2048), "small"),  # bw=8, num_w_chunks=8, 8 cores used
]


def _run_one(device, shape, label, mode_name):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output, plan = run_variant_from_tt(tt_input, mode_name)
    print(
        f"\n[writer_bank_rotate mode={mode_name} {label}] {shape}: "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} "
        f"rows/blk={plan.tensor_row_blocks // plan.num_row_groups} "
        f"blocks={plan.num_blocks_total} cores={len(plan.assignment)}/{grid.x * grid.y} "
        f"split_reader={plan.groups[0].split_reader if plan.groups else 0}"
    )
    got = ttnn.to_torch(tt_output)
    assert torch.equal(got, torch_input), f"{mode_name}/{label}: writer_bank_rotate broke bit identity"
    return plan


def test_writer_bank_rotate_focus(device):
    """The mandatory focus shape, mode picked via WRITER_ROTATE_MODE_NAME env
    (default: baseline). One CSV row per --profile invocation."""
    shape, label = CASES[0]
    mode_name = current_mode_from_env()
    plan = _run_one(device, shape, label, mode_name)
    assert plan.tensor_row_blocks == 1 and plan.tensor_col_tiles == 512
    assert len(plan.assignment) == 64, "the flagged number is only meaningful at 64/64 cores"


def test_writer_bank_rotate_domain_sweep(device):
    """The rest of the domain sweep, same mode, same invocation's CSV (rows
    2..6, in CASES[1:] order)."""
    mode_name = current_mode_from_env()
    for shape, label in CASES[1:]:
        _run_one(device, shape, label, mode_name)


def test_writer_bank_rotate_all_modes_correctness(device):
    """Non-profiled correctness gate: every mode x every shape, bit-identity
    only. Run this WITHOUT --profile before trusting any perf number above."""
    from ttnn.operations.tilize.perf_experiments.writer_bank_rotate.bench import MODES

    for mode_name in MODES:
        for shape, label in CASES:
            _run_one(device, shape, label, mode_name)
