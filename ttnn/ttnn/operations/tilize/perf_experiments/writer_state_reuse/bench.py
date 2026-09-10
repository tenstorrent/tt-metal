# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""writer_state_reuse — isolated bake-off for the tilize writer's per-page
NoC-write ISSUE cost (`noc_async_write_one_packet_set_state`/`_with_state` vs
today's per-page `noc_async_write<out_tile_bytes>`).

Reconstructs ONLY the writer (+ a trivial, held-constant reader that hands it
real tile bytes with no compute stage at all — a pure TILE-to-TILE copy, the
same shape the real op's `is_retile` path already takes). Never touches the
real op's kernels or program descriptor.

Variants (see kernels/writer_bench.cpp for the full mechanism writeup):
  0 baseline          today's op, unchanged
  1 set_state_naive   the idea AS STATED — set_state once/row, reuse for the
                       rest. INCORRECT for an interleaved DRAM destination
                       (wrong bank past the first page); kept to make that
                       concrete and measured, marked xfail(strict=True).
  2 addr_recurrence   baseline's write command, incremental bank/offset
                       stepping instead of TensorAccessor's per-page div/mod.
  3 coord_reuse_raw   raw-LLK: CTRL+LEN programmed once for the kernel's
                       whole run, coordinate reprogrammed by hand every call.
  4 combined          2 + 3.

Run for numbers (device kernel ns lands in
`generated/profiler/reports/*/ops_perf_results*.csv`, `DEVICE KERNEL DURATION
[ns]`; per-zone `writer_issue`/`writer_wait_out`/`writer_barrier` means in
`generated/profiler/.logs/profile_log_device.csv`, read with
`ttnn/ttnn/operations/tilize/perf_experiments/zone_report.py`):

    scripts/run_safe_pytest.sh --profile --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/writer_state_reuse/bench.py

`--run-all` matters here: variant 1's correctness check is an EXPECTED
xfail, and `run_safe_pytest.sh` otherwise appends `-x` (stop on first
failure), which would cut the sweep off before the later variants run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
CB_OUT = 0
TILE_WIDTH = 32

VARIANT_BASELINE = 0
VARIANT_SET_STATE_NAIVE = 1
VARIANT_ADDR_RECURRENCE = 2
VARIANT_COORD_REUSE_RAW = 3
VARIANT_COMBINED = 4
VARIANT_NAMES = {
    VARIANT_BASELINE: "baseline",
    VARIANT_SET_STATE_NAIVE: "set_state_naive",
    VARIANT_ADDR_RECURRENCE: "addr_recurrence",
    VARIANT_COORD_REUSE_RAW: "coord_reuse_raw",
    VARIANT_COMBINED: "combined",
}

# Focus shape (mandatory) + domain sweep, all ROW x COL in ELEMENTS (both
# tensors are TILE layout here — this bench has no ROW_MAJOR side at all).
FOCUS_SHAPE = (32, 16384)  # attention: R=1, C=512 tile-cols, bw=8 -- the plan in tilize_perf_context.md
SWEEP_SHAPES = {
    "wide_short": (32, 32768),
    "square": (1024, 1024),
    "tall_narrow": (16384, 32),
    "small": (32, 2048),
}
ALL_SHAPES = {"attention_focus": FOCUS_SHAPE, **SWEEP_SHAPES}


class _Group:
    """Minimal stand-in for tilize_program_descriptor.ColumnGroup — just the
    fields _make_program_descriptor reads. Lets this bench derive its own
    block plan instead of importing the real op's `derive_plan` (no coupling
    to its grid-filling heuristics, and no risk of ever importing something
    that then gets edited under us). `num_row_groups` is carried here (unlike
    the real ColumnGroup, which reads it off the parent TilizePlan) since this
    bench has no plan object above the group."""

    def __init__(
        self,
        block_width_tiles,
        num_w_chunks,
        num_row_groups,
        col_tile_offset,
        write_rows_per_barrier,
        cores,
        assignment,
    ):
        self.block_width_tiles = block_width_tiles
        self.num_w_chunks = num_w_chunks
        self.num_row_groups = num_row_groups
        self.col_tile_offset = col_tile_offset
        self.write_rows_per_barrier = write_rows_per_barrier
        self.cores = cores
        self.assignment = assignment


def _plan_full_grid(rows_elems: int, cols_elems: int, grid):
    """Occupancy-filled like the real op's Rule 2 step 1: cut the WIDTH first
    (block_width_tiles so num_w_chunks == min(C, num_cores)), then cut the
    remaining ROWS across whatever core budget is left
    (num_row_groups == min(R, num_cores // num_w_chunks)) so a C == 1 or
    C-small shape (tall_narrow, small) still spreads across the grid instead
    of collapsing onto one core. A ragged tail is not needed for any of this
    bench's 5 shapes (every C here divides evenly — asserted below)."""
    tile_h = TILE_WIDTH
    R = -(-rows_elems // tile_h)  # tensor_row_blocks
    C = -(-cols_elems // TILE_WIDTH)  # tensor_col_tiles
    num_cores = int(grid.x) * int(grid.y)
    target_chunks = min(C, num_cores)
    block_width_tiles = max(1, -(-C // target_chunks)) if target_chunks else C
    assert C % block_width_tiles == 0, f"C={C} bw={block_width_tiles}: this bench assumes a smooth divisor (no tail)"
    num_w_chunks = C // block_width_tiles
    num_row_groups = max(1, min(R, num_cores // max(1, num_w_chunks)))
    num_blocks_total = num_row_groups * num_w_chunks
    cores = []
    for y in range(int(grid.y)):
        for x in range(int(grid.x)):
            cores.append(ttnn.CoreCoord(x, y))
    used = min(len(cores), num_blocks_total)
    base, rem = divmod(num_blocks_total, used)
    assignment = []
    start = 0
    for i in range(used):
        per_core = base + (1 if i < rem else 0)
        assignment.append((cores[i], start, per_core, 1))
        start += per_core
    core_range = (
        ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[used - 1])})
        if used > 1
        else (ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[0])}) if used == 1 else ttnn.CoreRangeSet({}))
    )
    write_rows_per_barrier = 1  # WRITE_BATCH_MIN_TILES==4 is inert once bw>=4; every bw here is >=8 except tall_narrow
    if block_width_tiles < 4:
        write_rows_per_barrier = max(1, -(-4 // block_width_tiles))
    group = _Group(block_width_tiles, num_w_chunks, num_row_groups, 0, write_rows_per_barrier, core_range, assignment)
    return group, R, C


def _plan_single_core(block_width_tiles: int):
    """Exactly ONE core, one block of `block_width_tiles` tiles — the decisive
    fabric-vs-RISC probe's "few cores active" leg. Tensor is sized to match
    (1 tile row x block_width_tiles tile cols)."""
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    assignment = [(core, 0, 1, 1)]
    group = _Group(block_width_tiles, 1, 1, 0, 1, core_range, assignment)
    return group, 1, block_width_tiles


def _make_program_descriptor(in_tensor, out_tensor, group, R, C, variant, ablate_reads):
    tile_desc = ttnn.TileDescriptor(TILE_WIDTH, TILE_WIDTH)
    out_page_bytes = out_tensor.buffer_page_size()

    out_accessor_args = ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args()
    in_accessor_args = ttnn.TensorAccessorArgs(in_tensor).get_compile_time_args()
    in_addr = in_tensor.buffer_address()
    out_addr = out_tensor.buffer_address()

    output_cb_pages = 2 * group.write_rows_per_barrier * group.block_width_tiles  # depth 2, matches the real op
    cbs = [
        ttnn.CBDescriptor(
            total_size=output_cb_pages * out_page_bytes,
            core_ranges=group.cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT,
                    data_format=out_tensor.dtype,
                    page_size=out_page_bytes,
                    tile=tile_desc,
                )
            ],
        )
    ]

    reader_ct_args = [
        CB_OUT,
        group.block_width_tiles,
        R,
        C,
        group.num_row_groups,
        group.num_w_chunks,
        group.col_tile_offset,
        out_page_bytes,
    ]
    reader_ct_args.extend(in_accessor_args)

    writer_ct_args = [
        CB_OUT,
        group.block_width_tiles,
        R,
        C,
        group.num_row_groups,
        group.num_w_chunks,
        group.write_rows_per_barrier,
        out_page_bytes,
        group.col_tile_offset,
        variant,
    ]
    writer_ct_args.extend(out_accessor_args)

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, start_block_id, num_blocks, block_stride in group.assignment:
        reader_rt_args[core.x][core.y] = [in_addr, start_block_id, num_blocks, block_stride]
        writer_rt_args[core.x][core.y] = [out_addr, start_block_id, num_blocks, block_stride]

    reader_defines = [("BENCH_ABLATE_READS", "1")] if ablate_reads else []

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader_bench.cpp"),
            core_ranges=group.cores,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt_args,
            defines=reader_defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_bench.cpp"),
            core_ranges=group.cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            defines=[],
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _run_copy(device, rows_elems, cols_elems, variant, *, single_core_width=None, ablate_reads=False):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_data = torch.randn((rows_elems, cols_elems), dtype=torch.float32).bfloat16()

    in_tensor = ttnn.from_torch(
        torch_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tensor = ttnn.from_torch(
        torch.zeros((rows_elems, cols_elems), dtype=torch.float32).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if single_core_width is not None:
        group, R, C = _plan_single_core(single_core_width)
    else:
        group, R, C = _plan_full_grid(rows_elems, cols_elems, grid)

    program_descriptor = _make_program_descriptor(in_tensor, out_tensor, group, R, C, variant, ablate_reads)
    ttnn.generic_op([in_tensor, out_tensor], program_descriptor)
    ttnn.synchronize_device(device)
    print(
        f"\n[{VARIANT_NAMES[variant]}] shape=({rows_elems},{cols_elems}) bw={group.block_width_tiles} "
        f"chunks={group.num_w_chunks} wrpb={group.write_rows_per_barrier} cores={len(group.assignment)}"
    )
    return torch_data, out_tensor


# ---------------------------------------------------------------------------
# Menu / domain sweep: correctness + device kernel ns, variants x shapes.
# ---------------------------------------------------------------------------
CORRECT_VARIANTS = [VARIANT_BASELINE, VARIANT_ADDR_RECURRENCE, VARIANT_COORD_REUSE_RAW, VARIANT_COMBINED]


@pytest.mark.parametrize("shape_name", list(ALL_SHAPES.keys()))
@pytest.mark.parametrize("variant", CORRECT_VARIANTS, ids=[VARIANT_NAMES[v] for v in CORRECT_VARIANTS])
def test_writer_variant(device, variant, shape_name):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    rows, cols = ALL_SHAPES[shape_name]
    torch_data, out_tensor = _run_copy(device, rows, cols, variant)
    assert torch.equal(
        ttnn.to_torch(out_tensor), torch_data
    ), f"variant={VARIANT_NAMES[variant]} shape={shape_name}: writer produced non-bit-identical output"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "set_state_naive is the idea AS LITERALLY STATED (set_state once/row, "
        "reuse for the rest) and is INCORRECT here: TensorAccessor round-robins "
        "DRAM banks by page_id % num_banks, so every page after the row's first "
        "lands on a DIFFERENT bank than the one set_state programmed, and "
        "with_state never reprograms the coordinate register. This xfail IS the "
        "measured finding, not a bug in the bench."
    ),
)
def test_writer_variant_naive_is_incorrect(device):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    torch_data, out_tensor = _run_copy(device, *FOCUS_SHAPE, VARIANT_SET_STATE_NAIVE)
    assert torch.equal(ttnn.to_torch(out_tensor), torch_data)


# ---------------------------------------------------------------------------
# Decisive fabric-vs-RISC test (run this one FIRST when reading results).
# Same per-core work (one block of 8 tiles, i.e. FOCUS_SHAPE's own per-core
# slice), reads ablated so the writer is the only payload, at two very
# different amounts of NoC/DRAM contention: 64 cores all writing at once vs
# exactly 1. If per-call issue cost collapses at 1 core, the 546 ns/call the
# coordinator measured is fabric back-pressure, not RISC issue cost, and
# every variant below is a NULL by construction (state reuse cannot fix
# contention it did not cause). Read `writer_issue` in
# generated/profiler/.logs/profile_log_device.csv via zone_report.py.
# ---------------------------------------------------------------------------
def test_fabric_backpressure_full_grid(device):
    _run_copy(device, *FOCUS_SHAPE, VARIANT_BASELINE, ablate_reads=True)


def test_fabric_backpressure_single_core(device):
    _run_copy(device, 32, 256, VARIANT_BASELINE, single_core_width=8, ablate_reads=True)
