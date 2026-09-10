# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_state_reuse — isolated bake-off for the tilize reader's per-block
stick-read loop.

Reconstructs JUST `load_block`'s "reader_read_block" branch
(ttnn/ttnn/operations/tilize/kernels/tilize_reader.cpp, the
`!input_is_native && !pad_active && input_pages_per_row==1` path): each core
owns ONE block of `rows_per_block` consecutive stick reads of `row_bytes`
bytes each, from a fixed byte offset inside a ROW_MAJOR interleaved bf16
tensor. Five reader kernels contend for the same block:

    helper         -- dataflow_kernel_lib::read_sticks_for_tilize, unmodified
                       (the op's honest baseline)
    raw            -- the helper's own loop, split into reserve/issue/barrier
                       zones (isolates helper-call overhead alone)
    recurrence     -- address RECURRENCE (no per-stick division), plain reads
    state_naive    -- per-stick noc_async_read_one_packet_set_state/with_state,
                       NO bank grouping (expected null/regression)
    state_grouped  -- recurrence + BANK-GROUPED set_state/with_state (the real
                       candidate: idea 3+4 combined)

The writer is byte-identical across every variant (perf-lab concept
isolation): it drains the block and stores it as `rows_per_block` ordinary
interleaved pages, so any measured delta is attributable to the reader alone.

`(rows_per_block, row_bytes)` pairs are the EXACT per-block numbers
`tilize_program_descriptor.derive_plan()` produces for each domain shape
(probed once with a real device tensor; see DERIVED_BLOCKS below) — this bench
is a faithful reconstruction of the real per-core transaction pattern, not an
invented one. Every domain shape here happens to tile into exactly
`row_groups_used * w_chunks_used == 64` blocks (one per core on this 8x8 box),
so every core in this bench runs a DISTINCT (row_group, w_chunk) pair, exactly
as the real op would assign them.
"""

from __future__ import annotations

from pathlib import Path


import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_H = 32
TILE_W = 32
ELEM_SIZE = 2  # bfloat16
TILE_PAGE_BYTES = TILE_H * TILE_W * ELEM_SIZE  # 2048

READER_KERNELS = {
    "helper": "reader_helper.cpp",
    "raw": "reader_raw.cpp",
    "recurrence": "reader_recurrence.cpp",
    "state_naive": "reader_state_naive.cpp",
    "state_grouped": "reader_state_grouped.cpp",
}
VARIANTS = tuple(READER_KERNELS)

# (rows_per_block, row_bytes, row_groups_used, w_chunks_used), probed via
# `pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)` on an 8x8
# (64-core) Wormhole B0 box for each domain shape. `wide_short` truncates to
# 64 of its 128 real w_chunks (one bench block per core; the other 64 are the
# same per-block operation repeated -- see the domain note in the report).
DERIVED_BLOCKS = {
    "focus_1x1x32x16384": (32, 512, 1, 64),  # LOOSE_CASES[0], attention perf focus
    "wide_short_1x1x32x32768": (32, 512, 1, 64),  # truncated from num_w_chunks=128
    "square_1x1x1024x1024": (64, 512, 16, 4),
    "tall_narrow_1x1x16384x32": (256, 64, 64, 1),
    "small_1x1x32x2048": (32, 64, 1, 64),
}


def _full_grid(device):
    grid = device.compute_with_storage_grid_size()
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return grid, core_ranges


def run_variant(device, variant, *, rows_per_block, row_bytes, row_groups_used, w_chunks_used, seed=11):
    """Runs ONE reader variant over a synthetic block grid sized exactly
    `row_groups_used x w_chunks_used` (<= the device's core count; every shape
    in DERIVED_BLOCKS above uses exactly 64/64). Returns (got, expected) torch
    tensors for a bit-identity check -- tilize's own correctness contract."""
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    assert variant in READER_KERNELS, f"unknown variant {variant!r}"
    grid, _ = _full_grid(device)
    num_cores = grid.x * grid.y
    num_blocks = row_groups_used * w_chunks_used
    assert num_blocks <= num_cores, f"{num_blocks} blocks > {num_cores} cores -- one block per core only"
    assert row_bytes % (TILE_W * ELEM_SIZE) == 0, "row_bytes must be a whole number of tile-columns"

    row_bytes_elems = row_bytes // ELEM_SIZE
    rows_total = row_groups_used * rows_per_block
    cols_total = w_chunks_used * row_bytes_elems
    bw = row_bytes // (TILE_W * ELEM_SIZE)  # block_width_tiles

    torch.manual_seed(seed)
    torch_input = torch.randn((rows_total, cols_total), dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output: one page (row_bytes) per stick; core c's block occupies pages
    # [c*rows_per_block, (c+1)*rows_per_block).
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_blocks * rows_per_block, row_bytes_elems]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    cores = []
    for c in range(num_blocks):
        cores.append(ttnn.CoreCoord(c % grid.x, c // grid.x))
    used_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    tile_desc = ttnn.TileDescriptor(TILE_H, TILE_W)
    cb = ttnn.CBDescriptor(
        total_size=bw * TILE_PAGE_BYTES,
        core_ranges=used_core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0, data_format=ttnn.bfloat16, page_size=TILE_PAGE_BYTES, tile=tile_desc
            )
        ],
    )

    reader_ct = [rows_per_block, row_bytes, bw] + list(ttnn.TensorAccessorArgs(tt_input).get_compile_time_args())
    writer_ct = [rows_per_block, row_bytes, bw] + list(ttnn.TensorAccessorArgs(tt_output).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    in_addr = tt_input.buffer_address()
    out_addr = tt_output.buffer_address()

    c = 0
    for rg in range(row_groups_used):
        for wc in range(w_chunks_used):
            core = cores[c]
            start_page = rg * rows_per_block
            byte_offset = wc * row_bytes
            reader_rt[core.x][core.y] = [in_addr, start_page, byte_offset]
            writer_rt[core.x][core.y] = [out_addr, c * rows_per_block, 0]
            c += 1

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / READER_KERNELS[variant]),
        core_ranges=used_core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_block.cpp"),
        core_ranges=used_core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])
    result = ttnn.generic_op([tt_input, tt_output], program)

    got = ttnn.to_torch(result)[: num_blocks * rows_per_block]
    expected_blocks = []
    for rg in range(row_groups_used):
        for wc in range(w_chunks_used):
            expected_blocks.append(
                torch_input[
                    rg * rows_per_block : (rg + 1) * rows_per_block, wc * row_bytes_elems : (wc + 1) * row_bytes_elems
                ]
            )
    expected = torch.cat(expected_blocks, dim=0)
    return got, expected
