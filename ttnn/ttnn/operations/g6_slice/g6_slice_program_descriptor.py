# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_slice ProgramDescriptor (op_design.md -> Blocking Model, Work Distribution, Circular Buffers).

One program, two data-movement kernels, no compute kernel (pure raw-byte copy; a compute copy
would pass values through unpack/pack formats -- design regime R13, rejected):

  reader (NCRISC, NoC0): load_block -- the block's input pages (R1) or source tiles (R2) from x
                         assemble_block (R2 only) -- face-row runs from cb_source_tiles
  writer (BRISC,  NoC1): store_block -- the block's output pages to out

Work split: the four output tile axes (lead_0, lead_1, out_tile_row, out_tile_col) of the
rank-4-normalized output fold into one flattened output-tile axis, split contiguously over the
grid with split_work_to_cores(row_wise=True). Each core walks its range in blocks of
`block_tiles` consecutive output tiles.

Knobs (single source of truth, each defined once here):
  CB_OUTPUT_DEPTH      buffer depth of cb_output_tiles (blocks)
  CB_SOURCE_DEPTH      buffer depth of cb_source_tiles (blocks, R2 only)
  L1_BUDGET_BYTES      per-core L1 the op may take (BB-05)
  BLOCK_TILES_OVERRIDE None -> block_tiles = min(core_tiles_max, block_tiles_l1_max)
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# --- knobs -------------------------------------------------------------------------------------
CB_OUTPUT_DEPTH = 2
CB_SOURCE_DEPTH = 1
L1_BUDGET_BYTES = 1347776
BLOCK_TILES_OVERRIDE = None

# --- CB slots (semantic names) -----------------------------------------------------------------
CB_OUTPUT_TILES = 0
CB_SOURCE_TILES = 1


def _to_rank4(values, fill):
    values = tuple(int(v) for v in values)
    return (fill,) * (4 - len(values)) + values


def build_plan(input_tensor, begins, out_shape):
    """Host parameter derivation (no tensor op): geometry, regime, common RT args.

    Cheap; runs every call. Everything that may differ between two calls with the same
    `signature` is in the common runtime args (patched on a cache hit)."""
    b = _to_rank4(begins, 0)
    m = _to_rank4(out_shape, 1)
    n = _to_rank4(input_tensor.shape, 1)
    padded = _to_rank4(input_tensor.padded_shape, 1)

    ht_in = padded[2] // TILE
    wt_in = padded[3] // TILE
    out_rows = -(-m[2] // TILE)
    out_cols = -(-m[3] // TILE)
    total_out_tiles = m[0] * m[1] * out_rows * out_cols

    q0, o_h = divmod(b[2], TILE)
    c0, o_w = divmod(b[3], TILE)
    shift_rows = o_h != 0
    shift_cols = o_w != 0
    shifted = shift_rows or shift_cols
    src_per_out_cap = (1 + int(shift_rows)) * (1 + int(shift_cols))

    reader_common_tail = [m[1], out_rows, out_cols, n[1], ht_in, wt_in, b[0], b[1], q0, c0, o_h, o_w, m[2], m[3]]
    regime = "R2_shifted_gather" if shifted else "R1_aligned_copy"
    return {
        "total_out_tiles": total_out_tiles,
        "out_rows": out_rows,
        "out_cols": out_cols,
        "m1": m[1],
        "shift_rows": shift_rows,
        "shift_cols": shift_cols,
        "src_per_out_cap": src_per_out_cap,
        "src_per_out_staging": src_per_out_cap if shifted else 0,
        "reader_common_tail": reader_common_tail,
        "signature": (shift_rows, shift_cols),
        "report": {
            "regime": regime,
            "SHIFT_ROWS": shift_rows,
            "SHIFT_COLS": shift_cols,
            "total_out_tiles": total_out_tiles,
        },
    }


def _split(device, total_out_tiles):
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, tiles_g1, tiles_g2) = ttnn.split_work_to_cores(
        grid, total_out_tiles, row_wise=True
    )
    assignment = []  # (core, start_out_tile, core_tiles)
    start = 0
    for group, per_core in ((group_1, tiles_g1), (group_2, tiles_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    assert start == total_out_tiles, (start, total_out_tiles)
    return all_cores, assignment, max(tiles_g1, tiles_g2)


def _start_coords(start, plan):
    out_cols, out_rows, m1 = plan["out_cols"], plan["out_rows"], plan["m1"]
    c = start % out_cols
    t = start // out_cols
    r = t % out_rows
    t //= out_rows
    i1 = t % m1
    i0 = t // m1
    return [i0, i1, r, c]


def create_program_descriptor(input_tensor, output_tensor, plan):
    device = input_tensor.device()
    page_size = input_tensor.buffer_aligned_page_size()
    assert page_size == output_tensor.buffer_aligned_page_size()
    total_out_tiles = plan["total_out_tiles"]
    assert total_out_tiles == output_tensor.buffer_num_pages(), (total_out_tiles, output_tensor.buffer_num_pages())

    all_cores, assignment, core_tiles_max = _split(device, total_out_tiles)

    # --- block knob: the coarsest block that fits L1, capped at the core's assignment --------
    staging = CB_SOURCE_DEPTH * plan["src_per_out_staging"]
    block_tiles_l1_max = L1_BUDGET_BYTES // ((CB_OUTPUT_DEPTH + staging) * page_size)
    if block_tiles_l1_max < 1:
        raise RuntimeError("g6_slice: one block does not fit the L1 budget (BB-05)")
    block_tiles = BLOCK_TILES_OVERRIDE if BLOCK_TILES_OVERRIDE is not None else core_tiles_max
    block_tiles = max(1, min(block_tiles, core_tiles_max, block_tiles_l1_max))

    # --- circular buffers ----------------------------------------------------------------------
    cbs = [
        ttnn.CBDescriptor(
            total_size=CB_OUTPUT_DEPTH * block_tiles * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES, data_format=input_tensor.dtype, page_size=page_size
                )
            ],
        )
    ]
    if staging:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=staging * block_tiles * page_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_SOURCE_TILES, data_format=input_tensor.dtype, page_size=page_size
                    )
                ],
            )
        )

    # --- kernels -------------------------------------------------------------------------------
    defines = []
    if plan["shift_rows"]:
        defines.append(("SHIFT_ROWS", "1"))
    if plan["shift_cols"]:
        defines.append(("SHIFT_COLS", "1"))

    reader_ct = [
        CB_OUTPUT_TILES,
        CB_SOURCE_TILES,
        block_tiles,
        page_size,
        input_tensor.element_size(),
        plan["src_per_out_cap"],
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [CB_OUTPUT_TILES, block_tiles, page_size]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for core, start, core_tiles in assignment:
        reader_rt[core.x][core.y] = [core_tiles] + _start_coords(start, plan)
        writer_rt[core.x][core.y] = [start, core_tiles]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        defines=defines,
        runtime_args=reader_rt,
        common_runtime_args=[0] + plan["reader_common_tail"],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_slice_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[0],
        config=ttnn.WriterConfigDescriptor(),
    )

    report = {
        "num_cores": len(assignment),
        "core_tiles_max": core_tiles_max,
        "block_tiles": block_tiles,
        "block_tiles_l1_max": block_tiles_l1_max,
        "cb_output_depth": CB_OUTPUT_DEPTH,
        "cb_source_depth": CB_SOURCE_DEPTH if staging else 0,
        "l1_bytes_per_core": (CB_OUTPUT_DEPTH + staging) * block_tiles * page_size,
        "roles": [(c.x, c.y, "reader+writer", s, t) for c, s, t in assignment],
    }
    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    return descriptor, report


READER_KERNEL_INDEX = 0
WRITER_KERNEL_INDEX = 1


def patch_common_runtime_args(descriptor, input_tensor, output_tensor, plan):
    """Per-call values: buffer addresses + begin-derived geometry (cache-hit refresh path)."""
    descriptor.kernels[READER_KERNEL_INDEX].common_runtime_args = [input_tensor.buffer_address()] + plan[
        "reader_common_tail"
    ]
    descriptor.kernels[WRITER_KERNEL_INDEX].common_runtime_args = [output_tensor.buffer_address()]
