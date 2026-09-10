# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""writer_dual_risc — isolated bake-off for the tilize WRITE stage.

Idea under test: split the output tile store across BOTH data-movement
RISC-Vs instead of leaving BRISC to carry the whole `store_block` alone. See
this dir's report for the measured verdict.

GEOMETRY. The block grid (block_width_tiles, num_w_chunks, num_row_groups,
write_rows_per_barrier, per-core assignment, ...) is not re-derived here: it
is READ (never mutated) from the real op's own `derive_plan`, so every
variant in this bench runs on EXACTLY the op's own block plan for a given
shape. Only the WRITER (and, for one variant, which RISC-V/NoC a kernel is
bound to) changes across variants — reader and compute are byte-identical
plain reconstructions (`reader_plain.cpp` / `compute_plain.cpp`) shared by
every variant except `col_split`, whose own reader/compute are the mirror
construction described in their own file headers.

VARIANTS
  baseline   — current op behavior: BRISC alone drains cb_output_tiles.
  col_split  — candidate A: compute packs the block's LEFT half-width tile
               columns into cb_output_tiles (BRISC drains) and the RIGHT half
               into cb_output_tiles_split (NCRISC drains, after its own
               reads). Requires block_width_tiles even and >= 2.
  role_swap  — candidate C: NCRISC runs the (full) writer, BRISC runs the
               (full) reader — a pure NoC-index swap, no split at all. Tests
               how much of any split's win is "second RISC-V" vs "NoC choice".
  decisive   — the write-only, no-reader-no-compute ablation used for the
               issue-bound-vs-bandwidth-bound check. See
               `writeonly_decisive.cpp` and `build_decisive_descriptor`.
"""

from pathlib import Path

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as tilize_pd

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INPUT_ROWS = 0
CB_OUTPUT_TILES = 1
CB_OUTPUT_TILES_SPLIT = 2

VARIANTS = ("baseline", "col_split", "role_swap")


def derive_geometry(input_tensor, output_tensor, device):
    """The op's OWN block plan for this (shape, dtype, memory_config) — read
    only, never mutated. This is what makes every variant's reader/compute
    identical to the real op's current geometry; only the writer differs."""
    grid = device.compute_with_storage_grid_size()
    plan = tilize_pd.derive_plan(input_tensor, output_tensor, low_l1=False, grid=grid)
    if plan.tail_group is not None:
        raise ValueError("writer_dual_risc bench: ragged column tail shapes are out of scope for this bench")
    if plan.input_native or plan.output_native or plan.pad_active or plan.is_retile:
        raise ValueError("writer_dual_risc bench: only the plain interleaved->interleaved path is in scope")
    return plan


def col_split_applicable(plan) -> bool:
    return plan.block_width_tiles >= 2 and plan.block_width_tiles % 2 == 0


def _compute_config(plan):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=bool(plan.fp32_dest_acc_en),
        dst_full_sync_en=False,
    )


def _cb(index, dtype, page_bytes, pages, cores, tile_desc):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes, tile=tile_desc)
        ],
    )


def build_descriptor(input_tensor, output_tensor, plan, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"writer_dual_risc bench: unknown variant {variant!r}")

    device = input_tensor.device()
    tile_desc = ttnn.TileDescriptor(plan.tile_h, 32)
    cores = plan.all_cores
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    in_accessor_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    compute_config = _compute_config(plan)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core, start_block_id, num_blocks, block_stride in plan.assignment:
        reader_rt[core.x][core.y] = [in_addr, start_block_id, num_blocks, block_stride]
        writer_rt[core.x][core.y] = [out_addr, start_block_id, num_blocks, block_stride]
        compute_rt[core.x][core.y] = [start_block_id, num_blocks, block_stride]

    if variant in ("baseline", "role_swap"):
        cbs = [
            _cb(
                CB_INPUT_ROWS,
                input_tensor.dtype,
                plan.in_page_bytes,
                plan.input_depth_rows * plan.block_width_tiles,
                cores,
                tile_desc,
            ),
            _cb(
                CB_OUTPUT_TILES,
                output_tensor.dtype,
                plan.out_page_bytes,
                plan.output_depth_batches * plan.write_rows_per_barrier * plan.block_width_tiles,
                cores,
                tile_desc,
            ),
        ]

        reader_ct = [
            CB_INPUT_ROWS,
            plan.block_width_tiles,
            plan.tile_h,
            plan.tensor_row_blocks,
            plan.num_row_groups,
            plan.num_w_chunks,
            plan.block_row_bytes,
        ] + in_accessor_args
        writer_ct = [
            CB_OUTPUT_TILES,
            plan.block_width_tiles,
            plan.tensor_row_blocks,
            plan.tensor_col_tiles,
            plan.num_row_groups,
            plan.num_w_chunks,
            plan.write_rows_per_barrier,
            plan.out_page_bytes,
        ] + out_accessor_args
        compute_ct = [
            CB_INPUT_ROWS,
            CB_OUTPUT_TILES,
            plan.block_width_tiles,
            plan.tensor_row_blocks,
            plan.num_row_groups,
            plan.num_w_chunks,
        ]

        reader_config = ttnn.WriterConfigDescriptor() if variant == "role_swap" else ttnn.ReaderConfigDescriptor()
        writer_config = ttnn.ReaderConfigDescriptor() if variant == "role_swap" else ttnn.WriterConfigDescriptor()

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_plain.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=reader_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_plain.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=writer_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_plain.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_config,
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    # --- col_split ----------------------------------------------------------
    if not col_split_applicable(plan):
        raise ValueError(
            f"writer_dual_risc bench: col_split is inexpressible at block_width_tiles={plan.block_width_tiles} "
            "(must be even and >= 2)"
        )
    half_width = plan.block_width_tiles // 2

    cbs = [
        _cb(
            CB_INPUT_ROWS, input_tensor.dtype, plan.in_page_bytes, plan.input_depth_rows * half_width, cores, tile_desc
        ),
        _cb(
            CB_OUTPUT_TILES,
            output_tensor.dtype,
            plan.out_page_bytes,
            plan.output_depth_batches * plan.write_rows_per_barrier * half_width,
            cores,
            tile_desc,
        ),
        _cb(
            CB_OUTPUT_TILES_SPLIT,
            output_tensor.dtype,
            plan.out_page_bytes,
            plan.output_depth_batches * plan.write_rows_per_barrier * half_width,
            cores,
            tile_desc,
        ),
    ]

    reader_ct = [
        CB_INPUT_ROWS,
        CB_OUTPUT_TILES_SPLIT,
        plan.block_width_tiles,
        plan.tile_h,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.tensor_col_tiles,
        plan.write_rows_per_barrier,
        plan.out_page_bytes,
        plan.block_row_bytes,
    ]
    reader_ct.extend(in_accessor_args)
    reader_ct.extend(out_accessor_args)

    writer_ct = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.tensor_col_tiles,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.write_rows_per_barrier,
        plan.out_page_bytes,
    ] + out_accessor_args

    compute_ct = [
        CB_INPUT_ROWS,
        CB_OUTPUT_TILES,
        CB_OUTPUT_TILES_SPLIT,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
    ]

    reader_rt = ttnn.RuntimeArgs()
    for core, start_block_id, num_blocks, block_stride in plan.assignment:
        reader_rt[core.x][core.y] = [in_addr, out_addr, start_block_id, num_blocks, block_stride]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader_col_split.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer_col_split.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "compute_col_split.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


# ---------------------------------------------------------------------------
# The decisive check: write-only, no reader, no compute.
# ---------------------------------------------------------------------------

BLOCK_WIDTH_TILES_DECISIVE = 8  # matches the focus shape's own block width
TILE = 32
ELEM_BYTES = 2  # bfloat16
OUT_TILE_BYTES = TILE * TILE * ELEM_BYTES


def _cores_for(device, num_cores):
    """The device's own row-wise work-split utility, asked for exactly
    `num_cores` single-block workers — NOT a hand-rolled single-row range,
    which overruns the grid width the moment `num_cores > grid.x` (an 8x8 grid
    cannot hold a 1x16 CoreRange). This wraps to a second row exactly the way
    the real op's own core assignment does."""
    grid = device.compute_with_storage_grid_size()
    _used, all_cores, _g1, _g2, _n1, _n2 = ttnn.split_work_to_cores(grid, num_cores, row_wise=True)
    return all_cores


def build_decisive_descriptor(output_tensor, num_active_cores: int):
    """One core per tile-column-group of `BLOCK_WIDTH_TILES_DECISIVE` tiles,
    each writing ONLY its own 8 tiles — no reader, no compute. `output_tensor`
    must be shaped `[32, num_active_cores * BLOCK_WIDTH_TILES_DECISIVE * 32]`."""
    device = output_tensor.device()
    cores = _cores_for(device, num_active_cores)
    out_addr = output_tensor.buffer_address()
    out_accessor_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    cb = ttnn.CBDescriptor(
        total_size=BLOCK_WIDTH_TILES_DECISIVE * OUT_TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_OUTPUT_TILES,
                data_format=output_tensor.dtype,
                page_size=OUT_TILE_BYTES,
                tile=ttnn.TileDescriptor(TILE, TILE),
            )
        ],
    )

    writer_ct = [CB_OUTPUT_TILES, BLOCK_WIDTH_TILES_DECISIVE, OUT_TILE_BYTES] + out_accessor_args
    writer_rt = ttnn.RuntimeArgs()
    for core_index, core in enumerate(ttnn.corerange_to_cores(cores, num_active_cores, True)):
        col_base = core_index * BLOCK_WIDTH_TILES_DECISIVE
        writer_rt[core.x][core.y] = [out_addr, col_base]

    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writeonly_decisive.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
