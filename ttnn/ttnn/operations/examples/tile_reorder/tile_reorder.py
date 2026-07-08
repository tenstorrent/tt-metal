# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement example: relocate whole tiles (shuffle) vs. rebuild them per face.

`tile_reorder` reverses the order of the 32x32 column-tiles of a 2D tiled tensor.
It is a pure whole-tile relocation: every output tile is some input tile, moved
intact. Two structurally different writer kernels produce byte-identical output:

    method="relocate" : write each tile as one whole page (NoC1) — the shuffle.
    method="scatter"  : write each tile as 4 separate faces, barrier each — the
                        generic-permute anti-pattern (do the work).

Reader runs on NoC0, writer on NoC1, so reads and writes overlap.

See README.md (next to this file) for the problem this illustrates — permute vs.
transpose, why the op is DRAM-bandwidth-bound, when the shuffle win is large vs.
small — and the commands to run it.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_ID = 0  # reader -> writer

# method names the strategy: relocate whole tiles, or scatter them face-by-face.
METHODS = ("relocate", "scatter")

_WRITER_SOURCE = {
    "relocate": "tile_reorder_writer_relocate.cpp",  # one whole-page write / tile
    "scatter": "tile_reorder_writer_scatter.cpp",  # four 512 B face writes / tile
}


def validate(input_tensor):
    """Keep the example dead simple: 2D, TILE, bf16, tile-aligned, interleaved."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"tile_reorder example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("tile_reorder example: input must be TILE_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("tile_reorder example: input must be bfloat16")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"tile_reorder example: H and W must be multiples of {TILE}, got {shape}")


def _grid_assignment(device, num_pages):
    """Split the output tiles across the full Tensix grid (same for both methods)."""
    grid_size = device.compute_with_storage_grid_size()
    (
        _,
        all_cores,
        core_group_1,
        core_group_2,
        pages_per_core_g1,
        pages_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, num_pages, row_wise=True)

    assignment = []
    start = 0
    for group, per_core in ((core_group_1, pages_per_core_g1), (core_group_2, pages_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    return all_cores, assignment


def _create_program_descriptor(input_tensor, output_tensor, *, writer_source):
    device = input_tensor.device()

    _, w = list(input_tensor.shape)
    tiles_per_row = w // TILE  # Wt: number of column-tiles
    page_bytes = input_tensor.buffer_aligned_page_size()  # one bf16 tile = 2048 B
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()

    all_cores, assignment = _grid_assignment(device, num_pages)

    cb = ttnn.CBDescriptor(
        total_size=2 * page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ID, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    # Reader (NoC0): whole-page read of the remapped tile — identical for both methods.
    reader_ct_args = [tiles_per_row, page_bytes]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # Writer (NoC1): whole-page vs per-face is the whole difference; picked by source file.
    writer_ct_args = [page_bytes]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, start_page, per_core in assignment:
        reader_rt_args[core.x][core.y] = [in_addr, start_page, per_core]
        writer_rt_args[core.x][core.y] = [out_addr, start_page, per_core]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tile_reorder_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / writer_source),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb])


def tile_reorder(
    input_tensor: ttnn.Tensor, *, method: str = "relocate", memory_config: ttnn.MemoryConfig = None
) -> ttnn.Tensor:
    """Reverse the column-tile order of a 2D tiled bf16 tensor.

    ``method`` names the strategy:
      * ``"relocate"`` — move each whole 2 KB tile in one NoC write (shuffle). The win.
      * ``"scatter"``  — write each tile as 4 separate 512 B faces (generic-permute style). Baseline.

    Output is identical for both.
    """
    if method not in METHODS:
        raise ValueError(f"tile_reorder example: method must be one of {METHODS}, got {method!r}")
    validate(input_tensor)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = _create_program_descriptor(input_tensor, output_tensor, writer_source=_WRITER_SOURCE[method])
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
