# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone program descriptor: regimes `flat_page_range` and `logical_rows_partial_tile`.

Reader (NCRISC, NoC0) -> cb_page_staging -> writer (BRISC, NoC1); no compute kernel, so the
bytes never pass through an unpacker, DEST or a packer. The flat page axis is split into
contiguous page ranges over the compute grid; each core walks its range in blocks of
`block_pages` pages. Both regimes share that split, the CB and the block loop; they differ only
in the per-page transfer. `flat_page_range` moves every whole aligned page.
`logical_rows_partial_tile` (TILE with a partial last tile row or column) moves only the face
rows of a partial page that hold logical elements, selected by a CT flag.

Every block knob is defined once here and every dependent quantity is derived from it:
`block_pages` and `cb_depth_blocks` set the CB capacity, the push/pop quantum (CT arg) and the
per-core block count the kernels derive from `num_pages_this_core`.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

REGIME_FLAT = "flat_page_range"
REGIME_LOGICAL_ROWS = "logical_rows_partial_tile"

TILE_HW = 32
FACE_HW = 16

# Page kinds of the logical-rows walk, in the order of the kernels' CT array:
# kind = (tile row is the image's last) | (tile column is the image's last) << 1.
NUM_PAGE_KINDS = 4

# Buffer-depth knob of cb_page_staging: slots of `block_pages` pages when a core has more than
# one block, so the reader fills block k+1 while the writer drains block k.
CB_DEPTH_KNOB = 2

# Block-size knob: how many blocks a core's assignment is cut into before the L1 cap applies.
# 1 keeps the whole assignment in flight behind one barrier; on p1, 2-7 blocks with depth 2 measured
# within noise of 1 (63.5-64.4 us) and 28 blocks (one page each) 10% slower (70.3 us).
BLOCKS_PER_CORE_KNOB = 1

# Transfer-shape knob of logical_rows_partial_tile: two logical pieces of a page whose gap is at
# most this many bytes move as one span through the gap, trading padding bytes for one fewer
# NoC request per page. 0 merges only touching pieces. Small partial pieces are request-bound
# (Blackhole p150b, 1940 pages on 110 cores): a 448 B gap (one-row bf16 tiles) merged runs
# 7.06 us vs 9.35 us split, while merging a 512 B gap (17x16 bf16 tiles) costs +3%, 960 B
# (one-row f32 tiles) +10% and 1024 B (17x16 f32 tiles) +53%.
PIECE_MERGE_GAP_BYTES = 448

# L1 the model environment leaves to the op on every core (BB-05).
L1_BUDGET_BYTES = 1347776

CB_PAGE_STAGING = 0

# Order of the kernels in the descriptor; patch_addresses indexes by it.
READER_KERNEL_IDX = 0
WRITER_KERNEL_IDX = 1


def tile_geometry(input_tensor: ttnn.Tensor):
    """(h_tiles, w_tiles, rows_last, cols_last, elem_bytes) of a TILE tensor whose padding is plain
    tile padding of the last two dims, else None.

    rows_last / cols_last are the logical rows / columns of the last tile row / column of each
    image (1..32). Anything else (ROW_MAJOR, a non-32x32 or transposed tile, extra padding beyond
    the next tile edge, padded leading dims) returns None and runs `flat_page_range`, which copies
    whole pages and is correct for every layout.
    """
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return None
    tile = input_tensor.tile
    if (
        list(tile.tile_shape) != [TILE_HW, TILE_HW]
        or list(tile.face_shape) != [FACE_HW, FACE_HW]
        or tile.transpose_within_face
        or tile.transpose_of_faces
    ):
        return None
    logical = [int(d) for d in input_tensor.shape]
    padded = [int(d) for d in input_tensor.padded_shape]
    height = logical[-2] if len(logical) >= 2 else 1
    width = logical[-1]
    if len(padded) < 2 or height == 0 or width == 0:
        return None
    padded_h, padded_w = padded[-2], padded[-1]
    if padded_h != -(-height // TILE_HW) * TILE_HW or padded_w != -(-width // TILE_HW) * TILE_HW:
        return None
    if math.prod(padded[:-2]) != math.prod(logical[:-2]):
        return None
    h_tiles, w_tiles = padded_h // TILE_HW, padded_w // TILE_HW
    if math.prod(padded[:-2]) * h_tiles * w_tiles != input_tensor.buffer_num_pages():
        return None
    aligned_page_bytes = input_tensor.buffer_aligned_page_size()
    elem_bytes, remainder = divmod(aligned_page_bytes, TILE_HW * TILE_HW)
    if remainder or elem_bytes == 0:
        return None
    return (
        h_tiles,
        w_tiles,
        height - (h_tiles - 1) * TILE_HW,
        width - (w_tiles - 1) * TILE_HW,
        elem_bytes,
    )


def logical_pieces(rows: int, cols: int, elem_bytes: int, granule: int, merge_gap: int = 0) -> list:
    """Byte ranges (offset, length) of a 32x32 tile page that hold its first `rows` rows and
    first `cols` columns: whole face rows, merged where at most `merge_gap` bytes apart, each
    length rounded up to `granule` (the DRAM alignment) so every NoC transfer is aligned at both
    ends. Rounding adds at most one face row of padding and never crosses into the next face."""
    face_row_bytes = FACE_HW * elem_bytes
    face_bytes = FACE_HW * face_row_bytes
    top = min(rows, FACE_HW)
    bottom = rows - top
    right = cols > FACE_HW
    faces = [(0, top)]
    if right:
        faces.append((1, top))
    if bottom:
        faces.append((2, bottom))
        if right:
            faces.append((3, bottom))
    pieces = []
    for face, face_rows in faces:
        offset = face * face_bytes
        length = -(-face_rows * face_row_bytes // granule) * granule
        if pieces and offset - (pieces[-1][0] + pieces[-1][1]) <= merge_gap:
            pieces[-1] = (pieces[-1][0], offset + length - pieces[-1][0])
        else:
            pieces.append((offset, length))
    return pieces


def partial_tile_walk(geometry, aligned_page_bytes: int):
    """The logical-rows walk of `geometry`, or None when no page kind moves fewer bytes than the
    whole page (then the regime would only add transactions)."""
    if geometry is None:
        return None
    h_tiles, w_tiles, rows_last, cols_last, elem_bytes = geometry
    granule = ttnn.get_dram_alignment()
    kind_extents = [(TILE_HW, TILE_HW), (rows_last, TILE_HW), (TILE_HW, cols_last), (rows_last, cols_last)]
    kinds = [logical_pieces(rows, cols, elem_bytes, granule, PIECE_MERGE_GAP_BYTES) for rows, cols in kind_extents]
    for pieces in kinds:
        # Each piece must stay inside the page; the kernels issue at most two per page.
        if len(pieces) > 2 or any(offset + length > aligned_page_bytes for offset, length in pieces):
            raise RuntimeError(f"g6_clone: logical pieces {pieces} do not fit a {aligned_page_bytes} B page")
    if all(sum(length for _, length in pieces) >= aligned_page_bytes for pieces in kinds):
        return None
    return {
        "h_tiles": h_tiles,
        "w_tiles": w_tiles,
        "rows_last": rows_last,
        "cols_last": cols_last,
        "kinds": kinds,
    }


def _walk_ct_args(walk) -> list:
    """CT args of g6_clone_page_pieces.hpp's PartialTileWalk: w_tiles, h_tiles, then 4 values per
    kind. The flat regime passes a fixed placeholder so every flat descriptor compiles one binary."""
    if walk is None:
        return [1, 1] + [0] * (4 * NUM_PAGE_KINDS)
    args = [walk["w_tiles"], walk["h_tiles"]]
    for pieces in walk["kinds"]:
        padded = list(pieces) + [(0, 0)] * (2 - len(pieces))
        for offset, length in padded:
            args.extend([offset, length])
    return args


def compute_plan(device, tensor_pages: int, aligned_page_bytes: int, geometry=None) -> dict:
    """Work split, block knobs and regime for one (grid, page count, page size, tile geometry)."""
    grid = device.compute_with_storage_grid_size()
    walk = partial_tile_walk(geometry, aligned_page_bytes)

    block_pages_l1_cap = L1_BUDGET_BYTES // (CB_DEPTH_KNOB * aligned_page_bytes)
    if block_pages_l1_cap < 1:
        raise RuntimeError(
            f"g6_clone: a page of {aligned_page_bytes} B does not fit {CB_DEPTH_KNOB} staging slots "
            f"in the {L1_BUDGET_BYTES} B per-core L1 budget"
        )

    num_cores, all_cores, core_group_1, core_group_2, core_pages_g1, core_pages_g2 = ttnn.split_work_to_cores(
        grid, tensor_pages, row_wise=True
    )
    core_pages_max = core_pages_g1

    block_pages = min(-(-core_pages_max // BLOCKS_PER_CORE_KNOB), block_pages_l1_cap)
    num_blocks_max = -(-core_pages_max // block_pages)
    cb_depth_blocks = min(CB_DEPTH_KNOB, num_blocks_max)

    # split_work_to_cores(row_wise=True) hands group 1 the first cores in row-major order and
    # group 2 the rest; start_page is the running page count in that same order.
    num_cores_g1 = core_group_1.num_cores()
    roles = []
    start_page = 0
    for idx, core in enumerate(ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)):
        num_pages = core_pages_g1 if idx < num_cores_g1 else core_pages_g2
        roles.append((core, "reader+writer", start_page, num_pages))
        start_page += num_pages
    if start_page != tensor_pages:
        raise RuntimeError(f"g6_clone: work split covers {start_page} of {tensor_pages} pages")

    return {
        "regime": REGIME_LOGICAL_ROWS if walk is not None else REGIME_FLAT,
        "partial_tile_walk": walk,
        "grid": (grid.x, grid.y),
        "num_cores": num_cores,
        "core_ranges": all_cores,
        "core_pages_g1": core_pages_g1,
        "core_pages_g2": core_pages_g2,
        "block_pages": block_pages,
        "cb_depth_blocks": cb_depth_blocks,
        "num_blocks_max": num_blocks_max,
        "roles": roles,
        "tensor_pages": tensor_pages,
        "aligned_page_bytes": aligned_page_bytes,
    }


def create_program_descriptor(
    input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: dict
) -> ttnn.ProgramDescriptor:
    """Descriptor for `plan`. Buffer addresses sit in common RT args, slot 0 of each kernel,
    so a cached descriptor is reused by patching two values (see patch_addresses)."""
    all_cores = plan["core_ranges"]
    block_pages = plan["block_pages"]
    aligned_page_bytes = plan["aligned_page_bytes"]

    cb_page_staging = ttnn.CBDescriptor(
        total_size=plan["cb_depth_blocks"] * block_pages * aligned_page_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PAGE_STAGING,
                data_format=input_tensor.dtype,
                page_size=aligned_page_bytes,
            )
        ],
    )

    input_accessor_args = ttnn.TensorAccessorArgs(input_tensor)
    output_accessor_args = ttnn.TensorAccessorArgs(output_tensor)
    # Common RT slot 0 holds the buffer address; an interleaved accessor adds no common RT args.
    if len(input_accessor_args.get_common_runtime_args()) or len(output_accessor_args.get_common_runtime_args()):
        raise RuntimeError("g6_clone: interleaved TensorAccessorArgs unexpectedly carry common runtime args")

    walk = plan["partial_tile_walk"]
    shared_ct_args = [CB_PAGE_STAGING, block_pages, int(walk is not None)] + _walk_ct_args(walk)
    reader_ct_args = shared_ct_args + list(input_accessor_args.get_compile_time_args())
    writer_ct_args = shared_ct_args + list(output_accessor_args.get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, _, start_page, num_pages in plan["roles"]:
        reader_rt_args[core.x][core.y] = [start_page, num_pages]
        writer_rt_args[core.x][core.y] = [start_page, num_pages]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_clone_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_clone_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    # List order is READER_KERNEL_IDX, WRITER_KERNEL_IDX.
    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb_page_staging])


def patch_addresses(descriptor: ttnn.ProgramDescriptor, input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> None:
    """Write this call's buffer addresses into a cached descriptor, in place."""
    kernels = descriptor.kernels
    kernels[READER_KERNEL_IDX].common_runtime_args[0] = input_tensor.buffer_address()
    kernels[WRITER_KERNEL_IDX].common_runtime_args[0] = output_tensor.buffer_address()
