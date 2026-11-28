# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_sharded_to_interleaved — ProgramDescriptor (regime ``owner_writes``).

Each core holding shard k writes its resident shard rectangle straight from L1
(zero-copy CB ``cb_input_shard`` on the input's shard buffer) to the interleaved
output pages. One NoC crossing per tile, no reader, no compute kernel.

Knobs (single source of truth, see ``Knobs``):
  * ``blocks_per_shard`` — number of store blocks per write lane (Phase 0: 1 =
    the whole resident shard is one block). Block rows derive from it.
  * ``writer_lanes``     — 1: one writer (BRISC / NoC1); 2: the shard rectangle
    is split between a BRISC/NoC1 and an NCRISC/NoC0 instance of the same
    kernel (catalog ``split_reader``). Phase 0: 1.

The descriptor is built once per cache key and patched per call (output
address common RT arg + the shard CB's buffer binding), with
``custom_program_hash`` set from the key (BB-07).
"""

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = str(KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp")

TILE = 32
CB_INPUT_SHARD = 0  # the resident input shard (zero-copy, globally allocated on the input buffer)

REGIME = "owner_writes"


@dataclass(frozen=True)
class Knobs:
    blocks_per_shard: int = 1
    writer_lanes: int = 1


DEFAULT_KNOBS = Knobs()

# lane index -> data-movement config (lane 0 = BRISC / NoC1, lane 1 = NCRISC / NoC0)
_LANE_CONFIGS = (ttnn.WriterConfigDescriptor, ttnn.ReaderConfigDescriptor)


def shard_geometry(input_tensor):
    """Tile-level geometry of the input, alignment-aware (per image)."""
    padded = tuple(input_tensor.padded_shape)
    spec = input_tensor.memory_config().shard_spec
    tensor_col_tiles = padded[-1] // TILE
    rows = 1
    for d in padded[:-1]:
        rows *= d
    tensor_row_tiles = rows // TILE
    shard_row_tiles = spec.shape[0] // TILE
    shard_col_tiles = spec.shape[1] // TILE
    height = input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if height:
        num_shards = -(-tensor_row_tiles // shard_row_tiles)
    else:
        num_shards = -(-tensor_col_tiles // shard_col_tiles)
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    return dict(
        height=height,
        tensor_row_tiles=tensor_row_tiles,
        tensor_col_tiles=tensor_col_tiles,
        shard_row_tiles=shard_row_tiles,
        shard_col_tiles=shard_col_tiles,
        num_shards=num_shards,
        grid=spec.grid,
        row_wise=row_wise,
    )


def _shard_rect(geo, k):
    """(row0, col0, rows, cols) of shard k in the tile grid."""
    if geo["height"]:
        row0 = k * geo["shard_row_tiles"]
        return row0, 0, min(geo["shard_row_tiles"], geo["tensor_row_tiles"] - row0), geo["tensor_col_tiles"]
    col0 = k * geo["shard_col_tiles"]
    return 0, col0, geo["tensor_row_tiles"], min(geo["shard_col_tiles"], geo["tensor_col_tiles"] - col0)


def _split(n, parts):
    """Split n into `parts` contiguous chunks -> list of (offset, length)."""
    base, rem = divmod(n, parts)
    out, off = [], 0
    for i in range(parts):
        ln = base + (1 if i < rem else 0)
        out.append((off, ln))
        off += ln
    return out


def _lane_rects(rows, cols, lanes):
    """Disjoint sub-rectangles (r_off, c_off, rows, cols) of a shard rectangle, one per lane.

    Split along rows when there are enough rows, else along columns."""
    if lanes == 1:
        return [(0, 0, rows, cols)]
    if rows >= lanes:
        return [(off, 0, ln, cols) for off, ln in _split(rows, lanes)]
    return [(0, off, rows, ln) for off, ln in _split(cols, lanes)]


def _lane_rt_args(geo, rect, knobs):
    """RT args [num_blocks, block_rows, block_cols, total_rows, first_page, src_first_slot]."""
    row0, col0, r_off, c_off, rows, cols = rect
    if rows == 0 or cols == 0:
        return [0, 0, 0, 0, 0, 0]
    block_rows = -(-rows // knobs.blocks_per_shard)
    num_blocks = -(-rows // block_rows)
    first_page = (row0 + r_off) * geo["tensor_col_tiles"] + (col0 + c_off)
    src_first_slot = r_off * geo["shard_col_tiles"] + c_off  # slot stride = full shard width
    return [num_blocks, block_rows, cols, rows, first_page, src_first_slot]


def build_program_descriptor(input_tensor, output_tensor, knobs: Knobs = DEFAULT_KNOBS):
    """Build the cacheable parts of the program: kernels (with per-core RT args).

    Returns (kernel_descriptors, plan). The shard CB is bound per call
    (``bind_call``) because it carries the input buffer pointer."""
    assert knobs.writer_lanes in (1, 2), "writer_lanes must be 1 or 2"
    assert knobs.blocks_per_shard >= 1
    geo = shard_geometry(input_tensor)
    assert geo["tensor_row_tiles"] * geo["tensor_col_tiles"] < 2**32

    grid = geo["grid"]
    # The allocator's own shard -> core order (buffer.cpp): ranges in set order, each walked
    # row- or column-wise. Shard k lives on core k.
    cores = ttnn.corerange_to_cores(grid, row_wise=geo["row_wise"])
    num_shards = geo["num_shards"]
    assert num_shards <= len(cores), "more shards than shard-grid cores"

    out_page_size = output_tensor.buffer_page_size()
    in_slot_bytes = input_tensor.buffer_aligned_page_size()
    ct_args = [
        CB_INPUT_SHARD,
        geo["shard_col_tiles"],
        geo["tensor_col_tiles"],
        out_page_size,
        in_slot_bytes,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    lane_rt = [ttnn.RuntimeArgs() for _ in range(knobs.writer_lanes)]
    for k, core in enumerate(cores):
        if k < num_shards:
            row0, col0, rows, cols = _shard_rect(geo, k)
            rects = _lane_rects(rows, cols, knobs.writer_lanes)
        else:
            row0 = col0 = 0
            rects = [(0, 0, 0, 0)] * knobs.writer_lanes
        for lane, (r_off, c_off, lr, lc) in enumerate(rects):
            lane_rt[lane][core.x][core.y] = _lane_rt_args(geo, (row0, col0, r_off, c_off, lr, lc), knobs)

    kernels = []
    for lane in range(knobs.writer_lanes):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                core_ranges=grid,
                compile_time_args=ct_args,
                runtime_args=lane_rt[lane],
                common_runtime_args=[output_tensor.buffer_address()],
                config=_LANE_CONFIGS[lane](),
            )
        )

    plan = {
        "regime": REGIME,
        "num_shards": num_shards,
        "grid_cores": len(cores),
        "writer_lanes": knobs.writer_lanes,
        "blocks_per_shard": knobs.blocks_per_shard,
    }
    return kernels, plan


def bind_call(kernels, input_tensor, output_tensor, program_hash):
    """Per-call patch of a cached build: output address + the shard CB on the current input."""
    out_addr = output_tensor.buffer_address()
    for kd in kernels:
        kd.common_runtime_args = [out_addr]
    cb_input_shard = ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor)
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_input_shard])
    desc.custom_program_hash = program_hash
    return desc


def core_roles(input_tensor):
    """BB-21: core -> role ('writer' for shard cores, 'idle' for grid cores without a shard)."""
    geo = shard_geometry(input_tensor)
    cores = ttnn.corerange_to_cores(geo["grid"], row_wise=geo["row_wise"])
    return {(c.x, c.y): ("writer" if k < geo["num_shards"] else "idle") for k, c in enumerate(cores)}
