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
    # NoC per lane (lane 0 = BRISC, lane 1 = NCRISC). Default = the Writer/Reader config defaults
    # (BRISC/NoC1, NCRISC/NoC0) -> byte-identical to the Phase 0 kernels. Catalog ``noc_placement``.
    lane_nocs: tuple = (1, 0)
    # Rotate each lane's walk so its first write lands on a different output bank than its
    # neighbours' (catalog ``bank_stagger``). Host-derived start slot per lane:
    #   (first_page // num_output_banks) % lane_slots
    # Lanes whose first pages share a bank (first_page % banks equal) differ in first_page // banks,
    # so they start on different slots. False -> start slot 0 (Phase 0 walk).
    bank_stagger: bool = False
    # One RISC writes on BOTH NoCs (DM_DYNAMIC_NOC, alternate per page): the second NoC's link
    # capacity without a second kernel instance (whose dispatch cost shows up in trace replay).
    dual_noc: bool = False


DEFAULT_KNOBS = Knobs()

# Plan for an L1-interleaved output (Refinement 2). L1 pages spread over every worker core's bank,
# so the writes are all-to-all across the grid and congest one NoC's links: splitting them over both
# NoCs (dual_noc) and de-clustering the banks the cores hit at each step (bank_stagger) pays.
# Measured on p1 (103 cores, bf8b -> L1): 1939 -> 1289 ns kernel, replay 5931 -> 5882 ns/op.
# A DRAM output stays on DEFAULT_KNOBS: its 8 edge banks are reached best on NoC1 alone (NoC0 doubles
# p2's kernel time) and dual_noc there costs ~+2.7% replay with a flat kernel.
# DM_DYNAMIC_NOC carries a fixed launch cost (~25-40 ns FW / replay), so the plan only pays once the
# grid injects enough pages to congest one NoC: p1 (1030 pages) wins, p2 -> L1 (160 pages) is a
# flat kernel (987 -> 974 ns) and +1.5% replay, so it stays on DEFAULT_KNOBS.
L1_OUTPUT_KNOBS = Knobs(bank_stagger=True, dual_noc=True)
L1_OUTPUT_MIN_PAGES = 512


def select_knobs(output_buffer_type, num_pages):
    """Host predicate: the plan for this call (reported via last_plan(), part of the cache key)."""
    if output_buffer_type == ttnn.BufferType.L1 and num_pages >= L1_OUTPUT_MIN_PAGES:
        return L1_OUTPUT_KNOBS
    return DEFAULT_KNOBS


# lane index -> data-movement processor (lane 0 = BRISC, lane 1 = NCRISC); NoC from Knobs.lane_nocs
_LANE_PROCESSORS = (ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)
_NOCS = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)


def _lane_config(lane, knobs):
    mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC if knobs.dual_noc else ttnn.NOC_MODE.DM_DEDICATED_NOC
    return ttnn.DataMovementConfigDescriptor(
        processor=_LANE_PROCESSORS[lane], noc=_NOCS[knobs.lane_nocs[lane]], noc_mode=mode
    )


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


def _lane_rt_args(geo, rect, knobs, num_banks):
    """RT args [num_blocks, block_rows, block_cols, total_rows, first_page, src_first_slot, start_slot]."""
    row0, col0, r_off, c_off, rows, cols = rect
    if rows == 0 or cols == 0:
        return [0, 0, 0, 0, 0, 0, 0]
    block_rows = -(-rows // knobs.blocks_per_shard)
    num_blocks = -(-rows // block_rows)
    first_page = (row0 + r_off) * geo["tensor_col_tiles"] + (col0 + c_off)
    src_first_slot = r_off * geo["shard_col_tiles"] + c_off  # slot stride = full shard width
    start_slot = (first_page // num_banks) % (rows * cols) if knobs.bank_stagger else 0
    return [num_blocks, block_rows, cols, rows, first_page, src_first_slot, start_slot]


def build_program_descriptor(input_tensor, output_tensor, knobs: Knobs = DEFAULT_KNOBS):
    """Build the cacheable parts of the program: kernels (with per-core RT args).

    Returns (kernel_descriptors, plan). The shard CB is bound per call
    (``bind_call``) because it carries the input buffer pointer."""
    assert knobs.writer_lanes in (1, 2), "writer_lanes must be 1 or 2"
    assert len(knobs.lane_nocs) == 2 and all(n in (0, 1) for n in knobs.lane_nocs)
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
        int(knobs.dual_noc),
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    out_buffer_type = output_tensor.memory_config().buffer_type
    num_banks = ttnn.get_memory_view(output_tensor.device(), out_buffer_type).num_banks
    lane_rt = [ttnn.RuntimeArgs() for _ in range(knobs.writer_lanes)]
    for k, core in enumerate(cores):
        if k < num_shards:
            row0, col0, rows, cols = _shard_rect(geo, k)
            rects = _lane_rects(rows, cols, knobs.writer_lanes)
        else:
            row0 = col0 = 0
            rects = [(0, 0, 0, 0)] * knobs.writer_lanes
        for lane, (r_off, c_off, lr, lc) in enumerate(rects):
            lane_rt[lane][core.x][core.y] = _lane_rt_args(geo, (row0, col0, r_off, c_off, lr, lc), knobs, num_banks)

    kernels = []
    for lane in range(knobs.writer_lanes):
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=WRITER_KERNEL,
                core_ranges=grid,
                compile_time_args=ct_args,
                runtime_args=lane_rt[lane],
                common_runtime_args=[output_tensor.buffer_address()],
                config=_lane_config(lane, knobs),
            )
        )

    plan = {
        "regime": REGIME,
        "num_shards": num_shards,
        "grid_cores": len(cores),
        "writer_lanes": knobs.writer_lanes,
        "blocks_per_shard": knobs.blocks_per_shard,
        "lane_nocs": tuple(knobs.lane_nocs[: knobs.writer_lanes]),
        "bank_stagger": knobs.bank_stagger,
        "dual_noc": knobs.dual_noc,
        "knobs": knobs,
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
