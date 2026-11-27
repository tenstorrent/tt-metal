# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_sharded_to_interleaved: regime "owner writes whole shard".

Every core that holds a shard writes the tiles of that shard, read in place from its own L1
through a CB backed on the sharded input buffer, to their interleaved output pages. The kernel
group is the input's own shard grid. One shard per core (height/width spec), cut into
`num_writer_riscs` blocks, one per data-movement RISC of the core.

The kernel derives its block from its logical coordinate and the shard geometry, all compile-time
args, so the program has no per-core runtime args (each costs a dispatcher write per core on
every call). The host plan below mirrors that derivation for describe_plan and the L1 guard.

Host work is cached per key (BB-07): the plan and the descriptor are built once; a call only
rebinds the input CB to its tensor and the output base address (a common RT arg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "g6_sharded_to_interleaved_writer.cpp"

REGIME = "owner_writes_whole_shard"

TILE = 32

# Block knob: data-movement RISCs per core that share the writes of one shard. Each shard is
# cut into this many blocks along its longer axis; RISC i runs the writer on its own NoC. Two
# beat one on trace-replay host time for both production cases (p1 2590 vs 2613 ns, p2 1882 vs
# 1970 ns) and on p1 kernel time (1.64 vs 2.08 us), although p2's profiled kernel time is 8 %
# longer (its NoC0 DRAM drain is slower).
NUM_WRITER_RISCS = 2

# Write-order knob: True / False force it; None picks it from the bank geometry. When on, the
# core at walk index k starts its block walk at block tile k mod block_tiles. In order, step j of
# every core writes page k * S + j (S = page stride between consecutive shards), and those pages
# fall on only num_banks / gcd(S, num_banks) banks (p1: S = 10, 110 L1 banks -> 11 banks per
# step); rotation spreads them. With gcd 1 (p2: S = 5, 8 DRAM banks) the order is already spread
# and rotation only costs time (p2 replay +43 ns).
ROTATE_WRITE_ORDER = None

# Free L1 per core the model leaves to the op; an L1 output's own pages come out of it (BB-05).
L1_FREE_BYTES = 1347776

CB_INPUT_SHARD = 0

# Processor and NoC of writer RISC i. BRISC/NoC1 is the default writer placement; the second
# RISC takes the other NoC so the two write streams do not share one.
_WRITER_PLACEMENT = (
    (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
)


@dataclass(frozen=True)
class Block:
    origin_row: int  # first tensor tile row of the block
    origin_col: int  # first tensor tile column of the block
    rows: int
    cols: int
    local_start: int  # in-shard tile index of the block's top-left tile
    start_row: int = 0  # block tile the write walk begins at (row-major, wrapping)
    start_col: int = 0


@dataclass(frozen=True)
class CorePlan:
    core: tuple  # (x, y)
    shard_index: int | None
    blocks_per_riscv: tuple  # tuple over writer RISCs of tuple[Block, ...]


@dataclass(frozen=True)
class Plan:
    regime: str
    num_writer_riscs: int
    rotate_write_order: bool
    num_output_banks: int
    tensor_tile_rows: int
    tensor_tile_cols: int
    shard_tile_rows: int
    shard_tile_cols: int
    num_shard_cols: int
    num_shards: int
    page_bytes: int
    cores: tuple  # tuple[CorePlan, ...] in the shard walk order
    l1_output_bytes_per_core: int  # 0 for a DRAM output


def _split(extent, parts):
    """`parts` contiguous (offset, length) pieces of `extent`, lengths differing by at most 1."""
    base, rem = divmod(extent, parts)
    pieces, offset = [], 0
    for i in range(parts):
        length = base + (1 if i < rem else 0)
        pieces.append((offset, length))
        offset += length
    return pieces


def _cut_block(block, parts, shard_tile_cols):
    """Cut a shard block into `parts` blocks along its longer axis (empty parts kept as None)."""
    blocks = []
    if block.rows >= block.cols:
        for offset, length in _split(block.rows, parts):
            blocks.append(
                Block(
                    block.origin_row + offset,
                    block.origin_col,
                    length,
                    block.cols,
                    block.local_start + offset * shard_tile_cols,
                )
                if length
                else None
            )
    else:
        for offset, length in _split(block.cols, parts):
            blocks.append(
                Block(block.origin_row, block.origin_col + offset, block.rows, length, block.local_start + offset)
                if length
                else None
            )
    return blocks


def _rotated(block, walk_index):
    start_row, start_col = divmod(walk_index % (block.rows * block.cols), block.cols)
    return Block(block.origin_row, block.origin_col, block.rows, block.cols, block.local_start, start_row, start_col)


def _cache_key(input_tensor, memory_config, num_writer_riscs, rotate_write_order):
    in_mc = input_tensor.memory_config()
    spec = in_mc.shard_spec
    return (
        input_tensor.device().id(),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        in_mc.memory_layout,
        tuple(((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in spec.grid.ranges()),
        tuple(spec.shape),
        spec.orientation,
        memory_config.buffer_type,
        num_writer_riscs,
        rotate_write_order,
    )


def _build_plan(input_tensor, memory_config, num_writer_riscs, rotate_write_order):
    padded = tuple(input_tensor.padded_shape)
    tensor_tile_cols = padded[-1] // TILE
    tensor_tile_rows = math.prod(padded) // padded[-1] // TILE

    spec = input_tensor.memory_config().shard_spec
    shard_tile_rows = spec.shape[0] // TILE
    shard_tile_cols = spec.shape[1] // TILE
    num_shard_cols = -(-tensor_tile_cols // shard_tile_cols)
    num_shards = -(-tensor_tile_rows // shard_tile_rows) * num_shard_cols

    # The walk the sharded buffer itself uses: ranges in stored order, each along rows or columns.
    walk = ttnn.corerange_to_cores(spec.grid, None, spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
    assert num_shards <= len(walk), f"{num_shards} shards on {len(walk)} cores: a height/width spec holds one per core"

    page_bytes = input_tensor.buffer_page_size()
    assert (
        input_tensor.buffer_aligned_page_size() == page_bytes
    ), f"input page {page_bytes} B is not aligned ({input_tensor.buffer_aligned_page_size()} B)"

    num_output_banks = ttnn.device.get_memory_view(input_tensor.device(), memory_config.buffer_type).num_banks
    if rotate_write_order is None:
        shard_page_stride = shard_tile_cols if num_shard_cols > 1 else shard_tile_rows * tensor_tile_cols
        rotate_write_order = math.gcd(shard_page_stride, num_output_banks) > 1

    l1_output_bytes_per_core = 0
    if memory_config.buffer_type == ttnn.BufferType.L1:
        num_pages = tensor_tile_rows * tensor_tile_cols
        l1_output_bytes_per_core = -(-num_pages // num_output_banks) * page_bytes

    cores = []
    for k, core in enumerate(walk):
        if k < num_shards:
            origin_row = (k // num_shard_cols) * shard_tile_rows
            origin_col = (k % num_shard_cols) * shard_tile_cols
            shard_block = Block(
                origin_row,
                origin_col,
                min(shard_tile_rows, tensor_tile_rows - origin_row),
                min(shard_tile_cols, tensor_tile_cols - origin_col),
                0,
            )
            blocks = _cut_block(shard_block, num_writer_riscs, shard_tile_cols)
            if rotate_write_order:
                blocks = [_rotated(b, k) if b is not None else None for b in blocks]
            per_riscv = tuple((b,) if b is not None else () for b in blocks)
            cores.append(CorePlan((core.x, core.y), k, per_riscv))
        else:
            cores.append(CorePlan((core.x, core.y), None, tuple(() for _ in range(num_writer_riscs))))

    return Plan(
        regime=REGIME,
        num_writer_riscs=num_writer_riscs,
        rotate_write_order=rotate_write_order,
        num_output_banks=num_output_banks,
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        shard_tile_rows=shard_tile_rows,
        shard_tile_cols=shard_tile_cols,
        num_shard_cols=num_shard_cols,
        num_shards=num_shards,
        page_bytes=page_bytes,
        cores=tuple(cores),
        l1_output_bytes_per_core=l1_output_bytes_per_core,
    )


def _build_kernels(plan, input_tensor, output_tensor):
    assert output_tensor.buffer_aligned_page_size() == plan.page_bytes, (
        f"output page {output_tensor.buffer_aligned_page_size()} B != input page {plan.page_bytes} B: "
        "the shard stride and the write size must agree"
    )
    spec = input_tensor.memory_config().shard_spec
    ranges = [v for r in spec.grid.ranges() for v in (r.start.x, r.start.y, r.end.x, r.end.y)]
    output_accessor_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    kernels = []
    for riscv in range(plan.num_writer_riscs):
        ct_args = [
            CB_INPUT_SHARD,
            plan.shard_tile_rows,
            plan.shard_tile_cols,
            plan.tensor_tile_rows,
            plan.tensor_tile_cols,
            plan.page_bytes,
            plan.num_shard_cols,
            plan.num_shards,
            int(spec.orientation == ttnn.ShardOrientation.ROW_MAJOR),
            plan.num_writer_riscs,
            riscv,
            int(plan.rotate_write_order),
            len(ranges) // 4,
            *ranges,
            *output_accessor_args,
        ]
        processor, noc = _WRITER_PLACEMENT[riscv]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_KERNEL),
                core_ranges=spec.grid,
                compile_time_args=ct_args,
                common_runtime_args=[output_tensor.buffer_address()],
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
            )
        )
    return kernels


class _CacheEntry:
    def __init__(self, plan):
        self.plan = plan
        self.kernels = None
        self.descriptor = None


_CACHE = {}


def get_plan(input_tensor, memory_config):
    """The cached host plan for these arguments (built on first use)."""
    return _get_entry(input_tensor, memory_config).plan


def _get_entry(input_tensor, memory_config):
    num_writer_riscs, rotate_write_order = NUM_WRITER_RISCS, ROTATE_WRITE_ORDER
    assert 1 <= num_writer_riscs <= len(_WRITER_PLACEMENT), f"num_writer_riscs={num_writer_riscs}"
    key = _cache_key(input_tensor, memory_config, num_writer_riscs, rotate_write_order)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CacheEntry(_build_plan(input_tensor, memory_config, num_writer_riscs, rotate_write_order))
        _CACHE[key] = entry
    return entry


def check_l1_bound(plan):
    """BB-05: an L1 output's own pages must fit the L1 the model leaves free."""
    if plan.l1_output_bytes_per_core > L1_FREE_BYTES:
        raise RuntimeError(
            f"g6_sharded_to_interleaved: the L1 output needs {plan.l1_output_bytes_per_core} B per bank, "
            f"more than the {L1_FREE_BYTES} B left free per core"
        )


def create_program_descriptor(input_tensor, output_tensor, memory_config):
    """The cached descriptor, bound to this call's input shard and output buffer."""
    entry = _get_entry(input_tensor, memory_config)
    if entry.descriptor is None:
        entry.kernels = _build_kernels(entry.plan, input_tensor, output_tensor)
        entry.descriptor = ttnn.ProgramDescriptor(kernels=entry.kernels, semaphores=[], cbs=[])

    # Per call: this input's shard buffer and this output's base address. nanobind vector
    # properties hand out copies, so the lists are reassigned whole.
    output_addr = output_tensor.buffer_address()
    for kernel in entry.kernels:
        kernel.common_runtime_args = [output_addr]
    entry.descriptor.kernels = entry.kernels
    entry.descriptor.cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor)]
    return entry.descriptor
