# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for g6_sharded_to_interleaved: regime "owner writes whole shard".

Every core that holds a shard writes the tiles of that shard, read in place from its own L1
through a CB backed on the sharded input buffer, to their interleaved output pages. The kernel
group is the input's own shard grid. One shard per core (height/width spec), cut into
`num_writer_riscs` blocks, one per data-movement RISC of the core.

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
# cut into this many blocks along its longer axis; RISC i runs the writer on its own NoC.
NUM_WRITER_RISCS = 1

# Free L1 per core the model leaves to the op; an L1 output's own pages come out of it (BB-05).
L1_FREE_BYTES = 1347776

CB_INPUT_SHARD = 0

# Per-core RT args of each writer kernel: [num_blocks, BLOCK_ARGS words per block]. A height or
# width spec puts at most one shard on a core and one block of it on each RISC, so every core
# gets the same arg count (generic_op checks per-core counts on a program-cache hit).
MAX_BLOCKS_PER_RISC = 1
BLOCK_ARGS = 5  # origin_row, origin_col, rows, cols, local_start

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


@dataclass(frozen=True)
class CorePlan:
    core: tuple  # (x, y)
    shard_index: int | None
    blocks_per_riscv: tuple  # tuple over writer RISCs of tuple[Block, ...]


@dataclass(frozen=True)
class Plan:
    regime: str
    num_writer_riscs: int
    tensor_tile_rows: int
    tensor_tile_cols: int
    shard_tile_rows: int
    shard_tile_cols: int
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


def _cache_key(input_tensor, memory_config, num_writer_riscs):
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
    )


def _build_plan(input_tensor, memory_config, num_writer_riscs):
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
            per_riscv = tuple(
                (b,) if b is not None else () for b in _cut_block(shard_block, num_writer_riscs, shard_tile_cols)
            )
            cores.append(CorePlan((core.x, core.y), k, per_riscv))
        else:
            cores.append(CorePlan((core.x, core.y), None, tuple(() for _ in range(num_writer_riscs))))

    l1_output_bytes_per_core = 0
    if memory_config.buffer_type == ttnn.BufferType.L1:
        num_l1_banks = ttnn.device.get_memory_view(input_tensor.device(), ttnn.BufferType.L1).num_banks
        num_pages = tensor_tile_rows * tensor_tile_cols
        l1_output_bytes_per_core = -(-num_pages // num_l1_banks) * page_bytes

    return Plan(
        regime=REGIME,
        num_writer_riscs=num_writer_riscs,
        tensor_tile_rows=tensor_tile_rows,
        tensor_tile_cols=tensor_tile_cols,
        shard_tile_rows=shard_tile_rows,
        shard_tile_cols=shard_tile_cols,
        num_shards=num_shards,
        page_bytes=page_bytes,
        cores=tuple(cores),
        l1_output_bytes_per_core=l1_output_bytes_per_core,
    )


def _riscv_runtime_args(plan, riscv):
    rt_args = ttnn.RuntimeArgs()
    for core_plan in plan.cores:
        blocks = core_plan.blocks_per_riscv[riscv]
        args = [len(blocks)]
        for b in blocks:
            args += [b.origin_row, b.origin_col, b.rows, b.cols, b.local_start]
        args += [0] * (1 + MAX_BLOCKS_PER_RISC * BLOCK_ARGS - len(args))
        x, y = core_plan.core
        rt_args[x][y] = args
    return rt_args


def _build_kernels(plan, input_tensor, output_tensor):
    assert output_tensor.buffer_aligned_page_size() == plan.page_bytes, (
        f"output page {output_tensor.buffer_aligned_page_size()} B != input page {plan.page_bytes} B: "
        "the shard stride and the write size must agree"
    )
    grid = input_tensor.memory_config().shard_spec.grid
    ct_args = [CB_INPUT_SHARD, plan.shard_tile_cols, plan.tensor_tile_cols, plan.page_bytes]
    ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    kernels = []
    for riscv in range(plan.num_writer_riscs):
        processor, noc = _WRITER_PLACEMENT[riscv]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_KERNEL),
                core_ranges=grid,
                compile_time_args=ct_args,
                runtime_args=_riscv_runtime_args(plan, riscv),
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


def get_plan(input_tensor, memory_config, num_writer_riscs=None):
    """The cached host plan for these arguments (built on first use)."""
    return _get_entry(input_tensor, memory_config, num_writer_riscs).plan


def _get_entry(input_tensor, memory_config, num_writer_riscs=None):
    if num_writer_riscs is None:
        num_writer_riscs = NUM_WRITER_RISCS
    assert 1 <= num_writer_riscs <= len(_WRITER_PLACEMENT), f"num_writer_riscs={num_writer_riscs}"
    key = _cache_key(input_tensor, memory_config, num_writer_riscs)
    entry = _CACHE.get(key)
    if entry is None:
        entry = _CacheEntry(_build_plan(input_tensor, memory_config, num_writer_riscs))
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
