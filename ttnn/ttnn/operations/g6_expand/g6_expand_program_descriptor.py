# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program descriptor: regime R1 `column_mcast_2d` (op_design.md, Blocking Model).

The pattern axis (input page p = image * tensor_pattern_wt + col) is split over grid columns,
the replica axis (output tile row within an image) over grid rows. The top core of each grid
column (injector) fetches row 0 of its x pages, replicates it into whole pattern pages and
multicasts the block down the column; every core writes its replica rows x its pattern block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# Blocking-model knobs. Each is defined once; every page count, trip count, runtime argument and
# the core rectangle derive from select_plan(). None means "the device grid's extent on that axis".
MAX_PATTERN_GROUPS = None
MAX_REPLICA_GROUPS = None
PATTERN_CB_DEPTH_BLOCKS = 1

# Per-core L1 the model leaves to the op (BB-05), and the share kept back for the multicast
# semaphores and allocation alignment.
L1_OP_BUDGET_BYTES = 1_347_776
L1_RESERVED_BYTES = 4096

CB_PATTERN_TILES = 0
PATTERN_MCAST_PREFIX = "pattern_mcast"
PATTERN_MCAST_FIRST_SEMAPHORE_ID = 0

REGIME = "column_mcast_2d"
ROLE_INJECTOR = "injector"
ROLE_RECEIVER = "receiver"


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def output_shape(input_shape, sizes) -> list[int]:
    """sizes with every -1 replaced by the input's size along that dim."""
    return [int(d) if int(s) == -1 else int(s) for d, s in zip(input_shape, sizes)]


@dataclass(frozen=True)
class Plan:
    """The work split one call runs with (BB-08). Grid column gc = pattern group, grid row gr = replica group."""

    regime: str
    grid: tuple[int, int]
    tile_bytes: int
    tensor_pattern_wt: int
    tensor_pattern_tiles: int
    tensor_replica_rows: int
    pattern_tiles_per_group: int
    pattern_groups: int
    replica_rows_per_group: int
    replica_groups: int
    block_pattern_tiles: int
    pattern_cb_depth_blocks: int

    @property
    def num_cores(self) -> int:
        return self.pattern_groups * self.replica_groups

    @property
    def has_receivers(self) -> bool:
        return self.replica_groups > 1

    @property
    def cb_pattern_pages(self) -> int:
        return self.pattern_cb_depth_blocks * self.block_pattern_tiles

    def core_patterns(self, gc: int) -> tuple[int, int]:
        """(first pattern page, pattern page count) of grid column gc."""
        start = gc * self.pattern_tiles_per_group
        return start, min(self.tensor_pattern_tiles, start + self.pattern_tiles_per_group) - start

    def core_replicas(self, gr: int) -> tuple[int, int]:
        """(first replica row, replica row count) of grid row gr."""
        start = gr * self.replica_rows_per_group
        return start, min(self.tensor_replica_rows, start + self.replica_rows_per_group) - start

    def num_blocks(self, gc: int) -> int:
        return _div_up(self.core_patterns(gc)[1], self.block_pattern_tiles)

    def role(self, gc: int, gr: int) -> str:
        return ROLE_INJECTOR if gr == 0 else ROLE_RECEIVER

    def roles(self) -> dict[tuple[int, int], str]:
        """Role of every core (x, y) of the rectangle (BB-21)."""
        return {(gc, gr): self.role(gc, gr) for gr in range(self.replica_groups) for gc in range(self.pattern_groups)}


def select_plan(
    input_shape,
    sizes,
    grid: tuple[int, int],
    tile_bytes: int,
    *,
    max_pattern_groups: int | None = None,
    max_replica_groups: int | None = None,
    pattern_cb_depth_blocks: int | None = None,
) -> Plan:
    """Evenized ceil split of both axes over the grid rectangle (op_design.md, Work Distribution)."""
    grid_x, grid_y = int(grid[0]), int(grid[1])
    max_pattern_groups = grid_x if max_pattern_groups is None else int(max_pattern_groups)
    max_replica_groups = grid_y if max_replica_groups is None else int(max_replica_groups)
    depth = PATTERN_CB_DEPTH_BLOCKS if pattern_cb_depth_blocks is None else int(pattern_cb_depth_blocks)
    if not (1 <= max_pattern_groups <= grid_x and 1 <= max_replica_groups <= grid_y and depth >= 1):
        raise ValueError(
            f"g6_expand: invalid plan knobs max_pattern_groups={max_pattern_groups}, "
            f"max_replica_groups={max_replica_groups}, pattern_cb_depth_blocks={depth} for grid {grid_x}x{grid_y}"
        )

    out = output_shape(input_shape, sizes)
    tensor_pattern_wt = _div_up(out[3], TILE_HW)
    tensor_pattern_tiles = out[0] * out[1] * tensor_pattern_wt
    tensor_replica_rows = _div_up(out[2], TILE_HW)

    pattern_tiles_per_group = _div_up(tensor_pattern_tiles, min(tensor_pattern_tiles, max_pattern_groups))
    pattern_groups = _div_up(tensor_pattern_tiles, pattern_tiles_per_group)
    replica_rows_per_group = _div_up(tensor_replica_rows, min(tensor_replica_rows, max_replica_groups))
    replica_groups = _div_up(tensor_replica_rows, replica_rows_per_group)

    max_block_pattern_tiles = (L1_OP_BUDGET_BYTES - L1_RESERVED_BYTES) // (depth * tile_bytes)
    block_pattern_tiles = min(pattern_tiles_per_group, max_block_pattern_tiles)

    return Plan(
        regime=REGIME,
        grid=(grid_x, grid_y),
        tile_bytes=tile_bytes,
        tensor_pattern_wt=tensor_pattern_wt,
        tensor_pattern_tiles=tensor_pattern_tiles,
        tensor_replica_rows=tensor_replica_rows,
        pattern_tiles_per_group=pattern_tiles_per_group,
        pattern_groups=pattern_groups,
        replica_rows_per_group=replica_rows_per_group,
        replica_groups=replica_groups,
        block_pattern_tiles=block_pattern_tiles,
        pattern_cb_depth_blocks=depth,
    )


def _core_range_set(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


@dataclass
class _CachedProgram:
    descriptor: ttnn.ProgramDescriptor
    plan: Plan
    # Kernel indices whose common runtime arg 0 is the buffer address of x / of out.
    input_address_kernels: tuple[int, ...]
    output_address_kernels: tuple[int, ...]


def _build(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, plan: Plan) -> _CachedProgram:
    device = input_tensor.device()
    tile_bytes = plan.tile_bytes
    rectangle = _core_range_set(0, 0, plan.pattern_groups - 1, plan.replica_groups - 1)

    cb_pattern_tiles = ttnn.CBDescriptor(
        total_size=plan.cb_pattern_pages * tile_bytes,
        core_ranges=rectangle,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PATTERN_TILES,
                data_format=input_tensor.dtype,
                page_size=tile_bytes,
            )
        ],
    )

    x_accessor_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    out_accessor_ct = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    def reader_kernel(is_injector: bool, y0: int, y1: int) -> ttnn.KernelDescriptor:
        runtime_args = ttnn.RuntimeArgs()
        for gr in range(y0, y1 + 1):
            for gc in range(plan.pattern_groups):
                pattern_start, core_pattern_tiles = plan.core_patterns(gc)
                runtime_args[gc][gr] = [pattern_start, core_pattern_tiles]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=_core_range_set(0, y0, plan.pattern_groups - 1, y1),
            compile_time_args=[
                CB_PATTERN_TILES,
                tile_bytes,
                plan.block_pattern_tiles,
                int(is_injector),
                *x_accessor_ct,
            ],
            runtime_args=runtime_args,
            common_runtime_args=[x_addr],
            config=ttnn.ReaderConfigDescriptor(),
        )

    injector = reader_kernel(True, 0, 0)
    readers = [injector]
    if plan.has_receivers:
        readers.append(reader_kernel(False, 1, plan.replica_groups - 1))

    writer_runtime_args = ttnn.RuntimeArgs()
    for gr in range(plan.replica_groups):
        replica_start, core_replica_rows = plan.core_replicas(gr)
        for gc in range(plan.pattern_groups):
            pattern_start, core_pattern_tiles = plan.core_patterns(gc)
            writer_runtime_args[gc][gr] = [pattern_start, core_pattern_tiles, replica_start, core_replica_rows]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=rectangle,
        compile_time_args=[
            CB_PATTERN_TILES,
            tile_bytes,
            plan.block_pattern_tiles,
            plan.tensor_pattern_wt,
            plan.tensor_replica_rows,
            *out_accessor_ct,
        ],
        runtime_args=writer_runtime_args,
        common_runtime_args=[out_addr],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_pattern_tiles])
    if plan.has_receivers:
        # One column-major group per grid column; group member 0 (the top core) is the fixed sender.
        pattern_mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
            rectangle,
            plan.replica_groups,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.ColumnMajor,
        )
        pattern_mcast.attach(descriptor, PATTERN_MCAST_PREFIX, readers, PATTERN_MCAST_FIRST_SEMAPHORE_ID)
    else:
        # A one-core column has nobody to send to: the injector skips the multicast entirely.
        ttnn.attach_absent_mcast(injector, PATTERN_MCAST_PREFIX)

    kernels = [*readers, writer]
    descriptor.kernels = kernels
    return _CachedProgram(
        descriptor=descriptor,
        plan=plan,
        input_address_kernels=tuple(range(len(readers))),
        output_address_kernels=(len(readers),),
    )


_PROGRAM_CACHE: dict[tuple, _CachedProgram] = {}


def _cache_key(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, grid) -> tuple:
    memory_config = input_tensor.memory_config()
    return (
        input_tensor.device().id(),
        int(grid.x),
        int(grid.y),
        tuple(input_tensor.shape),
        tuple(output_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        memory_config.buffer_type,
        memory_config.memory_layout,
        MAX_PATTERN_GROUPS,
        MAX_REPLICA_GROUPS,
        PATTERN_CB_DEPTH_BLOCKS,
    )


def program_descriptor(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> tuple[ttnn.ProgramDescriptor, Plan]:
    """The descriptor for this call: built once per key, then only the two buffer addresses are patched (BB-07)."""
    grid = input_tensor.device().compute_with_storage_grid_size()
    key = _cache_key(input_tensor, output_tensor, grid)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        plan = select_plan(
            list(input_tensor.shape),
            list(output_tensor.shape),
            (grid.x, grid.y),
            input_tensor.buffer_aligned_page_size(),
            max_pattern_groups=MAX_PATTERN_GROUPS,
            max_replica_groups=MAX_REPLICA_GROUPS,
            pattern_cb_depth_blocks=PATTERN_CB_DEPTH_BLOCKS,
        )
        cached = _build(input_tensor, output_tensor, plan)
        _PROGRAM_CACHE[key] = cached
        return cached.descriptor, cached.plan

    # descriptor.kernels hands out copies: patch them, then store them back.
    kernels = cached.descriptor.kernels
    for index in cached.input_address_kernels:
        kernels[index].common_runtime_args[0] = input_tensor.buffer_address()
    for index in cached.output_address_kernels:
        kernels[index].common_runtime_args[0] = output_tensor.buffer_address()
    cached.descriptor.kernels = kernels
    return cached.descriptor, cached.plan
