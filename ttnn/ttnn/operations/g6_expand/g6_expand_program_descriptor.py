# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program descriptor: plan (work split + block sizes), CB, kernels, column multicast.

Index space: pattern p = input page (image * Wt + col), replica r = output tile row within an image.
Grid column gc owns a contiguous pattern range, grid row gr a contiguous replica range. Core (gc, 0)
is its column's injector; cores (gc, gr > 0) receive the column's pattern block by multicast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# --- Blocking-model knobs (op_design.md, Blocking Model) -------------------------------------
# None = the device grid's extent on that axis.
MAX_PATTERN_GROUPS = None
MAX_REPLICA_GROUPS = None
PATTERN_CB_DEPTH_BLOCKS = 1
L1_OP_BUDGET_BYTES = 1_347_776
# Multicast semaphores and allocation alignment.
L1_RESERVED_BYTES = 4096

CB_PATTERN_TILES = 0
MCAST_PREFIX = "pattern_mcast"


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Plan:
    grid: tuple
    tensor_pattern_wt: int
    tensor_pattern_tiles: int
    tensor_replica_rows: int
    pattern_tiles_per_group: int
    pattern_groups: int
    replica_rows_per_group: int
    replica_groups: int
    pattern_cb_depth_blocks: int
    block_pattern_tiles: int

    @property
    def num_cores(self):
        return self.pattern_groups * self.replica_groups

    def core_patterns(self, gc):
        """(pattern_start, core_pattern_tiles) of grid column gc."""
        start = gc * self.pattern_tiles_per_group
        return start, min(self.tensor_pattern_tiles, start + self.pattern_tiles_per_group) - start

    def core_replicas(self, gr):
        """(replica_start, core_replica_rows) of grid row gr."""
        start = gr * self.replica_rows_per_group
        return start, min(self.tensor_replica_rows, start + self.replica_rows_per_group) - start

    def num_blocks(self, gc):
        return _div_up(self.core_patterns(gc)[1], self.block_pattern_tiles)

    @staticmethod
    def role(gc, gr):
        return "injector" if gr == 0 else "receiver"


def select_plan(
    shape,
    sizes,
    grid,
    *,
    tile_bytes=2048,
    max_pattern_groups=MAX_PATTERN_GROUPS,
    max_replica_groups=MAX_REPLICA_GROUPS,
    pattern_cb_depth_blocks=PATTERN_CB_DEPTH_BLOCKS,
):
    """Plan for x of `shape` expanded to `sizes` (-1 keeps the dim) on a `grid` = (grid_x, grid_y)."""
    out_shape = [int(d) if int(s) == -1 else int(s) for d, s in zip(shape, sizes)]
    grid_x, grid_y = int(grid[0]), int(grid[1])
    images = out_shape[0] * out_shape[1]
    tensor_pattern_wt = _div_up(out_shape[-1], TILE)
    tensor_pattern_tiles = images * tensor_pattern_wt
    tensor_replica_rows = _div_up(out_shape[-2], TILE)

    pattern_cap = grid_x if max_pattern_groups is None else max_pattern_groups
    replica_cap = grid_y if max_replica_groups is None else max_replica_groups
    pattern_tiles_per_group = _div_up(tensor_pattern_tiles, min(tensor_pattern_tiles, pattern_cap))
    pattern_groups = _div_up(tensor_pattern_tiles, pattern_tiles_per_group)
    replica_rows_per_group = _div_up(tensor_replica_rows, min(tensor_replica_rows, replica_cap))
    replica_groups = _div_up(tensor_replica_rows, replica_rows_per_group)

    max_block_pattern_tiles = (L1_OP_BUDGET_BYTES - L1_RESERVED_BYTES) // (pattern_cb_depth_blocks * tile_bytes)
    block_pattern_tiles = min(pattern_tiles_per_group, max_block_pattern_tiles)

    return Plan(
        grid=(grid_x, grid_y),
        tensor_pattern_wt=tensor_pattern_wt,
        tensor_pattern_tiles=tensor_pattern_tiles,
        tensor_replica_rows=tensor_replica_rows,
        pattern_tiles_per_group=pattern_tiles_per_group,
        pattern_groups=pattern_groups,
        replica_rows_per_group=replica_rows_per_group,
        replica_groups=replica_groups,
        pattern_cb_depth_blocks=pattern_cb_depth_blocks,
        block_pattern_tiles=block_pattern_tiles,
    )


@dataclass
class CachedProgram:
    plan: Plan
    descriptor: ttnn.ProgramDescriptor
    # Kernel descriptors in descriptor order. Runtime arg 0 of every core is a tensor address:
    # x for the readers, out for the writer. Each entry: (kernel, [(x, y) cores]).
    kernels: list
    input_address_kernels: list
    output_address_kernels: list


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def create_program_descriptor(input_tensor, output_tensor, plan):
    device = input_tensor.device()
    tile_bytes = input_tensor.buffer_aligned_page_size()
    if output_tensor.buffer_aligned_page_size() != tile_bytes:
        raise RuntimeError("g6_expand: input and output pages differ in size")

    last_x, last_y = plan.pattern_groups - 1, plan.replica_groups - 1
    rectangle = _crs(0, 0, last_x, last_y)
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    cb_pattern_tiles = ttnn.CBDescriptor(
        total_size=plan.pattern_cb_depth_blocks * plan.block_pattern_tiles * tile_bytes,
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
    reader_config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)

    def cores(rows):
        return [(gc, gr) for gc in range(plan.pattern_groups) for gr in rows]

    def reader_kernel(is_injector, rows):
        rt = ttnn.RuntimeArgs()
        for gc in range(plan.pattern_groups):
            pattern_start, core_pattern_tiles = plan.core_patterns(gc)
            for gr in rows:
                rt[gc][gr] = [x_addr, pattern_start, core_pattern_tiles]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
            core_ranges=_crs(0, rows[0], last_x, rows[-1]),
            compile_time_args=[
                CB_PATTERN_TILES,
                plan.block_pattern_tiles,
                tile_bytes,
                int(is_injector),
                *x_accessor_ct,
            ],
            runtime_args=rt,
            config=reader_config,
        )

    injector_reader = reader_kernel(True, [0])
    readers = [injector_reader]
    reader_cores = [cores([0])]
    if plan.replica_groups > 1:
        receiver_rows = list(range(1, plan.replica_groups))
        readers.append(reader_kernel(False, receiver_rows))
        reader_cores.append(cores(receiver_rows))

    writer_rt = ttnn.RuntimeArgs()
    for gc in range(plan.pattern_groups):
        pattern_start, core_pattern_tiles = plan.core_patterns(gc)
        for gr in range(plan.replica_groups):
            replica_start, core_replica_rows = plan.core_replicas(gr)
            writer_rt[gc][gr] = [out_addr, pattern_start, core_pattern_tiles, replica_start, core_replica_rows]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=rectangle,
        compile_time_args=[
            CB_PATTERN_TILES,
            plan.block_pattern_tiles,
            tile_bytes,
            plan.tensor_pattern_wt,
            plan.tensor_replica_rows,
            *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
        ],
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )

    descriptor = ttnn.ProgramDescriptor(cbs=[cb_pattern_tiles])
    if plan.replica_groups > 1:
        # One group per grid column (column-major order), fixed sender = the column's top core.
        pattern_mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
            rectangle,
            plan.replica_groups,
            ttnn.McastFixedSenderConfig(),
            ttnn.McastCoreOrder.ColumnMajor,
        )
        pattern_mcast.attach(descriptor, MCAST_PREFIX, readers, 0)
    else:
        ttnn.attach_absent_mcast(injector_reader, MCAST_PREFIX)

    kernels = [*readers, writer]
    descriptor.kernels = kernels
    return CachedProgram(
        plan=plan,
        descriptor=descriptor,
        kernels=kernels,
        input_address_kernels=list(zip(readers, reader_cores)),
        output_address_kernels=[(writer, cores(range(plan.replica_groups)))],
    )


def _patch_address(kernel, kernel_cores, address):
    runtime_args = kernel.runtime_args
    for x, y in kernel_cores:
        runtime_args[x][y][0] = address


def patch_buffer_addresses(cached, input_tensor, output_tensor):
    """Point a cached descriptor at this call's tensors; nothing else changes between calls."""
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for kernel, kernel_cores in cached.input_address_kernels:
        _patch_address(kernel, kernel_cores, x_addr)
    for kernel, kernel_cores in cached.output_address_kernels:
        _patch_address(kernel, kernel_cores, out_addr)
    cached.descriptor.kernels = cached.kernels
