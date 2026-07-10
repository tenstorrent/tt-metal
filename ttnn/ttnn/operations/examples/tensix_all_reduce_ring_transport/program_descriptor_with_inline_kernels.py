# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ablate synchronization and payload costs in a rectangular Tensix ring."""

from dataclasses import dataclass

import ttnn

CB_SCRATCH = 0
SEM_PROGRESS = 0

VARIANTS = (
    "semaphore_only",
    "payload_ring",
    "semaphore_only_noc1",
    "payload_ring_noc1",
)


@dataclass(frozen=True)
class GroupLayout:
    group_shape: tuple[int, int]
    groups: tuple[tuple[tuple[int, int], ...], ...]
    active_cores: tuple[tuple[int, int], ...]
    core_ranges: ttnn.CoreRangeSet

    @property
    def group_size(self):
        return self.group_shape[0] * self.group_shape[1]


def build_group_layout(device, group_shape, num_groups):
    if group_shape is None:
        grid = device.compute_with_storage_grid_size()
        group_shape = (1, grid.x) if grid.x >= 2 else (grid.y, 1)
    if len(group_shape) != 2:
        raise ValueError(f"group_shape must be (rows, cols), got {group_shape!r}")
    rows, cols = (int(group_shape[0]), int(group_shape[1]))
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError(f"group_shape must contain at least two cores, got {(rows, cols)}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be positive, got {num_groups}")

    grid = device.compute_with_storage_grid_size()
    groups_across = grid.x // cols
    groups_down = grid.y // rows
    capacity = groups_across * groups_down
    if groups_across == 0 or groups_down == 0 or num_groups > capacity:
        raise ValueError(f"cannot place {num_groups} groups of {rows}x{cols}; capacity is {capacity}")

    groups = []
    ranges = []
    for group_index in range(num_groups):
        gx = (group_index % groups_across) * cols
        gy = (group_index // groups_across) * rows
        ring = []
        for y in range(rows):
            xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
            ring.extend((gx + x, gy + y) for x in xs)
        groups.append(tuple(ring))
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(gx, gy), ttnn.CoreCoord(gx + cols - 1, gy + rows - 1)))
    active = tuple(core for group in groups for core in group)
    return GroupLayout((rows, cols), tuple(groups), active, ttnn.CoreRangeSet(ranges))


def create_sharded_memory_config(device, group_shape, num_groups, num_tiles):
    layout = build_group_layout(device, group_shape, num_groups)
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE),
        core_grid=layout.core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


_SEMAPHORE_ONLY_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t group_size = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(3);

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t next_x = get_arg_val<uint32_t>(2);
    const uint32_t next_y = get_arg_val<uint32_t>(3);

    Semaphore<> progress(progress_sem_id);
    Noc noc;
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t step = 1; step < group_size; ++step) {
            progress.up(noc, next_x, next_y, 1);
            progress.wait_min(iter * (group_size - 1) + step);
        }
    }

    noc_async_read(
        get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
        output_addr,
        payload_bytes);
    noc_async_read_barrier();
}
"""


_PAYLOAD_RING_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t group_size = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(4);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t next_x = get_arg_val<uint32_t>(2);
    const uint32_t next_y = get_arg_val<uint32_t>(3);

    CircularBuffer scratch(cb_scratch);
    scratch.reserve_back(2 * num_tiles);
    const uint32_t scratch_base = scratch.get_write_ptr();
    Semaphore<> progress(progress_sem_id);
    Noc noc;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t step = 1; step < group_size; ++step) {
            const uint32_t src_addr = step == 1
                ? input_addr
                : scratch_base + ((step - 1) & 1) * payload_bytes;
            const uint32_t dst_addr = scratch_base + (step & 1) * payload_bytes;
            noc_async_write(src_addr, get_noc_addr(next_x, next_y, dst_addr), payload_bytes);
            noc_async_write_barrier();
            progress.up(noc, next_x, next_y, 1);
            progress.wait_min(iter * (group_size - 1) + step);
        }
    }

    noc_async_read(
        get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
        output_addr,
        payload_bytes);
    noc_async_read_barrier();
}
"""


def _virtual_coords(device, cores):
    coords = []
    for x, y in cores:
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        coords.append((core.x, core.y))
    return coords


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    group_shape,
    num_groups,
    num_tiles,
    kernel_iters=1,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if num_tiles < 1 or kernel_iters < 1:
        raise ValueError("num_tiles and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("tensix_all_reduce_ring_transport supports bfloat16 tensors")

    layout = build_group_layout(input_tensor.device(), group_shape, num_groups)
    expected_shape = [len(layout.active_cores) * ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_shape or list(output_tensor.shape) != expected_shape:
        raise ValueError(f"input and output shape must be {expected_shape}")

    runtime_args = ttnn.RuntimeArgs()
    for group in layout.groups:
        virtual = _virtual_coords(input_tensor.device(), group)
        for index, (x, y) in enumerate(group):
            next_x, next_y = virtual[(index + 1) % layout.group_size]
            runtime_args[x][y] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                next_x,
                next_y,
            ]

    page_bytes = input_tensor.buffer_aligned_page_size()
    semaphore_only = variant in ("semaphore_only", "semaphore_only_noc1")
    use_noc1 = variant in ("semaphore_only_noc1", "payload_ring_noc1")
    if semaphore_only:
        source = _SEMAPHORE_ONLY_KERNEL
        compile_time_args = [layout.group_size, kernel_iters, SEM_PROGRESS, num_tiles * page_bytes]
        cbs = []
    else:
        source = _PAYLOAD_RING_KERNEL
        compile_time_args = [
            CB_SCRATCH,
            layout.group_size,
            num_tiles,
            page_bytes,
            kernel_iters,
            SEM_PROGRESS,
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * num_tiles * page_bytes,
                core_ranges=layout.core_ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_SCRATCH, input_tensor.dtype, page_bytes)],
            )
        ]

    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=layout.core_ranges,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.WriterConfigDescriptor() if use_noc1 else ttnn.ReaderConfigDescriptor(),
    )
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0)]
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=semaphores, cbs=cbs)


def ring_transport(
    input_tensor,
    *,
    variant="payload_ring",
    group_shape=None,
    num_groups=1,
    num_tiles=6,
    kernel_iters=1,
):
    """Run a ring cost component and return an identity copy for correctness."""
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output,
        variant=variant,
        group_shape=group_shape,
        num_groups=num_groups,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
