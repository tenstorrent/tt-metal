# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row-local all-to-all broadcast: per-peer unicast writes versus multicast."""

from dataclasses import dataclass

import ttnn

SEM_UNICAST_PROGRESS = 0

VARIANTS = ("unicast", "mcast")


@dataclass(frozen=True)
class RowLayout:
    num_rows: int
    row_width: int
    active_cores: tuple[tuple[int, int], ...]
    core_ranges: ttnn.CoreRangeSet

    @property
    def num_cores(self):
        return self.num_rows * self.row_width


def build_row_layout(device, num_rows=None, row_width=None):
    grid = device.compute_with_storage_grid_size()
    num_rows = grid.y if num_rows is None else int(num_rows)
    row_width = grid.x if row_width is None else int(row_width)
    if num_rows < 1 or num_rows > grid.y:
        raise ValueError(f"num_rows must be in [1, {grid.y}], got {num_rows}")
    if row_width < 2 or row_width > grid.x:
        raise ValueError(f"row_width must be in [2, {grid.x}], got {row_width}")

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(row_width - 1, num_rows - 1))])
    active_cores = tuple((x, y) for y in range(num_rows) for x in range(row_width))
    return RowLayout(num_rows, row_width, active_cores, core_ranges)


def create_sharded_memory_config(device, num_rows, row_width, shard_tiles):
    if shard_tiles < 1:
        raise ValueError(f"shard_tiles must be positive, got {shard_tiles}")
    layout = build_row_layout(device, num_rows, row_width)
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, shard_tiles * ttnn.TILE_SIZE),
        core_grid=layout.core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


_UNICAST_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t row_width = get_compile_time_arg_val(0);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_writes = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t write_bytes = payload_bytes / num_writes;

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t coords_base = 2;

    Semaphore<> progress(progress_sem_id);
    Noc noc;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t round = 0; round < row_width; ++round) {
            if (round == my_index) {
                const uint32_t payload_addr = state_addr + my_index * payload_bytes;
                for (uint32_t dispatch = 0; dispatch < num_writes; ++dispatch) {
                    const uint32_t chunk_addr = payload_addr + dispatch * write_bytes;
                    for (uint32_t peer = 0; peer < row_width; ++peer) {
                        if (peer == my_index) {
                            continue;
                        }
                        const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * peer);
                        const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * peer + 1);
                        noc_async_write(chunk_addr, get_noc_addr(x, y, chunk_addr), write_bytes);
                    }
                    noc_async_write_barrier();

                    for (uint32_t peer = 0; peer < row_width; ++peer) {
                        if (peer == my_index) {
                            continue;
                        }
                        const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * peer);
                        const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * peer + 1);
                        progress.up(noc, x, y, 1);
                    }
                }
            } else {
                const uint32_t completed_rounds_before =
                    round - static_cast<uint32_t>(my_index < round);
                for (uint32_t dispatch = 0; dispatch < num_writes; ++dispatch) {
                    const uint32_t target =
                        iter * (row_width - 1) * num_writes +
                        completed_rounds_before * num_writes +
                        dispatch + 1;
                    progress.wait_min(target);
                }
            }
        }
    }
}
"""


_MCAST_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t row_width = get_compile_time_arg_val(0);
    constexpr auto row_mcast = McastArgs<1, 2, row_width>();
    constexpr uint32_t scalars = row_mcast.next_compile_time_args_offset();
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(scalars + 0);
    constexpr uint32_t num_writes = get_compile_time_arg_val(scalars + 1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(scalars + 2);
    constexpr uint32_t write_bytes = payload_bytes / num_writes;

    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);

    Noc noc;
    auto sender = row_mcast.sender(noc);
    auto receiver = row_mcast.receiver(noc);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t round = 0; round < row_width; ++round) {
            const uint32_t payload_addr = state_addr + round * payload_bytes;
            for (uint32_t dispatch = 0; dispatch < num_writes; ++dispatch) {
                const uint32_t chunk_addr = payload_addr + dispatch * write_bytes;
                if (round == my_index) {
                    // src == dst keeps the sender out of the receiver set: only peers receive.
                    sender.send(chunk_addr, chunk_addr, write_bytes);
                } else {
                    receiver.receive(round);
                }
            }
        }
    }
}
"""


def _virtual_row_coords(device, y, row_width):
    coords = []
    for x in range(row_width):
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        coords.extend([core.x, core.y])
    return coords


def _kernel(source, core_ranges, compile_time_args, runtime_args):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    num_rows=None,
    row_width=None,
    num_tiles=1,
    num_writes=1,
    kernel_iters=1,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if num_tiles < 1 or num_writes < 1 or kernel_iters < 1:
        raise ValueError("num_tiles, num_writes, and kernel_iters must be positive")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("row_unicast_vs_mcast supports bfloat16 tensors")
    if input_tensor.layout != ttnn.TILE_LAYOUT or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("row_unicast_vs_mcast requires TILE_LAYOUT tensors")

    device = input_tensor.device()
    layout = build_row_layout(device, num_rows, row_width)
    expected_shape = [
        layout.num_cores * ttnn.TILE_SIZE,
        layout.row_width * num_tiles * ttnn.TILE_SIZE,
    ]
    if list(input_tensor.shape) != expected_shape or list(output_tensor.shape) != expected_shape:
        raise ValueError(f"input and output shape must be {expected_shape}")
    if input_tensor.buffer_address() != output_tensor.buffer_address():
        raise ValueError("row_unicast_vs_mcast requires an in-place state tensor")

    payload_bytes = num_tiles * input_tensor.buffer_aligned_page_size()
    if payload_bytes % num_writes != 0 or (payload_bytes // num_writes) % 16 != 0:
        raise ValueError("num_writes must split the payload into 16-byte-aligned equal chunks")
    runtime_args = ttnn.RuntimeArgs()

    if variant == "unicast":
        for y in range(layout.num_rows):
            row_coords = _virtual_row_coords(device, y, layout.row_width)
            for x in range(layout.row_width):
                runtime_args[x][y] = [
                    output_tensor.buffer_address(),
                    x,
                ] + row_coords
        kernel = _kernel(
            _UNICAST_KERNEL,
            layout.core_ranges,
            [layout.row_width, payload_bytes, num_writes, kernel_iters, SEM_UNICAST_PROGRESS],
            runtime_args,
        )
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=SEM_UNICAST_PROGRESS,
                core_ranges=layout.core_ranges,
                initial_value=0,
            )
        ]
    else:
        row_mcast = ttnn.Mcast1D(
            device,
            layout.core_ranges,
            ttnn.Mcast1DShape.PerRow,
            0,
            ttnn.McastConfig(rotating_sender=True),
        )
        for x, y in layout.active_cores:
            runtime_args[x][y] = [
                output_tensor.buffer_address(),
                x,
            ] + list(row_mcast.runtime_args(ttnn.CoreCoord(x, y)))
        kernel = _kernel(
            _MCAST_KERNEL,
            layout.core_ranges,
            [layout.row_width] + list(row_mcast.compile_time_args()) + [payload_bytes, num_writes, kernel_iters],
            runtime_args,
        )
        semaphores = list(row_mcast.owned_semaphores())

    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=semaphores, cbs=[])


def row_all_gather(
    input_tensor,
    *,
    variant="mcast",
    num_rows=None,
    row_width=None,
    num_tiles=1,
    num_writes=1,
    kernel_iters=1,
):
    """Broadcast every core's resident L1 payload to the other members of its row."""
    layout = build_row_layout(input_tensor.device(), num_rows, row_width)
    descriptor = create_program_descriptor(
        input_tensor,
        input_tensor,
        variant=variant,
        num_rows=layout.num_rows,
        row_width=layout.row_width,
        num_tiles=num_tiles,
        num_writes=num_writes,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, input_tensor], descriptor)
