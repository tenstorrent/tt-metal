# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rectangular, on-chip all-reduce algorithm bake-off.

Each active Tensix core owns ``num_tiles`` bf16 tiles in sharded L1.  Cores are
partitioned into equal rectangular groups, and every variant computes the
elementwise sum of a group's shards into every member's output shard.  Inputs,
outputs, arithmetic, group placement, and NoC are held fixed; only the
collective communication/work-distribution algorithm changes.
"""

from dataclasses import dataclass

import ttnn

CB_GATHER = 0
CB_PARTIAL = 3
CB_OUTPUT = 16

SEM_PROGRESS = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2

VARIANTS = (
    "ring_push",
    "ring_pull",
    "unicast_all_gather",
    "mcast_all_gather",
    "reduce_root_mcast",
    "two_phase_reduce_mcast",
)


@dataclass(frozen=True)
class Group:
    index: int
    origin: tuple[int, int]
    cores: tuple[tuple[int, int], ...]
    ring_cores: tuple[tuple[int, int], ...]

    @property
    def root(self):
        x, y = self.cores[-1]
        return ttnn.CoreCoord(x, y)

    @property
    def core_range_set(self):
        x0, y0 = self.origin
        xs = [x for x, _ in self.cores]
        ys = [y for _, y in self.cores]
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(max(xs), max(ys)))])


@dataclass(frozen=True)
class GroupLayout:
    group_shape: tuple[int, int]
    groups: tuple[Group, ...]
    active_cores: tuple[tuple[int, int], ...]
    core_ranges: ttnn.CoreRangeSet

    @property
    def group_size(self):
        rows, cols = self.group_shape
        return rows * cols

    @property
    def num_cores(self):
        return len(self.active_cores)


def build_group_layout(device, group_shape, num_groups):
    """Pack equal rectangular groups row-major into the worker grid."""
    if group_shape is None:
        grid = device.compute_with_storage_grid_size()
        group_shape = (1, grid.x) if grid.x >= 2 else (grid.y, 1)
    if len(group_shape) != 2:
        raise ValueError(f"group_shape must be (rows, cols), got {group_shape!r}")
    rows, cols = (int(group_shape[0]), int(group_shape[1]))
    if rows < 1 or cols < 1:
        raise ValueError(f"group_shape dimensions must be positive, got {(rows, cols)}")
    if rows * cols < 2:
        raise ValueError("all-reduce groups need at least two cores")
    if num_groups < 1:
        raise ValueError(f"num_groups must be positive, got {num_groups}")

    grid = device.compute_with_storage_grid_size()
    groups_across = grid.x // cols
    groups_down = grid.y // rows
    capacity = groups_across * groups_down
    if groups_across == 0 or groups_down == 0 or num_groups > capacity:
        raise ValueError(
            f"cannot place {num_groups} groups of {rows}x{cols} on the {grid.y}x{grid.x} worker grid; "
            f"capacity is {capacity}"
        )

    groups = []
    ranges = []
    for group_index in range(num_groups):
        gx = (group_index % groups_across) * cols
        gy = (group_index // groups_across) * rows
        cores = tuple((gx + x, gy + y) for y in range(rows) for x in range(cols))
        ring = []
        for y in range(rows):
            xs = range(cols) if y % 2 == 0 else range(cols - 1, -1, -1)
            ring.extend((gx + x, gy + y) for x in xs)
        groups.append(Group(group_index, (gx, gy), cores, tuple(ring)))
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(gx, gy), ttnn.CoreCoord(gx + cols - 1, gy + rows - 1)))

    # HEIGHT sharding traverses each CoreRange in range-set order, row-major within a range.
    active = tuple(core for group in groups for core in group.cores)
    return GroupLayout((rows, cols), tuple(groups), active, ttnn.CoreRangeSet(ranges))


def create_sharded_memory_config(device, group_shape, num_groups, num_tiles):
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    layout = build_group_layout(device, group_shape, num_groups)
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE),
        core_grid=layout.core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


_REDUCE_KERNEL = r"""
#include <stdint.h>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(2);
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;

    CircularBuffer gather(cb_gather);
    CircularBuffer output(cb_output);
    binary_op_init_common(cb_gather, cb_gather, cb_output);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        gather.wait_front(num_blocks * num_tiles);
        for (uint32_t tile_base = 0; tile_base < num_tiles; tile_base += max_dst_tiles) {
            const uint32_t remaining = num_tiles - tile_base;
            const uint32_t dst_tiles = remaining < max_dst_tiles ? remaining : max_dst_tiles;
            output.reserve_back(dst_tiles);
            tile_regs_acquire();

            uint32_t first_pair = 0;
            if (num_blocks & 1) {
                copy_tile_to_dst_init_short(cb_gather);
                for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                    copy_tile(cb_gather, tile_base + tile, tile);
                }
                first_pair = 1;
            }

            if (num_blocks > 1) {
                add_tiles_init(cb_gather, cb_gather, true);
                for (uint32_t block = first_pair; block < num_blocks; block += 2) {
                    for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                        const uint32_t lhs = block * num_tiles + tile_base + tile;
                        const uint32_t rhs = (block + 1) * num_tiles + tile_base + tile;
                        add_tiles(cb_gather, cb_gather, lhs, rhs, tile);
                    }
                }
            }

            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                pack_tile(tile, cb_output);
            }
            tile_regs_release();
            output.push_back(dst_tiles);
        }
        gather.pop_front(num_blocks * num_tiles);
    }
}
"""


_RING_PUSH_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(7);

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t next_x = get_arg_val<uint32_t>(2);
    const uint32_t next_y = get_arg_val<uint32_t>(3);
    const uint32_t prev_x = get_arg_val<uint32_t>(4);
    const uint32_t prev_y = get_arg_val<uint32_t>(5);

    constexpr uint32_t payload_bytes = num_tiles * page_bytes;
    CircularBuffer gather(cb_gather);
    CircularBuffer output(cb_output);
    Semaphore<> progress(progress_sem_id);
    Semaphore<> consumed(consumed_sem_id);
    Noc noc;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t step = 1; step < group_size; ++step) {
            gather.reserve_back(2 * num_tiles);
            const uint32_t gather_addr = gather.get_write_ptr();
            uint32_t partial_addr = input_addr;
            if (step > 1) {
                output.wait_front(num_tiles);
                partial_addr = output.get_read_ptr();
                const uint32_t target = iter * (group_size - 1) + step - 1;
                consumed.up(noc, prev_x, prev_y, 1);
                consumed.wait_min(target);
            }

            noc_async_write(partial_addr, get_noc_addr(next_x, next_y, gather_addr), payload_bytes);
            noc_async_write_barrier();
            if (step > 1) {
                output.pop_front(num_tiles);
            }
            progress.up(noc, next_x, next_y, 1);
            progress.wait_min(iter * (group_size - 1) + step);

            noc_async_read(
                get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
                gather_addr + payload_bytes,
                payload_bytes);
            noc_async_read_barrier();
            gather.push_back(2 * num_tiles);
        }

        output.wait_front(num_tiles);
        consumed.up(noc, prev_x, prev_y, 1);
        consumed.wait_min((iter + 1) * (group_size - 1));
        output.pop_front(num_tiles);
    }
}
"""


_RING_PULL_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t consumed_sem_id = get_compile_time_arg_val(7);

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t next_x = get_arg_val<uint32_t>(2);
    const uint32_t next_y = get_arg_val<uint32_t>(3);
    const uint32_t prev_x = get_arg_val<uint32_t>(4);
    const uint32_t prev_y = get_arg_val<uint32_t>(5);

    constexpr uint32_t payload_bytes = num_tiles * page_bytes;
    CircularBuffer gather(cb_gather);
    CircularBuffer output(cb_output);
    Semaphore<> progress(progress_sem_id);
    Semaphore<> consumed(consumed_sem_id);
    Noc noc;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        for (uint32_t step = 1; step < group_size; ++step) {
            gather.reserve_back(2 * num_tiles);
            const uint32_t gather_addr = gather.get_write_ptr();
            uint32_t remote_partial_addr = input_addr;
            if (step > 1) {
                output.wait_front(num_tiles);
                remote_partial_addr = output.get_read_ptr();
                const uint32_t target = iter * (group_size - 2) + step - 1;
                progress.up(noc, next_x, next_y, 1);
                progress.wait_min(target);
            }

            noc_async_read(get_noc_addr(prev_x, prev_y, remote_partial_addr), gather_addr, payload_bytes);
            noc_async_read(
                get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
                gather_addr + payload_bytes,
                payload_bytes);
            noc_async_read_barrier();
            if (step > 1) {
                const uint32_t target = iter * (group_size - 2) + step - 1;
                consumed.up(noc, prev_x, prev_y, 1);
                consumed.wait_min(target);
                output.pop_front(num_tiles);
            }
            gather.push_back(2 * num_tiles);
        }

        output.wait_front(num_tiles);
        output.pop_front(num_tiles);
    }
}
"""


_UNICAST_ALL_GATHER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(6);

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t coords_base = 2;
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    CircularBuffer gather(cb_gather);
    CircularBuffer output(cb_output);
    Semaphore<> progress(progress_sem_id);
    Noc noc;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        gather.reserve_back(group_size * num_tiles);
        const uint32_t gather_addr = gather.get_write_ptr();
        noc_async_read(
            get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
            gather_addr + my_index * payload_bytes,
            payload_bytes);
        for (uint32_t peer = 0; peer < group_size; ++peer) {
            if (peer == my_index) continue;
            const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * peer);
            const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * peer + 1);
            noc_async_write(input_addr, get_noc_addr(x, y, gather_addr + my_index * payload_bytes), payload_bytes);
        }
        noc_async_read_barrier();
        noc_async_write_barrier();
        for (uint32_t peer = 0; peer < group_size; ++peer) {
            if (peer == my_index) continue;
            const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * peer);
            const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * peer + 1);
            progress.up(noc, x, y, 1);
        }
        progress.wait_min((iter + 1) * (group_size - 1));

        gather.push_back(group_size * num_tiles);
        output.wait_front(num_tiles);
        output.pop_front(num_tiles);
    }
}
"""


_MCAST_ALL_GATHER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t scalars = McastArgs<2, 2>::next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(scalars + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(scalars + 1);
    constexpr uint32_t group_size = get_compile_time_arg_val(scalars + 2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(scalars + 3);
    constexpr auto mc = McastArgs<2, 2, group_size>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    CircularBuffer gather(cb_gather);
    CircularBuffer output(cb_output);
    Noc noc;
    auto sender = mc.sender(noc);
    auto receiver = mc.receiver(noc);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        gather.reserve_back(group_size * num_tiles);
        const uint32_t gather_addr = gather.get_write_ptr();
        for (uint32_t round = 0; round < group_size; ++round) {
            const uint32_t dst = gather_addr + round * payload_bytes;
            if (round == my_index) {
                sender.send(input_addr, dst, payload_bytes);
            } else {
                receiver.receive(round);
            }
        }
        gather.push_back(group_size * num_tiles);
        output.wait_front(num_tiles);
        output.pop_front(num_tiles);
    }
}
"""


_REDUCE_ROOT_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t scalars = McastArgs<2, 5>::next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(scalars + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(scalars + 1);
    constexpr uint32_t group_size = get_compile_time_arg_val(scalars + 2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(scalars + 3);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(scalars + 4);
    constexpr auto mc = McastArgs<2, 5>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    CircularBuffer gather(cb_gather);
    Semaphore<> progress(progress_sem_id);
    Noc noc;

    if (my_index == group_size - 1) {
        CircularBuffer output(cb_output);
        auto sender = mc.sender(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            gather.reserve_back(group_size * num_tiles);
            const uint32_t gather_addr = gather.get_write_ptr();
            noc_async_read(
                get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr),
                gather_addr + my_index * payload_bytes,
                payload_bytes);
            noc_async_read_barrier();
            progress.wait_min((iter + 1) * (group_size - 1));
            gather.push_back(group_size * num_tiles);
            output.wait_front(num_tiles);
            sender.send(output_addr, output_addr, payload_bytes);
            output.pop_front(num_tiles);
        }
    } else {
        auto receiver = mc.receiver(noc);
        const uint32_t gather_addr = gather.get_write_ptr();
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            noc_async_write(
                input_addr,
                get_noc_addr(root_x, root_y, gather_addr + my_index * payload_bytes),
                payload_bytes);
            noc_async_write_barrier();
            progress.up(noc, root_x, root_y, 1);
            receiver.receive();
        }
    }
}
"""


_TWO_PHASE_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t scalars = McastArgs<2, 5>::next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(scalars + 0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(scalars + 1);
    constexpr uint32_t group_size = get_compile_time_arg_val(scalars + 2);
    constexpr uint32_t num_workers = get_compile_time_arg_val(scalars + 3);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(scalars + 4);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(scalars + 5);
    constexpr uint32_t coords_base = 5;
    constexpr uint32_t mcast_rt_base = coords_base + 2 * group_size;
    constexpr auto mc = McastArgs<2, mcast_rt_base>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    Semaphore<> progress(progress_sem_id);
    Noc noc;

    if (my_index == group_size - 1) {
        auto sender = mc.sender(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            progress.wait_min((iter + 1) * num_workers);
            sender.send(output_addr, output_addr, payload_bytes);
        }
    } else {
        if (my_index < num_workers) {
            CircularBuffer gather(cb_gather);
            CircularBuffer partial(cb_partial);
            const uint32_t assigned = (num_tiles + num_workers - 1 - my_index) / num_workers;
            for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
                gather.reserve_back(group_size * assigned);
                const uint32_t gather_addr = gather.get_write_ptr();
                for (uint32_t contributor = 0; contributor < group_size; ++contributor) {
                    const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * contributor);
                    const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * contributor + 1);
                    for (uint32_t local = 0; local < assigned; ++local) {
                        const uint32_t tile = my_index + local * num_workers;
                        noc_async_read(
                            get_noc_addr(x, y, input_addr + tile * page_bytes),
                            gather_addr + (contributor * assigned + local) * page_bytes,
                            page_bytes);
                    }
                }
                noc_async_read_barrier();
                gather.push_back(group_size * assigned);

                partial.wait_front(assigned);
                const uint32_t partial_addr = partial.get_read_ptr();
                for (uint32_t local = 0; local < assigned; ++local) {
                    const uint32_t tile = my_index + local * num_workers;
                    noc_async_write(
                        partial_addr + local * page_bytes,
                        get_noc_addr(root_x, root_y, output_addr + tile * page_bytes),
                        page_bytes);
                }
                noc_async_write_barrier();
                progress.up(noc, root_x, root_y, 1);
                partial.pop_front(assigned);
            }
        }
        auto receiver = mc.receiver(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            receiver.receive();
        }
    }
}
"""


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _normal_cb(cb_index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=dtype,
                page_size=page_bytes,
            )
        ],
    )


def _inline_kernel(source, core_ranges, compile_time_args, runtime_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=config,
    )


def _virtual_coords(device, cores):
    result = []
    for x, y in cores:
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        result.extend([core.x, core.y])
    return result


def _mcast_helpers(device, layout, *, rotating, sem_ids):
    helpers = []
    config = ttnn.McastConfig(rotating_sender=rotating, sem_ids=list(sem_ids))
    for group in layout.groups:
        helpers.append(ttnn.Mcast2D(device, group.core_range_set, group.root, config))
    return helpers


def _compute_kernel(core_ranges, runtime_by_core, output_cb=CB_OUTPUT):
    runtime_args = ttnn.RuntimeArgs()
    for (x, y), values in runtime_by_core.items():
        runtime_args[x][y] = values
    return _inline_kernel(
        _REDUCE_KERNEL,
        core_ranges,
        [CB_GATHER, output_cb],
        runtime_args,
        ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )


def _create_ring_descriptor(input_tensor, output_tensor, layout, variant, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = {}
    for group in layout.groups:
        virtual_ring = _virtual_coords(input_tensor.device(), group.ring_cores)
        for ring_index, (x, y) in enumerate(group.ring_cores):
            prev_index = (ring_index - 1) % group_size
            next_index = (ring_index + 1) % group_size
            dataflow_rt[x][y] = [
                input_tensor.buffer_address(),
                ring_index,
                virtual_ring[2 * next_index],
                virtual_ring[2 * next_index + 1],
                virtual_ring[2 * prev_index],
                virtual_ring[2 * prev_index + 1],
            ]
            compute_rt[(x, y)] = [2, num_tiles, kernel_iters * (group_size - 1)]

    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, 4 * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0)]
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=layout.core_ranges, initial_value=0))
    source = _RING_PUSH_KERNEL if variant == "ring_push" else _RING_PULL_KERNEL
    compile_time_args = [
        CB_GATHER,
        CB_OUTPUT,
        num_tiles,
        page_bytes,
        group_size,
        kernel_iters,
        SEM_PROGRESS,
        SEM_MCAST_READY,
    ]
    dataflow = _inline_kernel(
        source,
        layout.core_ranges,
        compile_time_args,
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _compute_kernel(layout.core_ranges, compute_rt)
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


def _create_unicast_all_gather_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = {}
    for group in layout.groups:
        virtual = _virtual_coords(input_tensor.device(), group.cores)
        for index, (x, y) in enumerate(group.cores):
            dataflow_rt[x][y] = [input_tensor.buffer_address(), index] + virtual
            compute_rt[(x, y)] = [group_size, num_tiles, kernel_iters]

    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0)]
    dataflow = _inline_kernel(
        _UNICAST_ALL_GATHER_KERNEL,
        layout.core_ranges,
        [CB_GATHER, CB_OUTPUT, num_tiles, page_bytes, group_size, kernel_iters, SEM_PROGRESS],
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _compute_kernel(layout.core_ranges, compute_rt)
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


def _create_mcast_all_gather_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    helpers = _mcast_helpers(input_tensor.device(), layout, rotating=True, sem_ids=(SEM_PROGRESS, SEM_MCAST_READY))
    mcast_ct = list(helpers[0].compile_time_args())
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = {}
    for group, helper in zip(layout.groups, helpers):
        for index, (x, y) in enumerate(group.cores):
            core = ttnn.CoreCoord(x, y)
            dataflow_rt[x][y] = [input_tensor.buffer_address(), index] + list(helper.runtime_args(core))
            compute_rt[(x, y)] = [group_size, num_tiles, kernel_iters]

    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=layout.core_ranges, initial_value=0),
    ]
    dataflow = _inline_kernel(
        _MCAST_ALL_GATHER_KERNEL,
        layout.core_ranges,
        [CB_GATHER, CB_OUTPUT] + mcast_ct + [num_tiles, page_bytes, group_size, kernel_iters],
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _compute_kernel(layout.core_ranges, compute_rt)
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


def _create_reduce_root_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    helpers = _mcast_helpers(
        input_tensor.device(),
        layout,
        rotating=False,
        sem_ids=(SEM_MCAST_READY, SEM_MCAST_CONSUMED),
    )
    mcast_ct = list(helpers[0].compile_time_args())
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = {}
    roots = []
    for group, helper in zip(layout.groups, helpers):
        root_virtual = input_tensor.device().worker_core_from_logical_core(group.root)
        roots.append((group.root.x, group.root.y))
        for index, (x, y) in enumerate(group.cores):
            core = ttnn.CoreCoord(x, y)
            dataflow_rt[x][y] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                index,
                root_virtual.x,
                root_virtual.y,
            ] + list(helper.runtime_args(core))
        compute_rt[(group.root.x, group.root.y)] = [group_size, num_tiles, kernel_iters]

    root_ranges = _core_range_set(roots)
    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=layout.core_ranges, initial_value=0),
    ]
    dataflow = _inline_kernel(
        _REDUCE_ROOT_KERNEL,
        layout.core_ranges,
        [CB_GATHER, CB_OUTPUT] + mcast_ct + [num_tiles, page_bytes, group_size, kernel_iters, SEM_PROGRESS],
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _compute_kernel(root_ranges, compute_rt)
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


def _create_two_phase_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    num_workers = min(num_tiles, group_size - 1)
    max_assigned = (num_tiles + num_workers - 1) // num_workers
    helpers = _mcast_helpers(
        input_tensor.device(),
        layout,
        rotating=False,
        sem_ids=(SEM_MCAST_READY, SEM_MCAST_CONSUMED),
    )
    mcast_ct = list(helpers[0].compile_time_args())
    dataflow_rt = ttnn.RuntimeArgs()
    compute_rt = {}
    worker_cores = []
    for group, helper in zip(layout.groups, helpers):
        virtual = _virtual_coords(input_tensor.device(), group.cores)
        root_virtual = input_tensor.device().worker_core_from_logical_core(group.root)
        for index, (x, y) in enumerate(group.cores):
            core = ttnn.CoreCoord(x, y)
            dataflow_rt[x][y] = (
                [
                    input_tensor.buffer_address(),
                    output_tensor.buffer_address(),
                    index,
                    root_virtual.x,
                    root_virtual.y,
                ]
                + virtual
                + list(helper.runtime_args(core))
            )
            if index < num_workers:
                assigned = (num_tiles + num_workers - 1 - index) // num_workers
                worker_cores.append((x, y))
                compute_rt[(x, y)] = [group_size, assigned, kernel_iters]

    worker_ranges = _core_range_set(worker_cores)
    cbs = [
        _normal_cb(CB_GATHER, worker_ranges, group_size * max_assigned, page_bytes, input_tensor.dtype),
        _normal_cb(CB_PARTIAL, worker_ranges, max_assigned, page_bytes, output_tensor.dtype),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=layout.core_ranges, initial_value=0),
    ]
    dataflow = _inline_kernel(
        _TWO_PHASE_KERNEL,
        layout.core_ranges,
        [CB_GATHER, CB_PARTIAL]
        + mcast_ct
        + [num_tiles, page_bytes, group_size, num_workers, kernel_iters, SEM_PROGRESS],
        dataflow_rt,
        ttnn.ReaderConfigDescriptor(),
    )
    compute = _compute_kernel(worker_ranges, compute_rt, output_cb=CB_PARTIAL)
    return ttnn.ProgramDescriptor(kernels=[dataflow, compute], semaphores=semaphores, cbs=cbs)


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
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    if kernel_iters < 1:
        raise ValueError(f"kernel_iters must be positive, got {kernel_iters}")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("tensix_all_reduce supports bfloat16 inputs and outputs")
    if input_tensor.layout != ttnn.TILE_LAYOUT or output_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("tensix_all_reduce requires TILE_LAYOUT inputs and outputs")

    layout = build_group_layout(input_tensor.device(), group_shape, num_groups)
    expected_shape = [layout.num_cores * ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_shape or list(output_tensor.shape) != expected_shape:
        raise ValueError(
            f"input/output shape must be {expected_shape} for {layout.num_cores} cores and {num_tiles} tiles"
        )
    page_bytes = input_tensor.buffer_aligned_page_size()

    if variant in ("ring_push", "ring_pull"):
        return _create_ring_descriptor(
            input_tensor, output_tensor, layout, variant, num_tiles, page_bytes, kernel_iters
        )
    if variant == "unicast_all_gather":
        return _create_unicast_all_gather_descriptor(
            input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters
        )
    if variant == "mcast_all_gather":
        return _create_mcast_all_gather_descriptor(
            input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters
        )
    if variant == "reduce_root_mcast":
        return _create_reduce_root_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters)
    return _create_two_phase_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters)


def all_reduce(
    input_tensor,
    *,
    variant="mcast_all_gather",
    group_shape=None,
    num_groups=1,
    num_tiles=6,
    kernel_iters=1,
):
    """All-reduce each rectangular group and replicate its sum on every member."""
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        group_shape=group_shape,
        num_groups=num_groups,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
