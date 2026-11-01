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
CB_STAGE2 = 4
CB_OUTPUT = 16

SEM_PROGRESS = 0
SEM_MCAST_READY = 1
# Multicast attachment allocates its data-ready and consumer-ready semaphores from this id (1 and 2).
SEM_MCAST_BASE = SEM_MCAST_READY
SEM_STAGE2 = 3
# Shares id 3 with SEM_STAGE2: the tree reducer and the push reduce-scatter are
# different programs and neither declares the other's semaphore.
SEM_GATHER = 3

# Ordered BEST-FIRST. The first four are the ones worth running; the trailing three are
# MEASURED NULLS kept for the record (see README) and excluded from the default sweep, so a
# comparison baseline is never one of them.
VARIANTS = (
    "reduce_root_mcast",
    "reduce_scatter_mcast",
    "reduce_scatter_push",
    # Same as their twins above, but the write that a semaphore increment publishes is
    # fenced with a flush (writes left this core) instead of a barrier (writes acknowledged).
    "reduce_scatter_mcast_flush",
    "reduce_scatter_push_flush",
    "tree_reduce_mcast",
    "unicast_all_gather",
    # --- measured nulls, opt in with `--variant all` -------------------------
    "mcast_all_gather",
    "ring_pull",
    "ring_push",
)

# Consistently ~4x off the best on-chip reducer at every size measured. They are not bugs — they
# are the wrong shape for an ON-CHIP group (rings pay a hop per step; rotating-sender all-gather
# serializes G broadcast rounds) — and they stay in the file so nobody re-derives that by running
# them. Excluded from the default sweep and from the report's baseline.
MEASURED_NULLS = ("mcast_all_gather", "ring_pull", "ring_push")

# What `--variant` runs when not asked otherwise.
RECOMMENDED = tuple(v for v in VARIANTS if v not in MEASURED_NULLS)


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
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

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
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(6);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

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


# Fence between data writes and the semaphore increment that tells the destination they
# landed. Default (the barrier twins): wait until every write is ACKNOWLEDGED, i.e. one full
# round trip to the destination and back. SIGNAL_AFTER_FLUSH (the `_flush` variants): wait
# only until the writes have LEFT this core, then issue the increment. That is safe because
# the increment and the writes travel on the same NoC and the same static unicast-write VC
# (`NOC_UNICAST_WRITE_VC`, the default of both `noc_async_write` and `Semaphore::up`) to the
# same destination, and writes on the same NoC + VC are delivered in program order even from
# different command buffers. A flush is not completion, so the flush variants retire the
# outstanding write acknowledgements once, at kernel exit.
_SIGNAL_FENCE = r"""
FORCE_INLINE void fence_before_signal() {
#if defined(SIGNAL_AFTER_FLUSH)
    noc_async_writes_flushed();
#else
    noc_async_write_barrier();
#endif
}

FORCE_INLINE void retire_writes_at_exit() {
#if defined(SIGNAL_AFTER_FLUSH)
    noc_async_write_barrier();
#endif
}
"""


_REDUCE_SCATTER_KERNEL = (
    r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;
"""
    + _SIGNAL_FENCE
    + r"""

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_workers = get_compile_time_arg_val(5);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t coords_base = 5;
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;
    // The tile split across workers is ragged: worker i owns either `q` or `q+1`
    // tiles (round-robin, stride num_workers). Both CBs are allocated at the
    // UNIFORM `max_assigned`, and a CB's capacity must be an exact multiple of its
    // push/pop quantum, so every worker must push/pop `max_assigned` even when its
    // own share is smaller - a ragged quantum wraps illegally and corrupts the
    // gather. The pad slots are neither read nor written back, so their contents
    // never reach the output.
    constexpr uint32_t max_assigned = (num_tiles + num_workers - 1) / num_workers;

    Semaphore<> progress(progress_sem_id);
    Noc noc;

    // The root (index group_size - 1) takes a worker share like everyone else and
    // only then multicasts. It writes its own partials to itself (NoC loopback) and
    // bumps its own progress semaphore, so the sender's wait_min still counts a
    // uniform one increment per worker per iteration.
    {
        if (my_index < num_workers) {
            CircularBuffer gather(cb_gather);
            CircularBuffer partial(cb_partial);
            const uint32_t assigned = (num_tiles + num_workers - 1 - my_index) / num_workers;
            for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
                gather.reserve_back(group_size * max_assigned);
                const uint32_t gather_addr = gather.get_write_ptr();
                for (uint32_t contributor = 0; contributor < group_size; ++contributor) {
                    const uint32_t x = get_arg_val<uint32_t>(coords_base + 2 * contributor);
                    const uint32_t y = get_arg_val<uint32_t>(coords_base + 2 * contributor + 1);
                    for (uint32_t local = 0; local < assigned; ++local) {
                        const uint32_t tile = my_index + local * num_workers;
                        noc_async_read(
                            get_noc_addr(x, y, input_addr + tile * page_bytes),
                            gather_addr + (contributor * max_assigned + local) * page_bytes,
                            page_bytes);
                    }
                }
                noc_async_read_barrier();
                gather.push_back(group_size * max_assigned);

                partial.wait_front(max_assigned);
                const uint32_t partial_addr = partial.get_read_ptr();
                for (uint32_t local = 0; local < assigned; ++local) {
                    const uint32_t tile = my_index + local * num_workers;
                    noc_async_write(
                        partial_addr + local * page_bytes,
                        get_noc_addr(root_x, root_y, output_addr + tile * page_bytes),
                        page_bytes);
                }
                fence_before_signal();
                progress.up(noc, root_x, root_y, 1);
                partial.pop_front(max_assigned);
            }
        }
    }

    if (my_index == group_size - 1) {
        auto sender = mc.sender(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            progress.wait_min((iter + 1) * num_workers);
            sender.send(output_addr, output_addr, payload_bytes);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            receiver.receive();
        }
    }
    retire_writes_at_exit();
}
"""
)


_GRID_REDUCE_COMPUTE_KERNEL = r"""
#include <stdint.h>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// Reduce `num_blocks` contiguous blocks (each `num_tiles` tiles) waiting in `in_cb` into a single
// `num_tiles` block pushed to `out_cb`.  Contributors are paired with FPU add_tiles accumulating in
// FP32 DST; an odd block count seeds DST with one copied block.  The live DST batch is the
// JIT-derived DEST_AUTO_LIMIT.
inline void reduce_blocks(uint32_t in_cb, uint32_t out_cb, uint32_t num_blocks, uint32_t num_tiles) {
    constexpr uint32_t max_dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
    CircularBuffer in(in_cb);
    CircularBuffer out(out_cb);

    in.wait_front(num_blocks * num_tiles);
    for (uint32_t tile_base = 0; tile_base < num_tiles; tile_base += max_dst_tiles) {
        const uint32_t remaining = num_tiles - tile_base;
        const uint32_t dst_tiles = remaining < max_dst_tiles ? remaining : max_dst_tiles;
        out.reserve_back(dst_tiles);
        tile_regs_acquire();

        uint32_t first_pair = 0;
        if (num_blocks & 1) {
            copy_tile_to_dst_init_short(in_cb);
            for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                copy_tile(in_cb, tile_base + tile, tile);
            }
            first_pair = 1;
        }

        if (num_blocks > 1) {
            add_tiles_init(in_cb, in_cb, true);
            for (uint32_t block = first_pair; block < num_blocks; block += 2) {
                for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
                    const uint32_t lhs = block * num_tiles + tile_base + tile;
                    const uint32_t rhs = (block + 1) * num_tiles + tile_base + tile;
                    add_tiles(in_cb, in_cb, lhs, rhs, tile);
                }
            }
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < dst_tiles; ++tile) {
            pack_tile(tile, out_cb);
        }
        tile_regs_release();
        out.push_back(dst_tiles);
    }
    in.pop_front(num_blocks * num_tiles);
}

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);

    const uint32_t stage1_blocks = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(2);
    const uint32_t is_root = get_arg_val<uint32_t>(3);
    const uint32_t stage2_blocks = get_arg_val<uint32_t>(4);

    // All CBs are the same bf16 tile format, so one init covers the packer/unpacker for both stages.
    binary_op_init_common(cb_gather, cb_gather, cb_partial);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // Stage 1: reduce this row's blocks into the row partial.
        reduce_blocks(cb_gather, cb_partial, stage1_blocks, num_tiles);
        // Stage 2 (root only): reduce the gathered row partials into the group sum.
        if (is_root) {
            reduce_blocks(cb_stage2, cb_output, stage2_blocks, num_tiles);
        }
    }
}
"""


# Same algorithm and same gather layout as _REDUCE_SCATTER_KERNEL, with the data
# moved in the opposite direction: contributors PUSH their tiles into the owning
# worker's gather buffer instead of each worker PULLING from every contributor.
#
# Two consequences worth measuring, both structural rather than incidental:
#
#  * Addressing. A pusher needs the DESTINATION's gather address, and it gets one
#    by using its own `get_write_ptr()` - which only resolves correctly if the CB
#    is allocated identically on every core that participates. So both CBs are
#    declared group-wide here, where the pull variant declares them on the worker
#    cores alone. When `num_workers == group_size` (any `num_tiles >= group_size`,
#    i.e. the whole regime where reduce-scatter is the fastest reducer) that is
#    the same set of cores and the L1 cost is IDENTICAL; only a short group with
#    idle non-workers pays extra.
#  * Handshake. The pull variant reads out of the input tensor, which nobody
#    writes, so a worker needs no notification that a contributor is ready. A push
#    has to tell the worker its data landed: every contributor bumps every
#    worker's gather semaphore after its write barrier, so the gather phase costs
#    `group_size * num_workers` extra semaphore increments per all-reduce.
_REDUCE_SCATTER_PUSH_KERNEL = (
    r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;
"""
    + _SIGNAL_FENCE
    + r"""

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t group_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_workers = get_compile_time_arg_val(5);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);
    constexpr uint32_t progress_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t coords_base = 5;
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;
    constexpr uint32_t max_assigned = (num_tiles + num_workers - 1) / num_workers;

    Semaphore<> progress(progress_sem_id);
    Semaphore<> gathered(gather_sem_id);
    Noc noc;

    CircularBuffer gather(cb_gather);
    CircularBuffer partial(cb_partial);
    // The gather CB holds exactly one round (group_size * max_assigned pages), so
    // its write pointer is a fixed address - which is what lets a remote pusher
    // target it, and what makes this local read of it a valid stand-in for every
    // worker's copy.
    const uint32_t gather_base = gather.get_write_ptr();
    const uint32_t assigned = (num_tiles + num_workers - 1 - my_index) / num_workers;

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // PUSH phase - every core is a contributor. Walk the workers rather than
        // the tiles so each destination's pages are issued back to back.
        for (uint32_t worker = 0; worker < num_workers; ++worker) {
            const uint32_t wx = get_arg_val<uint32_t>(coords_base + 2 * worker);
            const uint32_t wy = get_arg_val<uint32_t>(coords_base + 2 * worker + 1);
            const uint32_t worker_assigned = (num_tiles + num_workers - 1 - worker) / num_workers;
            for (uint32_t local = 0; local < worker_assigned; ++local) {
                const uint32_t tile = worker + local * num_workers;
                noc_async_write(
                    input_addr + tile * page_bytes,
                    get_noc_addr(wx, wy, gather_base + (my_index * max_assigned + local) * page_bytes),
                    page_bytes);
            }
        }
        fence_before_signal();
        for (uint32_t worker = 0; worker < num_workers; ++worker) {
            const uint32_t wx = get_arg_val<uint32_t>(coords_base + 2 * worker);
            const uint32_t wy = get_arg_val<uint32_t>(coords_base + 2 * worker + 1);
            gathered.up(noc, wx, wy, 1);
        }

        // REDUCE phase - only the workers have compute to hand off to.
        if (my_index < num_workers) {
            gather.reserve_back(group_size * max_assigned);
            gathered.wait_min((iter + 1) * group_size);
            gather.push_back(group_size * max_assigned);

            partial.wait_front(max_assigned);
            const uint32_t partial_addr = partial.get_read_ptr();
            for (uint32_t local = 0; local < assigned; ++local) {
                const uint32_t tile = my_index + local * num_workers;
                noc_async_write(
                    partial_addr + local * page_bytes,
                    get_noc_addr(root_x, root_y, output_addr + tile * page_bytes),
                    page_bytes);
            }
            fence_before_signal();
            progress.up(noc, root_x, root_y, 1);
            partial.pop_front(max_assigned);
        }
    }

    if (my_index == group_size - 1) {
        auto sender = mc.sender(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            progress.wait_min((iter + 1) * num_workers);
            sender.send(output_addr, output_addr, payload_bytes);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            receiver.receive();
        }
    }
    retire_writes_at_exit();
}
"""
)


_TREE_REDUCE_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gather = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t cols = get_compile_time_arg_val(7);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(8);
    constexpr uint32_t stage1_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t stage2_sem_id = get_compile_time_arg_val(10);
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_col = get_arg_val<uint32_t>(2);
    const uint32_t my_row = get_arg_val<uint32_t>(3);
    const uint32_t leader_x = get_arg_val<uint32_t>(4);
    const uint32_t leader_y = get_arg_val<uint32_t>(5);
    const uint32_t root_x = get_arg_val<uint32_t>(6);
    const uint32_t root_y = get_arg_val<uint32_t>(7);
    constexpr uint32_t payload_bytes = num_tiles * page_bytes;

    Semaphore<> stage1(stage1_sem_id);
    Semaphore<> stage2(stage2_sem_id);
    Noc noc;

    const bool is_leader = (my_col == 0);
    const bool is_root = is_leader && (my_row == 0);

    if (is_root) {
        // Reduces its own row (stage 1), assembles the row partials (stage 2), then multicasts.
        CircularBuffer gather(cb_gather);
        CircularBuffer partial(cb_partial);
        CircularBuffer stage2cb(cb_stage2);
        CircularBuffer output(cb_output);
        auto sender = mc.sender(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            gather.reserve_back(cols * num_tiles);
            const uint32_t gather_addr = gather.get_write_ptr();
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr), gather_addr, payload_bytes);
            noc_async_read_barrier();
            if (cols > 1) {
                stage1.wait_min((iter + 1) * (cols - 1));
            }
            gather.push_back(cols * num_tiles);

            stage2cb.reserve_back(rows * num_tiles);
            const uint32_t stage2_addr = stage2cb.get_write_ptr();
            partial.wait_front(num_tiles);
            const uint32_t partial_addr = partial.get_read_ptr();
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], partial_addr), stage2_addr, payload_bytes);
            noc_async_read_barrier();
            partial.pop_front(num_tiles);
            if (rows > 1) {
                stage2.wait_min((iter + 1) * (rows - 1));
            }
            stage2cb.push_back(rows * num_tiles);

            output.wait_front(num_tiles);
            sender.send(output_addr, output_addr, payload_bytes);
            output.pop_front(num_tiles);
        }
    } else if (is_leader) {
        // Reduces its own row (stage 1) and forwards the row partial to the root (stage 2 input).
        CircularBuffer gather(cb_gather);
        CircularBuffer partial(cb_partial);
        CircularBuffer stage2cb(cb_stage2);
        auto receiver = mc.receiver(noc);
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            gather.reserve_back(cols * num_tiles);
            const uint32_t gather_addr = gather.get_write_ptr();
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], input_addr), gather_addr, payload_bytes);
            noc_async_read_barrier();
            if (cols > 1) {
                stage1.wait_min((iter + 1) * (cols - 1));
            }
            gather.push_back(cols * num_tiles);

            partial.wait_front(num_tiles);
            const uint32_t partial_addr = partial.get_read_ptr();
            const uint32_t stage2_addr = stage2cb.get_write_ptr();
            noc_async_write(
                partial_addr,
                get_noc_addr(root_x, root_y, stage2_addr + my_row * payload_bytes),
                payload_bytes);
            noc_async_write_barrier();
            stage2.up(noc, root_x, root_y, 1);
            partial.pop_front(num_tiles);
            receiver.receive();
        }
    } else {
        // Row member: contributes its block to the row leader (stage 1), then takes the broadcast.
        CircularBuffer gather(cb_gather);
        auto receiver = mc.receiver(noc);
        const uint32_t gather_addr = gather.get_write_ptr();
        for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
            noc_async_write(
                input_addr,
                get_noc_addr(leader_x, leader_y, gather_addr + my_col * payload_bytes),
                payload_bytes);
            noc_async_write_barrier();
            stage1.up(noc, leader_x, leader_y, 1);
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


def _inline_kernel(source, core_ranges, compile_time_args, runtime_args, config, defines=()):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=compile_time_args,
        defines=list(defines),
        runtime_args=runtime_args,
        config=config,
    )


def _virtual_coords(device, cores):
    result = []
    for x, y in cores:
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        result.extend([core.x, core.y])
    return result


def _mcast_dataflow_kernels(
    descriptor, device, layout, source, compile_time_args, runtime_by_core, *, rotating, first_root=False, defines=()
):
    """One dataflow kernel and one multicast channel per group rectangle, attached as `mcast`.

    Mcast partitions its receivers into consecutive row-major runs, which match the group
    rectangles only when they are stacked vertically, so each group gets its own channel. The
    groups are disjoint, so every channel reuses the same semaphore ids.
    """
    kernels = []
    for group in layout.groups:
        runtime_args = ttnn.RuntimeArgs()
        for x, y in group.cores:
            runtime_args[x][y] = runtime_by_core[(x, y)]
        kernel = _inline_kernel(
            source, group.core_range_set, compile_time_args, runtime_args, ttnn.ReaderConfigDescriptor(), defines
        )
        if rotating:
            sender_config = ttnn.McastRotatingSenderConfig()
        else:
            sender_config = ttnn.McastFixedSenderConfig(sender_index=0 if first_root else layout.group_size - 1)
        mcast = ttnn.Mcast(device, ttnn.McastConfig(), group.core_range_set, layout.group_size, sender_config)
        mcast.attach(descriptor, "mcast", [kernel], SEM_MCAST_BASE)
        kernels.append(kernel)
    return kernels


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
    dataflow_rt = {}
    compute_rt = {}
    for group in layout.groups:
        for index, (x, y) in enumerate(group.cores):
            dataflow_rt[(x, y)] = [input_tensor.buffer_address(), index]
            compute_rt[(x, y)] = [group_size, num_tiles, kernel_iters]

    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    compute = _compute_kernel(layout.core_ranges, compute_rt)
    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    dataflow = _mcast_dataflow_kernels(
        descriptor,
        input_tensor.device(),
        layout,
        _MCAST_ALL_GATHER_KERNEL,
        [CB_GATHER, CB_OUTPUT] + [num_tiles, page_bytes, group_size, kernel_iters],
        dataflow_rt,
        rotating=True,
    )
    descriptor.kernels = [*dataflow, compute]
    return descriptor


def _create_reduce_root_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    group_size = layout.group_size
    dataflow_rt = {}
    compute_rt = {}
    roots = []
    for group in layout.groups:
        root_virtual = input_tensor.device().worker_core_from_logical_core(group.root)
        roots.append((group.root.x, group.root.y))
        for index, (x, y) in enumerate(group.cores):
            dataflow_rt[(x, y)] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                index,
                root_virtual.x,
                root_virtual.y,
            ]
        compute_rt[(group.root.x, group.root.y)] = [group_size, num_tiles, kernel_iters]

    root_ranges = _core_range_set(roots)
    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * num_tiles, page_bytes, input_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0)]
    compute = _compute_kernel(root_ranges, compute_rt)
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    dataflow = _mcast_dataflow_kernels(
        descriptor,
        input_tensor.device(),
        layout,
        _REDUCE_ROOT_KERNEL,
        [CB_GATHER, CB_OUTPUT] + [num_tiles, page_bytes, group_size, kernel_iters, SEM_PROGRESS],
        dataflow_rt,
        rotating=False,
    )
    descriptor.kernels = [*dataflow, compute]
    return descriptor


def _signal_defines(flush):
    """`flush=True` builds the `_flush` twin: flush (not barrier) before each publishing semaphore increment."""
    return [("SIGNAL_AFTER_FLUSH", "1")] if flush else []


def _create_reduce_scatter_descriptor(
    input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters, *, flush=False
):
    group_size = layout.group_size
    # Every core works, the root included - it takes its share and then multicasts.
    # Reserving a core made num_workers coprime with power-of-two tile counts, which
    # forced a ragged split (32 over 7 -> 5,5,5,5,4,4,4) and left 1/G of the group
    # idle through the whole gather/reduce phase.
    num_workers = min(num_tiles, group_size)
    max_assigned = (num_tiles + num_workers - 1) // num_workers
    dataflow_rt = {}
    compute_rt = {}
    worker_cores = []
    for group in layout.groups:
        virtual = _virtual_coords(input_tensor.device(), group.cores)
        root_virtual = input_tensor.device().worker_core_from_logical_core(group.root)
        for index, (x, y) in enumerate(group.cores):
            dataflow_rt[(x, y)] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                index,
                root_virtual.x,
                root_virtual.y,
            ] + virtual
            if index < num_workers:
                worker_cores.append((x, y))
                # max_assigned, not this worker's ragged share: the CBs are sized on
                # max_assigned and a CB's capacity must be an exact multiple of its
                # push/pop quantum. Compute therefore reduces (and the dataflow
                # kernel gathers at) a uniform stride; the pad tiles on a
                # lighter-loaded worker are never written back to the root.
                compute_rt[(x, y)] = [group_size, max_assigned, kernel_iters]

    worker_ranges = _core_range_set(worker_cores)
    cbs = [
        _normal_cb(CB_GATHER, worker_ranges, group_size * max_assigned, page_bytes, input_tensor.dtype),
        _normal_cb(CB_PARTIAL, worker_ranges, max_assigned, page_bytes, output_tensor.dtype),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0)]
    compute = _compute_kernel(worker_ranges, compute_rt, output_cb=CB_PARTIAL)
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    dataflow = _mcast_dataflow_kernels(
        descriptor,
        input_tensor.device(),
        layout,
        _REDUCE_SCATTER_KERNEL,
        [CB_GATHER, CB_PARTIAL] + [num_tiles, page_bytes, group_size, num_workers, kernel_iters, SEM_PROGRESS],
        dataflow_rt,
        rotating=False,
        defines=_signal_defines(flush),
    )
    descriptor.kernels = [*dataflow, compute]
    return descriptor


def _create_reduce_scatter_push_descriptor(
    input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters, *, flush=False
):
    group_size = layout.group_size
    num_workers = min(num_tiles, group_size)
    max_assigned = (num_tiles + num_workers - 1) // num_workers
    dataflow_rt = {}
    compute_rt = {}
    worker_cores = []
    for group in layout.groups:
        virtual = _virtual_coords(input_tensor.device(), group.cores)
        root_virtual = input_tensor.device().worker_core_from_logical_core(group.root)
        for index, (x, y) in enumerate(group.cores):
            dataflow_rt[(x, y)] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                index,
                root_virtual.x,
                root_virtual.y,
            ] + virtual
            if index < num_workers:
                worker_cores.append((x, y))
                compute_rt[(x, y)] = [group_size, max_assigned, kernel_iters]

    worker_ranges = _core_range_set(worker_cores)
    # Group-wide, not worker-only: a pusher derives the destination gather address
    # from its OWN get_write_ptr(), which requires an identical CB layout on every
    # participating core. Identical to the pull variant's footprint whenever
    # num_workers == group_size.
    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, group_size * max_assigned, page_bytes, input_tensor.dtype),
        _normal_cb(CB_PARTIAL, layout.core_ranges, max_assigned, page_bytes, output_tensor.dtype),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=layout.core_ranges, initial_value=0),
    ]
    compute = _compute_kernel(worker_ranges, compute_rt, output_cb=CB_PARTIAL)
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    dataflow = _mcast_dataflow_kernels(
        descriptor,
        input_tensor.device(),
        layout,
        _REDUCE_SCATTER_PUSH_KERNEL,
        [CB_GATHER, CB_PARTIAL]
        + [num_tiles, page_bytes, group_size, num_workers, kernel_iters, SEM_PROGRESS, SEM_GATHER],
        dataflow_rt,
        rotating=False,
        defines=_signal_defines(flush),
    )
    descriptor.kernels = [*dataflow, compute]
    return descriptor


def _create_tree_reduce_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters):
    rows, cols = layout.group_shape
    # A 1-D group has only one grid axis to reduce along, so the hierarchy collapses to a single
    # gather-to-root: reuse the root-reduce path directly rather than run a degenerate second stage.
    if rows == 1 or cols == 1:
        return _create_reduce_root_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters)

    device = input_tensor.device()
    dataflow_rt = {}
    compute_rt = ttnn.RuntimeArgs()
    leader_cores = []
    for group in layout.groups:
        root_virtual = device.worker_core_from_logical_core(ttnn.CoreCoord(*group.cores[0]))
        leader_virtual = []
        for row in range(rows):
            leader = device.worker_core_from_logical_core(ttnn.CoreCoord(*group.cores[row * cols]))
            leader_virtual.append((leader.x, leader.y))
        for index, (x, y) in enumerate(group.cores):
            my_row = index // cols
            my_col = index % cols
            leader_x, leader_y = leader_virtual[my_row]
            dataflow_rt[(x, y)] = [
                input_tensor.buffer_address(),
                output_tensor.buffer_address(),
                my_col,
                my_row,
                leader_x,
                leader_y,
                root_virtual.x,
                root_virtual.y,
            ]
            if my_col == 0:
                leader_cores.append((x, y))
                compute_rt[x][y] = [cols, num_tiles, kernel_iters, 1 if index == 0 else 0, rows]

    leader_ranges = _core_range_set(leader_cores)
    cbs = [
        _normal_cb(CB_GATHER, layout.core_ranges, cols * num_tiles, page_bytes, input_tensor.dtype),
        _normal_cb(CB_PARTIAL, leader_ranges, num_tiles, page_bytes, output_tensor.dtype),
        _normal_cb(CB_STAGE2, leader_ranges, rows * num_tiles, page_bytes, output_tensor.dtype),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PROGRESS, core_ranges=layout.core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_STAGE2, core_ranges=layout.core_ranges, initial_value=0),
    ]
    compute = _inline_kernel(
        _GRID_REDUCE_COMPUTE_KERNEL,
        leader_ranges,
        [CB_GATHER, CB_PARTIAL, CB_STAGE2, CB_OUTPUT],
        compute_rt,
        ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    dataflow = _mcast_dataflow_kernels(
        descriptor,
        device,
        layout,
        _TREE_REDUCE_KERNEL,
        [CB_GATHER, CB_PARTIAL, CB_STAGE2, CB_OUTPUT]
        + [num_tiles, page_bytes, rows, cols, kernel_iters, SEM_PROGRESS, SEM_STAGE2],
        dataflow_rt,
        rotating=False,
        first_root=True,
    )
    descriptor.kernels = [*dataflow, compute]
    return descriptor


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
    if variant in ("reduce_scatter_mcast", "reduce_scatter_mcast_flush"):
        return _create_reduce_scatter_descriptor(
            input_tensor,
            output_tensor,
            layout,
            num_tiles,
            page_bytes,
            kernel_iters,
            flush=variant.endswith("_flush"),
        )
    if variant in ("reduce_scatter_push", "reduce_scatter_push_flush"):
        return _create_reduce_scatter_push_descriptor(
            input_tensor,
            output_tensor,
            layout,
            num_tiles,
            page_bytes,
            kernel_iters,
            flush=variant.endswith("_flush"),
        )
    return _create_tree_reduce_descriptor(input_tensor, output_tensor, layout, num_tiles, page_bytes, kernel_iters)


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


# =============================================================================================
# Per-row-vector payload: a row reduction split across K Tensix cores.
#
# The all-reduce above sums WHOLE TILES elementwise. A very common special case only needs one value
# per row: a row is WIDTH_SHARDED across K Tensix cores ("members"), b tile-rows and w tiles per
# member per tile-row, bf16 in L1. Each member row-reduces its slice (REDUCE_ROW, float32 DST), which
# leaves its per-row partial in COLUMN 0 of one float32 tile per tile-row; the K partials of a
# tile-row are added by that tile-row's OWNER (one member per tile-row), and every member needs the
# per-row totals back. The output is that column-0 tile at every member (float32, one tile per
# tile-row, WIDTH_SHARDED like the input): column 0 = sum over the WHOLE row, the other 31 columns
# don't-care. The result is never expanded across the tile; a consumer applies it with a column
# broadcast (`mul_tiles_bcast_cols` and friends).
#
# Two kernel-level choices vary; the row reduce and the owner placement are identical in every variant.
#
# Concept 1, bytes per member on the wire, both directions.
#   Gather (member -> owner), the variant prefix:
#   - `tile_partial_*`:  the whole float32 partial tile, 4096 B per tile-row; the owner adds K tiles.
#   - `faces_partial_*`: only faces 0 and 2, the two 16x16 faces that hold column 0, 2 x 1024 B =
#                        2048 B per tile-row; the owner still adds K tiles (faces 1/3 are don't-care).
#   - `row_partial_*`:   the member transposes the partial so the 32 values are row 0 and ships that row,
#                        2 x 64 B = 128 B per tile-row, into row m % 32 of gather tile m // 32 at the
#                        owner; the owner column-reduces ceil(K / 32) gather tiles instead of adding K,
#                        and transposes the one result back to column 0.
#   Return (owner -> members): the whole column-0 tile (4096 B), or with `_faces_return` only its
#   faces 0/2 (2 x 1024 B).
# Concept 2, the return path, the variant suffix:
#   - `*_unicast`: every owner writes its result tile to all K members, then bumps each one's semaphore.
#   - `*_mcast`:   owners write their result tile to one root (member 0); the root multicasts all b
#                  result tiles to the members' bounding rectangle, then multicasts the semaphore.
#
# Kernels: compute (row reduce, [transpose], owner combine), a gather kernel on RISCV_1 + NoC0
# (scaler tile, partials to the owners, wait for the result), a return kernel on RISCV_0 + NoC1
# (owner hand-off and the return).
# =============================================================================================

RV_PARTIALS = ("tile", "faces", "row")
RV_RETURNS = ("tile", "faces")
RV_PATHS = ("unicast", "mcast")


def row_vector_variant_name(partial, returned, path):
    return f"{partial}_partial{'_faces_return' if returned == 'faces' else ''}_{path}"


# Every combination builds; ROW_VECTOR_VARIANTS is the measured set, baseline first.
ALL_ROW_VECTOR_VARIANTS = tuple(
    row_vector_variant_name(p, r, path) for p in RV_PARTIALS for r in RV_RETURNS for path in RV_PATHS
)
ROW_VECTOR_VARIANTS = (
    "tile_partial_unicast",
    "faces_partial_unicast",
    "row_partial_unicast",
    "tile_partial_mcast",
    "faces_partial_mcast",
    "row_partial_mcast",
    "faces_partial_faces_return_unicast",
    "faces_partial_faces_return_mcast",
)

FP32_TILE_BYTES = 4096

RV_CB_IN = 0
RV_CB_SCALER = 1
RV_CB_PARTIAL = 2
RV_CB_TMP = 3
RV_CB_GATHER = 4
RV_CB_OWNED = 5

RV_SEM_GATHER = 0  # on an owner: +1 per member partial landed
RV_SEM_RESULT = 1  # on a member: result landed (+1 per owner for unicast, iteration stamp for mcast)
RV_SEM_COLLECT = 2  # on the root (mcast): +1 per owner result landed
RV_SEM_LOCAL = 3  # on the root (mcast): the word the semaphore multicast copies from

RV_NOT_OWNER = 0xFFFFFFFF


def parse_row_vector_variant(variant):
    """(partial, returned, path), e.g. faces_partial_faces_return_mcast -> (faces, faces, mcast)."""
    for p in RV_PARTIALS:
        for r in RV_RETURNS:
            for path in RV_PATHS:
                if row_vector_variant_name(p, r, path) == variant:
                    return p, r, path
    raise ValueError(f"variant must be one of {ALL_ROW_VECTOR_VARIANTS}, got {variant!r}")


@dataclass(frozen=True)
class MemberLayout:
    """K members packed row-major into a rows x cols rectangle at logical (0, 0)."""

    num_members: int
    rows: int
    cols: int
    members: tuple

    @property
    def member_ranges(self):
        ranges = []
        full_rows = self.num_members // self.cols
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.cols - 1, full_rows - 1)))
        tail = self.num_members % self.cols
        if tail:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(tail - 1, full_rows)))
        return ttnn.CoreRangeSet(ranges)

    @property
    def bbox_ranges(self):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.cols - 1, self.rows - 1))])

    @property
    def placement(self):
        tail = self.num_members % self.cols
        shape = f"{self.rows}x{self.cols}"
        return shape if tail == 0 else f"{shape} (last row {tail})"


def build_member_layout(device, num_members):
    """The widest exact rectangle that fits the worker grid; else whole grid rows plus a partial row."""
    grid = device.compute_with_storage_grid_size()
    if num_members < 2:
        raise ValueError("a row all-reduce needs at least two members")
    if num_members > grid.x * grid.y:
        raise ValueError(f"{num_members} members do not fit the {grid.x}x{grid.y} worker grid")
    rows = cols = None
    for c in range(min(grid.x, num_members), 0, -1):
        if num_members % c == 0 and num_members // c <= grid.y:
            rows, cols = num_members // c, c
            break
    if cols is None:
        # No exact rectangle: whole grid rows, then a partial row. The multicast return then also
        # reaches the bounding rectangle's few non-member Tensix cores (they hold only the result
        # semaphore; the output's L1 range is reserved on every Tensix core).
        cols = grid.x
        rows = -(-num_members // cols)
    members = tuple((i % cols, i // cols) for i in range(num_members))
    return MemberLayout(num_members, rows, cols, members)


def row_owner_indices(layout, tile_rows):
    """Member index that owns each tile-row. Owners sit on a diagonal of the members' rectangle
    (tile-row r -> grid column r * cols / b, grid row r * rows / b), so no two owners share a grid
    column when b <= cols: owners stacked in one column share that column's NoC links for every
    incoming partial. Falls back to an even spread over the member order."""
    k, rows, cols = layout.num_members, layout.rows, layout.cols
    diagonal = [((r * rows) // tile_rows) * cols + (r * cols) // tile_rows for r in range(tile_rows)]
    if all(i < k for i in diagonal) and len(set(diagonal)) == tile_rows:
        return diagonal
    return [(r * k) // tile_rows for r in range(tile_rows)]


def create_row_vector_memory_config(device, num_members, tile_rows, width_tiles):
    """WIDTH_SHARDED over the K members: (b x 32) x (w x 32) per member (w = 1 for the output)."""
    layout = build_member_layout(device, num_members)
    return ttnn.create_sharded_memory_config(
        shape=(tile_rows * ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE),
        core_grid=layout.member_ranges,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def num_gather_tiles(variant, num_members):
    """Tiles an owner reduces per tile-row: K partial tiles, or ceil(K / 32) gather tiles of rows."""
    return -(-num_members // ttnn.TILE_SIZE) if parse_row_vector_variant(variant)[0] == "row" else num_members


# Compute: per-member row reduce -> [transpose] -> owner combine. Differs only by ROW_PARTIAL
# (FACES_PARTIAL and FACES_RETURN change only what the dataflow kernels move).
# RT args: [tile_rows, width_tiles, is_owner, num_gather, kernel_iters]
_RV_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_partial = 2, cb_tmp = 3, cb_gather = 4, cb_owned = 5;

// Transpose n tiles of icb into ocb (row 0 <-> column 0), one tile per wait/pop: cb_tmp carries both
// the b member partials and the owner's single result, so its quantum must stay 1 tile.
ALWI void transpose_tiles(uint32_t icb, uint32_t ocb, uint32_t n) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    transpose_init(icb);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(icb, 1);
        cb_reserve_back(ocb, 1);
        tile_regs_acquire();
        transpose_tile(icb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, ocb);
        tile_regs_release();
        cb_push_back(ocb, 1);
        cb_pop_front(icb, 1);
    }
}

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t width_tiles = get_arg_val<uint32_t>(1);
    const uint32_t is_owner = get_arg_val<uint32_t>(2);
    const uint32_t num_gather = get_arg_val<uint32_t>(3);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(4);

#ifdef ROW_PARTIAL
    constexpr uint32_t cb_reduced = cb_tmp;      // transposed before it ships
#else
    constexpr uint32_t cb_reduced = cb_partial;  // ships as is
#endif

    compute_kernel_hw_startup(cb_scaler, cb_in, cb_reduced);
    cb_wait_front(cb_in, tile_rows * width_tiles);  // resident shard, read by index, never popped
    cb_wait_front(cb_scaler, 1);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // (1) Member row reduce: one float32 partial tile per tile-row, the sums in column 0.
        reconfig_data_format(cb_scaler, cb_in);
        pack_reconfig_data_format(cb_reduced);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_reduced);
        for (uint32_t r = 0; r < tile_rows; ++r) {
            cb_reserve_back(cb_reduced, 1);
            tile_regs_acquire();
            for (uint32_t t = 0; t < width_tiles; ++t) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, r * width_tiles + t, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_reduced);
            tile_regs_release();
            cb_push_back(cb_reduced, 1);
        }
        reduce_uninit();
#ifdef ROW_PARTIAL
        // Column 0 -> row 0: the 32 values become two contiguous 64 B face rows.
        transpose_tiles(cb_tmp, cb_partial, tile_rows);
#endif

        // (2) Owner combine of its tile-row.
        if (is_owner) {
            cb_wait_front(cb_gather, num_gather);
#ifdef ROW_PARTIAL
            // Gather tile g holds member partials as rows; a column reduce adds them into row 0.
            reconfig_data_format(cb_gather, cb_scaler);
            pack_reconfig_data_format(cb_tmp);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, cb_tmp);
            cb_reserve_back(cb_tmp, 1);
            tile_regs_acquire();
            for (uint32_t g = 0; g < num_gather; ++g) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, g, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);
            reduce_uninit();
            cb_pop_front(cb_gather, num_gather);
            transpose_tiles(cb_tmp, cb_owned, 1);  // back to column 0
#else
            // K partial tiles added in float32 DST, paired (seeded with a copy when K is odd). With
            // FACES_PARTIAL faces 1/3 hold stale bytes; faces add independently and nothing reads
            // column > 0 of the result, so they are don't-care.
            reconfig_data_format(cb_gather, cb_gather);
            pack_reconfig_data_format(cb_owned);
            cb_reserve_back(cb_owned, 1);
            tile_regs_acquire();
            uint32_t first = 0;
            if (num_gather & 1) {
                copy_init(cb_gather);
                copy_tile(cb_gather, 0, 0);
                first = 1;
            }
            if (num_gather > 1) {
                add_init(cb_gather, cb_gather, true);
                for (uint32_t k = first; k < num_gather; k += 2) {
                    add_tiles(cb_gather, cb_gather, k, k + 1, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_owned);
            tile_regs_release();
            cb_push_back(cb_owned, 1);
            cb_pop_front(cb_gather, num_gather);
#endif
        }
    }
}
"""


# Gather (RISCV_1, NoC0): scaler tile, each tile-row's partial to its owner, then wait for this
# iteration's result (so the next iteration never overwrites a gather slot an owner still reads).
# RT args: [tile_rows, width_tiles, member_index, kernel_iters, (owner_x, owner_y) * tile_rows]
_RV_GATHER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_partial = 2, cb_gather = 4;
constexpr uint32_t tile_bytes = 4096;     // float32 32x32 tile
constexpr uint32_t face_row_bytes = 64;   // 16 float32 datums
constexpr uint32_t face_bytes = 1024;     // 16x16 float32 face
constexpr uint32_t sem_gather_id = 0, sem_result_id = 1;

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t width_tiles = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(3);
    constexpr uint32_t owners_base = 4;

    // The gather CB is one round deep everywhere, so its base is the same address on every owner.
    const uint32_t gather_base = get_write_ptr(cb_gather);
    const uint32_t gather_sem = get_semaphore(sem_gather_id);
    volatile tt_l1_ptr uint32_t* result_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_result_id));

    cb_reserve_back(cb_in, tile_rows * width_tiles);
    cb_push_back(cb_in, tile_rows * width_tiles);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

#ifdef ROW_PARTIAL
    // Member m owns row m % 32 of gather tile m / 32. Row j < 16 lives in faces 0/1, j >= 16 in faces 2/3.
    const uint32_t j = my_index % 32;
    const uint32_t slot =
        (my_index / 32) * tile_bytes + (j < 16 ? j * face_row_bytes : 2 * face_bytes + (j - 16) * face_row_bytes);
#else
    const uint32_t slot = my_index * tile_bytes;  // whole tile or faces 0/2 of it: slot m, tile layout kept
#endif

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_wait_front(cb_partial, tile_rows);
        const uint32_t src = get_read_ptr(cb_partial);
        for (uint32_t r = 0; r < tile_rows; ++r) {
            const uint32_t ox = get_arg_val<uint32_t>(owners_base + 2 * r);
            const uint32_t oy = get_arg_val<uint32_t>(owners_base + 2 * r + 1);
            const uint32_t from = src + r * tile_bytes;
#ifdef ROW_PARTIAL
            // Row 0 of the transposed partial = face 0 row 0 + face 1 row 0: 2 x 64 B.
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), face_row_bytes);
            noc_async_write(from + face_bytes, get_noc_addr(ox, oy, gather_base + slot + face_bytes), face_row_bytes);
#elif defined(FACES_PARTIAL)
            // Column 0 lives in faces 0 and 2 (byte offsets 0 and 2048): 2 x 1024 B instead of 4096 B.
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), face_bytes);
            noc_async_write(
                from + 2 * face_bytes, get_noc_addr(ox, oy, gather_base + slot + 2 * face_bytes), face_bytes);
#else
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), tile_bytes);
#endif
        }
        noc_async_write_barrier();
        for (uint32_t r = 0; r < tile_rows; ++r) {
            const uint32_t ox = get_arg_val<uint32_t>(owners_base + 2 * r);
            const uint32_t oy = get_arg_val<uint32_t>(owners_base + 2 * r + 1);
            noc_semaphore_inc(get_noc_addr(ox, oy, gather_sem), 1);
        }
        cb_pop_front(cb_partial, tile_rows);

        // This member holds the all-reduced row vector once the result semaphore says so.
#ifdef MCAST_RETURN
        noc_semaphore_wait_min(result_sem, iter + 1);              // iteration stamp from the root
#else
        noc_semaphore_wait_min(result_sem, (iter + 1) * tile_rows);  // +1 per owner
#endif
    }
    noc_async_atomic_barrier();
}
"""


# Return (RISCV_0, NoC1): owner hand-off to compute, then the result to every member's output shard.
# RT args: [tile_rows, num_members, owned_row (or NOT_OWNER), num_gather, kernel_iters, output_addr,
#           MCAST:   is_root, root_x, root_y, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests
#           UNICAST: (member_x, member_y) * num_members]
_RV_RETURN_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_gather = 4, cb_owned = 5;
constexpr uint32_t tile_bytes = 4096;
constexpr uint32_t face_row_bytes = 64;
constexpr uint32_t face_bytes = 1024;
constexpr uint32_t sem_gather_id = 0, sem_result_id = 1, sem_collect_id = 2, sem_local_id = 3;
constexpr uint32_t NOT_OWNER = 0xFFFFFFFF;

#ifdef FACES_RETURN
// Only column 0 goes back: faces 0 and 2 of the result tile, 2 x 1024 B at byte offsets 0 and 2048.
constexpr uint32_t piece_bytes = face_bytes, num_pieces = 2, piece_stride = 2 * face_bytes;
#else
constexpr uint32_t piece_bytes = tile_bytes, num_pieces = 1, piece_stride = 0;
#endif

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_members = get_arg_val<uint32_t>(1);
    const uint32_t owned_row = get_arg_val<uint32_t>(2);
    const uint32_t num_gather = get_arg_val<uint32_t>(3);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(4);
    const uint32_t output_base = get_arg_val<uint32_t>(5);
    const bool is_owner = owned_row != NOT_OWNER;

    const uint32_t gather_base = get_write_ptr(cb_gather);
    volatile tt_l1_ptr uint32_t* gather_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather_id));
    const uint32_t result_sem_addr = get_semaphore(sem_result_id);

#ifdef MCAST_RETURN
    const bool is_root = get_arg_val<uint32_t>(6) != 0;
    const uint32_t root_x = get_arg_val<uint32_t>(7);
    const uint32_t root_y = get_arg_val<uint32_t>(8);
    const uint32_t mc_sx = get_arg_val<uint32_t>(9);
    const uint32_t mc_sy = get_arg_val<uint32_t>(10);
    const uint32_t mc_ex = get_arg_val<uint32_t>(11);
    const uint32_t mc_ey = get_arg_val<uint32_t>(12);
    const uint32_t num_dests = get_arg_val<uint32_t>(13);
    const uint32_t collect_sem_addr = get_semaphore(sem_collect_id);
    volatile tt_l1_ptr uint32_t* collect_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(collect_sem_addr);
    const uint32_t local_addr = get_semaphore(sem_local_id);
    volatile tt_l1_ptr uint32_t* local_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_addr);
    volatile tt_l1_ptr uint32_t* result_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_sem_addr);
#else
    constexpr uint32_t members_base = 6;
#endif

#ifdef ROW_PARTIAL
    // Rows no member writes (K % 32 .. 31 of the last gather tile) must read as zero. Members never
    // touch them, so clearing them cannot race with an incoming partial. In face order they are up to
    // three contiguous byte ranges; clear them with NoC reads of the hardware zeros region (RISC-V
    // stores would cost ~3 ns per word, more than the whole combine at small K).
    if (is_owner && (num_members % 32) != 0) {
        const uint32_t last = gather_base + (num_gather - 1) * tile_bytes;
        const uint32_t j0 = num_members % 32;
        uint32_t starts[3], ends[3], n = 0;
        if (j0 < 16) {
            starts[n] = last + j0 * face_row_bytes; ends[n++] = last + face_bytes;                    // face 0
            starts[n] = last + face_bytes + j0 * face_row_bytes; ends[n++] = last + 2 * face_bytes;   // face 1
            starts[n] = last + 2 * face_bytes; ends[n++] = last + tile_bytes;                         // faces 2, 3
        } else {
            starts[n] = last + 2 * face_bytes + (j0 - 16) * face_row_bytes; ends[n++] = last + 3 * face_bytes;
            starts[n] = last + 3 * face_bytes + (j0 - 16) * face_row_bytes; ends[n++] = last + tile_bytes;
        }
        const uint64_t zeros = get_noc_addr(my_x[noc_index], my_y[noc_index], MEM_ZEROS_BASE);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t a = starts[i]; a < ends[i]; a += MEM_ZEROS_SIZE) {
                const uint32_t chunk = ends[i] - a < MEM_ZEROS_SIZE ? ends[i] - a : MEM_ZEROS_SIZE;
                noc_async_read(zeros, a, chunk);
            }
        }
        noc_async_read_barrier();
    }
#endif

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if (is_owner) {
            cb_reserve_back(cb_gather, num_gather);
            noc_semaphore_wait_min(gather_sem, (iter + 1) * num_members);
            cb_push_back(cb_gather, num_gather);

            cb_wait_front(cb_owned, 1);
            const uint32_t src = get_read_ptr(cb_owned);
            const uint32_t dst = output_base + owned_row * tile_bytes;
#ifdef MCAST_RETURN
            for (uint32_t p = 0; p < num_pieces; ++p) {
                noc_async_write(src + p * piece_stride, get_noc_addr(root_x, root_y, dst + p * piece_stride), piece_bytes);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(root_x, root_y, collect_sem_addr), 1);
#else
            for (uint32_t m = 0; m < num_members; ++m) {
                const uint32_t mx = get_arg_val<uint32_t>(members_base + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(members_base + 2 * m + 1);
                for (uint32_t p = 0; p < num_pieces; ++p) {
                    noc_async_write(src + p * piece_stride, get_noc_addr(mx, my, dst + p * piece_stride), piece_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t m = 0; m < num_members; ++m) {
                const uint32_t mx = get_arg_val<uint32_t>(members_base + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(members_base + 2 * m + 1);
                noc_semaphore_inc(get_noc_addr(mx, my, result_sem_addr), 1);
            }
#endif
            cb_pop_front(cb_owned, 1);
        }

#ifdef MCAST_RETURN
        if (is_root) {
            noc_semaphore_wait_min(collect_sem, (iter + 1) * tile_rows);
            // Data, then the semaphore: same NoC and multicast VC, so they land in this order.
#ifdef FACES_RETURN
            for (uint32_t r = 0; r < tile_rows; ++r) {
                for (uint32_t p = 0; p < num_pieces; ++p) {
                    const uint32_t a = output_base + r * tile_bytes + p * piece_stride;
                    noc_async_write_multicast(a, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, a), piece_bytes, num_dests);
                }
            }
#else
            noc_async_write_multicast(
                output_base,
                get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, output_base),
                tile_rows * tile_bytes,
                num_dests);
#endif
            *local_word = iter + 1;
            noc_semaphore_set_multicast(
                local_addr, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, result_sem_addr), num_dests);
            // The root is not a multicast destination: its own output is complete (the owners wrote it),
            // and it releases its own gather kernel once the multicast has read its source.
            noc_async_write_barrier();
            *result_sem = iter + 1;
        }
#endif
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
"""


def _rv_fp32_cb(cb_index, core_ranges, num_tiles):
    return _normal_cb(cb_index, core_ranges, num_tiles, FP32_TILE_BYTES, ttnn.float32)


def _rv_kernel(source, core_ranges, runtime_args, config, defines):
    return _inline_kernel(source, core_ranges, [], runtime_args, config, defines)


def create_row_vector_program_descriptor(
    input_tensor, output_tensor, *, variant, num_members, tile_rows, width_tiles=1, kernel_iters=1
):
    partial, returned, path = parse_row_vector_variant(variant)
    if tile_rows < 1 or width_tiles < 1 or kernel_iters < 1:
        raise ValueError("tile_rows, width_tiles and kernel_iters must be positive")
    if tile_rows > num_members:
        raise ValueError(f"one owner per tile-row needs tile_rows <= members, got {tile_rows} > {num_members}")
    device = input_tensor.device()
    layout = build_member_layout(device, num_members)
    expected_in = [tile_rows * ttnn.TILE_SIZE, num_members * width_tiles * ttnn.TILE_SIZE]
    expected_out = [tile_rows * ttnn.TILE_SIZE, num_members * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_in or list(output_tensor.shape) != expected_out:
        raise ValueError(f"input shape must be {expected_in} and output shape {expected_out}")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.float32:
        raise ValueError("input must be bfloat16 and output float32")

    mcast = path == "mcast"
    defines = [("ROW_PARTIAL", "1")] if partial == "row" else []
    defines += [("FACES_PARTIAL", "1")] if partial == "faces" else []
    defines += [("FACES_RETURN", "1")] if returned == "faces" else []
    defines += [("MCAST_RETURN", "1")] if mcast else []
    n_gather = num_gather_tiles(variant, num_members)

    def virt(xy):
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return core.x, core.y

    member_virtual = [virt(xy) for xy in layout.members]
    owner_of_row = row_owner_indices(layout, tile_rows)
    owned = {member: r for r, member in enumerate(owner_of_row)}
    root_x, root_y = member_virtual[owner_of_row[0]]  # member 0, the owner of tile-row 0
    # Multicast rectangle in NoC1 order: the start corner is the far (max x, max y) one.
    lo = virt((0, 0))
    hi = virt((layout.cols - 1, layout.rows - 1))
    mcast_rect = [hi[0], hi[1], lo[0], lo[1]]
    num_dests = layout.rows * layout.cols - 1
    output_addr = output_tensor.buffer_address()

    compute_rt = ttnn.RuntimeArgs()
    gather_rt = ttnn.RuntimeArgs()
    return_rt = ttnn.RuntimeArgs()
    flat_members = [v for xy in member_virtual for v in xy]
    flat_owners = [v for member in owner_of_row for v in member_virtual[member]]
    for index, (x, y) in enumerate(layout.members):
        owned_row = owned.get(index, RV_NOT_OWNER)
        is_owner = owned_row != RV_NOT_OWNER
        compute_rt[x][y] = [tile_rows, width_tiles, int(is_owner), n_gather, kernel_iters]
        gather_rt[x][y] = [tile_rows, width_tiles, index, kernel_iters] + flat_owners
        common = [tile_rows, num_members, owned_row, n_gather, kernel_iters, output_addr]
        if mcast:
            return_rt[x][y] = common + [int(index == 0), root_x, root_y] + mcast_rect + [num_dests]
        else:
            return_rt[x][y] = common + flat_members

    members = layout.member_ranges
    # The multicast also lands on the bounding rectangle's non-member Tensix cores, so the result
    # semaphore exists there too.
    result_ranges = layout.bbox_ranges if mcast else members
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(RV_CB_IN, input_tensor),
        _normal_cb(RV_CB_SCALER, members, 1, 2048, ttnn.bfloat16),
        _rv_fp32_cb(RV_CB_PARTIAL, members, tile_rows),
        _rv_fp32_cb(RV_CB_GATHER, members, n_gather),
        _rv_fp32_cb(RV_CB_OWNED, members, 1),
    ]
    if partial == "row":
        cbs.append(_rv_fp32_cb(RV_CB_TMP, members, tile_rows))
    semaphores = [
        ttnn.SemaphoreDescriptor(id=RV_SEM_GATHER, core_ranges=members, initial_value=0),
        ttnn.SemaphoreDescriptor(id=RV_SEM_RESULT, core_ranges=result_ranges, initial_value=0),
    ]
    if mcast:
        semaphores += [
            ttnn.SemaphoreDescriptor(id=RV_SEM_COLLECT, core_ranges=members, initial_value=0),
            ttnn.SemaphoreDescriptor(id=RV_SEM_LOCAL, core_ranges=members, initial_value=0),
        ]

    kernels = [
        _rv_kernel(
            _RV_COMPUTE_KERNEL,
            members,
            compute_rt,
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            defines,
        ),
        _rv_kernel(
            _RV_GATHER_KERNEL,
            members,
            gather_rt,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            defines,
        ),
        _rv_kernel(
            _RV_RETURN_KERNEL,
            members,
            return_rt,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            defines,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def row_sum_all_reduce(
    input_tensor, *, variant, num_members, tile_rows, width_tiles=1, kernel_iters=1, output_tensor=None
):
    """Every member ends with the sum over the WHOLE row as column 0 of a float32 tile per tile-row
    (output WIDTH_SHARDED over the same members, one tile column each; columns 1..31 don't-care)."""
    device = input_tensor.device()
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([tile_rows * ttnn.TILE_SIZE, num_members * ttnn.TILE_SIZE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            create_row_vector_memory_config(device, num_members, tile_rows, 1),
        )
    descriptor = create_row_vector_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        num_members=num_members,
        tile_rows=tile_rows,
        width_tiles=width_tiles,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
