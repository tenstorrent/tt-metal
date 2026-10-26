# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row all-reduce between compute steps: one-shot exchange vs the same exchange pipelined in waves.

Each active Tensix core holds an L1 shard of ``rows`` tile-rows x ``width_tiles`` tiles (bf16).
Cores form groups of ``group_size`` along a grid row, and every group computes, per tile-row r:

    partial[m][r] = sum over w of x[m][r][w]          (each member m, compute)
    total[r]      = sum over m of partial[m][r]       (reduce-scatter: owner of r sums, then all-gathers)
    y[m][r][w]    = x[m][r][w] + total[r]             (each member, compute)

The exchange is a reduce-scatter: every member sends its partial of row r to the row's owner, the
owner sums the group's partials and all-gathers the total back to every member. Both variants run
the SAME kernels and the same arithmetic in the same order (so their outputs are bit-identical);
only the number of waves the block's rows are split into changes:

* ``one_shot`` - one wave: all partials, then the exchange, then all of the apply step. Compute
  and exchange run back to back.
* ``waves`` - the rows are split into ``waves`` waves. Wave k's partials travel while wave k+1's
  are summed, and wave k is applied as soon as its totals land. Each wave has its own pair of
  semaphores (a single counter cannot tell wave k's arrivals from a faster member's wave k+1), and
  the reader drives the four exchange steps from a non-blocking poll loop, so whichever step is
  ready runs.
"""

from dataclasses import dataclass

import ttnn

CB_X = 0
CB_PARTIAL = 1
CB_GATHER = 2
CB_COMBINED = 3
CB_RESULT = 4
CB_OUT = 16

# One semaphore pair per wave: gather arrivals at [0, waves), total arrivals at [waves, 2 * waves).
MAX_WAVES = 8

# Baseline first.
VARIANTS = ("one_shot", "waves")


@dataclass(frozen=True)
class WavePlan:
    """How one block of ``rows`` tile-rows is split into waves and owners."""

    waves: int
    wave_rows: int  # tile-rows per wave
    owner_rows: int  # tile-rows each owner sums per wave
    owners: int  # members that own rows (the first ``owners`` of the group)


def wave_plan(rows, group_size, waves):
    if waves < 1 or waves > MAX_WAVES:
        raise ValueError(f"waves must be in [1, {MAX_WAVES}], got {waves}")
    if rows % waves:
        raise ValueError(f"rows ({rows}) must be a multiple of waves ({waves})")
    wave_rows = rows // waves
    owner_rows = -(-wave_rows // group_size)
    if wave_rows % owner_rows:
        raise ValueError(
            f"{wave_rows} tile-rows per wave do not split evenly over {group_size} members; pick another wave count"
        )
    return WavePlan(waves, wave_rows, owner_rows, wave_rows // owner_rows)


@dataclass(frozen=True)
class GroupLayout:
    group_size: int
    num_groups: int

    @property
    def num_cores(self):
        return self.group_size * self.num_groups

    @property
    def core_ranges(self):
        # Group g is grid row g, members x = 0 .. group_size-1. HEIGHT sharding walks the range
        # row-major, so shard index = g * group_size + member.
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.group_size - 1, self.num_groups - 1))]
        )

    def cores(self, group):
        return [(member, group) for member in range(self.group_size)]


def build_group_layout(device, group_size, num_groups):
    grid = device.compute_with_storage_grid_size()
    if group_size < 2 or group_size > grid.x:
        raise ValueError(f"group_size must be in [2, {grid.x}] on this grid, got {group_size}")
    if num_groups < 1 or num_groups > grid.y:
        raise ValueError(f"num_groups must be in [1, {grid.y}] on this grid, got {num_groups}")
    return GroupLayout(group_size, num_groups)


def create_sharded_memory_config(device, group_size, num_groups, rows, width_tiles):
    layout = build_group_layout(device, group_size, num_groups)
    return ttnn.create_sharded_memory_config(
        shape=(rows * ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE),
        core_grid=layout.core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# Reader (RISCV_1): hands the resident shard to compute, then runs the exchange of every wave.
_EXCHANGE_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(2);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(3);
    constexpr uint32_t cb_result = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t waves = get_compile_time_arg_val(9);
    constexpr uint32_t Rw = get_compile_time_arg_val(10);
    constexpr uint32_t qw = get_compile_time_arg_val(11);
    constexpr uint32_t owners = get_compile_time_arg_val(12);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(13);
    constexpr uint32_t sem_gather_base = 0;
    constexpr uint32_t sem_total_base = waves;

    const uint32_t my_index = get_arg_val<uint32_t>(0);
    auto member_x = [](uint32_t m) { return get_arg_val<uint32_t>(1 + m) >> 16; };
    auto member_y = [](uint32_t m) { return get_arg_val<uint32_t>(1 + m) & 0xFFFF; };
    auto sem_value = [](uint32_t id) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };

    CircularBuffer x(cb_x);
    CircularBuffer partial(cb_partial);
    CircularBuffer gather(cb_gather);
    CircularBuffer combined(cb_combined);
    CircularBuffer result(cb_result);
    // Every core allocates these CBs at the same L1 address, so a sender addresses a peer's slot
    // with its own base. Each CB is consumed in whole blocks, so every block starts at the base.
    const uint32_t gather_base = gather.get_write_ptr();
    const uint32_t result_base = result.get_write_ptr();
    const bool is_owner = my_index < owners;
    // Totals of a wave arrive from every owner but this core (its own rows it writes itself).
    constexpr uint32_t owner_senders = owners - 1;
    const uint32_t total_senders = is_owner ? owner_senders : owners;

    uint32_t gather_expected[waves] = {};
    uint32_t total_expected[waves] = {};

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        x.reserve_back(R * W);
        x.push_back(R * W);  // the shard is already in L1: hand all of it to compute

        uint32_t ks = 0;  // next wave whose partials this core sends
        uint32_t kp = 0;  // next wave whose gather this core (an owner) hands to compute
        uint32_t ka = 0;  // next wave whose totals this core (an owner) all-gathers
        uint32_t ki = 0;  // next wave whose totals have all landed here

        // A. wave ks's partials -> each owner's slot for this member.
        auto try_send = [&]() -> bool {
            if (ks >= waves || !partial.pages_available_at_front(Rw)) {
                return false;
            }
            const uint32_t src = partial.get_read_ptr();
            const uint32_t slot = gather_base + (ks * G + my_index) * qw * page_bytes;
            for (uint32_t oo = 0; oo < owners; ++oo) {
                const uint32_t o = (oo + my_index) % owners;  // rotated, so owners are not hit in lockstep
                noc_async_write(
                    src + o * qw * page_bytes, get_noc_addr(member_x(o), member_y(o), slot), qw * page_bytes);
            }
            noc_async_write_barrier();
            const uint32_t sem_addr = get_semaphore(sem_gather_base + ks);
            for (uint32_t o = 0; o < owners; ++o) {
                if (o != my_index) {
                    noc_semaphore_inc(get_noc_addr(member_x(o), member_y(o), sem_addr), 1);
                }
            }
            partial.pop_front(Rw);
            ++ks;
            return true;
        };
        // B. (owner) every member's partials of wave kp landed -> compute sums them.
        auto try_push_gather = [&]() -> bool {
            if (kp >= ks) {
                return false;
            }
            if (!is_owner) {
                kp = ks;
                return true;
            }
            if (sem_value(sem_gather_base + kp) < gather_expected[kp] + (G - 1)) {
                return false;
            }
            gather_expected[kp] += G - 1;
            gather.reserve_back(G * qw);
            gather.push_back(G * qw);
            ++kp;
            return true;
        };
        // C. (owner) compute summed wave ka's rows -> write them into every member's result rows.
        auto try_allgather = [&]() -> bool {
            if (ka >= kp) {
                return false;
            }
            if (!is_owner) {
                ka = kp;
                return true;
            }
            if (!combined.pages_available_at_front(qw)) {
                return false;
            }
            const uint32_t src = combined.get_read_ptr();
            const uint32_t dst = result_base + (ka * Rw + my_index * qw) * page_bytes;
            for (uint32_t mm = 0; mm < G; ++mm) {
                const uint32_t m = (mm + my_index) % G;
                noc_async_write(src, get_noc_addr(member_x(m), member_y(m), dst), qw * page_bytes);
            }
            noc_async_write_barrier();
            const uint32_t sem_addr = get_semaphore(sem_total_base + ka);
            for (uint32_t m = 0; m < G; ++m) {
                if (m != my_index) {
                    noc_semaphore_inc(get_noc_addr(member_x(m), member_y(m), sem_addr), 1);
                }
            }
            combined.pop_front(qw);
            ++ka;
            return true;
        };
        // D. every owner's totals of wave ki landed -> compute applies the wave.
        auto try_push_result = [&]() -> bool {
            if (ki >= ka) {
                return false;
            }
            if (sem_value(sem_total_base + ki) < total_expected[ki] + total_senders) {
                return false;
            }
            total_expected[ki] += total_senders;
            result.reserve_back(Rw);
            result.push_back(Rw);
            ++ki;
            return true;
        };

        // A non-blocking poll loop: a blocking wait on any one step would serialize the waves again.
        // The cheap hand-offs (D, B) drain first so a long NoC step never sits on a ready push; then
        // one send (other owners wait on it), then one all-gather.
        while (ki < waves) {
            invalidate_l1_cache();
            while (try_push_result() || try_push_gather()) {
            }
            if (try_send()) {
                continue;
            }
            try_allgather();
        }
    }
}
"""


# Compute: partials of every wave, then this core's owned sums, then the apply step wave by wave.
_COMPUTE_KERNEL = r"""
#include <stdint.h>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// Sum `count` tiles of in_cb (front-relative indices first, first + stride, ...) into one out_cb tile.
// The running sum moves DEST -> SrcA at in_cb's format (bf16 here), so long sums of arbitrary values
// truncate toward zero; it is exact for the test's inputs and enough to give the exchange real work.
inline void sum_tiles(uint32_t in_cb, uint32_t first, uint32_t count, uint32_t stride, uint32_t out_cb) {
    CircularBuffer out(out_cb);
    out.reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(in_cb);
    copy_tile(in_cb, first, 0);
    if (count > 1) {
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb);
        for (uint32_t i = 1; i < count; ++i) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(in_cb, first + i * stride, 0);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
    out.push_back(1);
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(2);
    constexpr uint32_t cb_combined = get_compile_time_arg_val(3);
    constexpr uint32_t cb_result = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t waves = get_compile_time_arg_val(9);
    constexpr uint32_t Rw = get_compile_time_arg_val(10);
    constexpr uint32_t qw = get_compile_time_arg_val(11);
    constexpr uint32_t owners = get_compile_time_arg_val(12);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(13);
    constexpr uint32_t dst_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;

    const bool is_owner = get_arg_val<uint32_t>(0) < owners;

    compute_kernel_hw_startup(cb_x, cb_result, cb_out);
    CircularBuffer x(cb_x);
    CircularBuffer gather(cb_gather);
    CircularBuffer result(cb_result);
    CircularBuffer out(cb_out);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        x.wait_front(R * W);

        // 1. Row partials, in wave order: the reader ships wave k while wave k+1 is summed.
        for (uint32_t r = 0; r < R; ++r) {
            sum_tiles(cb_x, r * W, W, 1, cb_partial);
        }

        // 2. (owner) Sum the group's partials of this core's rows, wave by wave. The gather of wave k
        //    holds G slots of qw rows ([member][row]), so row j's G partials are qw tiles apart.
        if (is_owner) {
            for (uint32_t k = 0; k < waves; ++k) {
                gather.wait_front(G * qw);
                for (uint32_t j = 0; j < qw; ++j) {
                    sum_tiles(cb_gather, j, G, qw, cb_combined);
                }
                gather.pop_front(G * qw);
            }
        }

        // 3. Apply each wave as soon as its totals have landed.
        add_init(cb_x, cb_result);
        for (uint32_t k = 0; k < waves; ++k) {
            result.wait_front((k + 1) * Rw);
            for (uint32_t r = k * Rw; r < (k + 1) * Rw; ++r) {
                for (uint32_t w0 = 0; w0 < W; w0 += dst_tiles) {
                    const uint32_t n = W - w0 < dst_tiles ? W - w0 : dst_tiles;
                    out.reserve_back(n);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < n; ++i) {
                        add_tiles(cb_x, cb_result, r * W + w0 + i, r, i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < n; ++i) {
                        pack_tile(i, cb_out);
                    }
                    tile_regs_release();
                    out.push_back(n);
                }
            }
        }
        result.pop_front(R);
        x.pop_front(R * W);
    }
}
"""


# Writer (RISCV_0): the output CB is the output shard; it only releases each block.
_DRAIN_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(2);
    CircularBuffer out(cb_out);
    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        out.wait_front(block_tiles);
        out.pop_front(block_tiles);
    }
}
"""


def _normal_cb(cb_index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_index, data_format=dtype, page_size=page_bytes)],
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


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    group_size,
    num_groups,
    rows,
    width_tiles,
    waves=4,
    kernel_iters=1,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if kernel_iters < 1:
        raise ValueError(f"kernel_iters must be positive, got {kernel_iters}")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.bfloat16:
        raise ValueError("tensix_all_reduce_wave_pipeline supports bfloat16 inputs and outputs")
    device = input_tensor.device()
    layout = build_group_layout(device, group_size, num_groups)
    expected_shape = [layout.num_cores * rows * ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_shape or list(output_tensor.shape) != expected_shape:
        raise ValueError(f"input/output shape must be {expected_shape}")
    plan = wave_plan(rows, group_size, 1 if variant == "one_shot" else waves)

    page_bytes = input_tensor.buffer_aligned_page_size()
    core_ranges = layout.core_ranges
    dtype = input_tensor.dtype
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
        _normal_cb(CB_PARTIAL, core_ranges, rows, page_bytes, dtype),
        # One region of group_size * owner_rows tiles per wave: no slot is reused within a block.
        _normal_cb(CB_GATHER, core_ranges, plan.waves * group_size * plan.owner_rows, page_bytes, dtype),
        _normal_cb(CB_COMBINED, core_ranges, plan.waves * plan.owner_rows, page_bytes, dtype),
        _normal_cb(CB_RESULT, core_ranges, rows, page_bytes, dtype),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=sem_id, core_ranges=core_ranges, initial_value=0)
        for sem_id in range(2 * plan.waves)
    ]

    shape_args = [
        group_size,
        rows,
        width_tiles,
        plan.waves,
        plan.wave_rows,
        plan.owner_rows,
        plan.owners,
        kernel_iters,
    ]
    exchange_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for group in range(layout.num_groups):
        members = []
        for x, y in layout.cores(group):
            core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            members.append((core.x << 16) | core.y)
        for index, (x, y) in enumerate(layout.cores(group)):
            exchange_rt[x][y] = [index] + members
            compute_rt[x][y] = [index]

    kernels = [
        _inline_kernel(
            _EXCHANGE_KERNEL,
            core_ranges,
            [CB_X, CB_PARTIAL, CB_GATHER, CB_COMBINED, CB_RESULT, page_bytes] + shape_args,
            exchange_rt,
            ttnn.ReaderConfigDescriptor(),
        ),
        _inline_kernel(
            _COMPUTE_KERNEL,
            core_ranges,
            [CB_X, CB_PARTIAL, CB_GATHER, CB_COMBINED, CB_RESULT, CB_OUT] + shape_args,
            compute_rt,
            ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
        ),
        _inline_kernel(
            _DRAIN_KERNEL,
            core_ranges,
            [CB_OUT, rows * width_tiles, kernel_iters],
            ttnn.RuntimeArgs(),
            ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def row_all_reduce(
    input_tensor,
    *,
    variant="waves",
    group_size=8,
    num_groups=8,
    rows=32,
    width_tiles=4,
    waves=4,
    kernel_iters=1,
):
    """y = x + (the group's sum over members and the row's tiles), per tile-row, on every member."""
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
        group_size=group_size,
        num_groups=num_groups,
        rows=rows,
        width_tiles=width_tiles,
        waves=waves,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
