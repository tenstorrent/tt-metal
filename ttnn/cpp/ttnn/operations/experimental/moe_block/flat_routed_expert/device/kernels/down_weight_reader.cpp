// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// down_weight_reader: streams a down core's down weights from DRAM; with row-major y, also writes y
// Runs on: down cores (one kernel per column-width group), RISCV_1 (NCRISC), NoC 0
//
// Down weights. This core's slice of every scheduled expert's down weights is a sequence of [K block x columns]
// blocks, one read per block from a contiguous run of one DRAM bank; consecutive blocks rotate over the banks. The
// blocks go into the in1 ring of down_compute.cpp, up to `batch` reads per read barrier. The ring holds about two
// experts' blocks, so the next expert's blocks load while the current one computes.
//
// y. With row-major y (flat_options::y_row_major) the y rows are written from here, on NoC 0, because the BRISC's
// NoC 1 carries the h chain (down_core_writer.cpp); the row-major y writer (y_row_major.hpp) is polled while the
// weight reads wait. When the op runs with the combine, the same writer hands the rows to the combine's cores instead
// of writing DRAM, and this kernel reports each finished expert to the combine (combine_done.hpp).
//
// Compile-time args: 0 in1 CB, 1 tiles per block, 2 bytes per tile, 3 blocks over all local experts, 4 reads per
// barrier, 5 local experts, 6 row tiles per sub-block, 7 y tile columns this core writes, 8 y tile columns in all
// Runtime args: 0 down weights address, 1 this core's first bank, 2 byte offset of this core's run in its banks,
// 3.. the schedule args (expert_schedule.hpp); with row-major y then the y address and this core's first y tile
// column; running with the combine then this core's writer index, report and hand-off args (flat_program.hpp)
#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "combine_done.hpp"
#include "expert_schedule.hpp"
#include "flat_options.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "y_row_major.hpp"

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    constexpr uint32_t in1_cb = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t local_experts = get_compile_time_arg_val(5);
    static_assert(get_compile_time_arg_val(6) == kSubBlockRowTiles);
    constexpr uint32_t y_columns = get_compile_time_arg_val(7);
    constexpr uint32_t y_hidden_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t block_bytes = block_tiles * tile_bytes;
    constexpr uint32_t blocks_per_expert = blocks / local_experts;
    constexpr uint32_t banks = flat_options::down_weight_banks;
    constexpr uint32_t ring_blocks = flat_options::down_weight_ring_blocks;
    constexpr bool reads_per_load = flat_options::down_weights_per_load;
    constexpr bool writes_y = flat_options::y_row_major;

    // block k of this core's sequence sits in bank (first bank + k) % banks, k / banks blocks into the core's run
    // there (the host packs it so: flat_weight_regions), so every core's reads rotate over the banks
    const uint32_t first_bank = moe_block_args::arg(1);
    const uint32_t run_address = moe_block_args::arg(0) + moe_block_args::arg(2);
    auto block_source = [&](uint32_t block) {
        return get_noc_addr_from_bank_id<true>(
            (first_bank + block) % banks, run_address + (block / banks) * block_bytes);
    };

    constexpr uint32_t schedule_args = 3;
    ExpertSchedule schedule;
    load_schedule<local_experts>(schedule, schedule_args, schedule_scratch(ScratchOwner::Ncrisc));
    // The blocks this core reads, in order. With down_weights_per_load (a pinned schedule) a load's blocks are read
    // once, into the load's ring region, and the compute frees them after the load's last use; otherwise each entry's
    // blocks are read in turn (a pinned expert's chunks each read its blocks again), into the next ring slots.
    const uint32_t total = (reads_per_load ? schedule.loads : schedule.entries) * blocks_per_expert;
    const uint32_t ring_base = get_write_ptr(in1_cb);  // (nothing pushed yet)
    auto destination = [&](uint32_t block, uint32_t in_batch) {
        if constexpr (reads_per_load) {
            const uint32_t region = schedule.load_region[block / blocks_per_expert];
            return ring_base + (region * blocks_per_expert + block % blocks_per_expert) * block_bytes;
        } else {
            return get_write_ptr(in1_cb) + in_batch * block_bytes;
        }
    };
    auto expert_block = [&](uint32_t block) {
        const uint32_t expert = reads_per_load ? schedule.load_expert[block / blocks_per_expert]
                                               : schedule.expert[block / blocks_per_expert];
        return expert * blocks_per_expert + block % blocks_per_expert;
    };

    // the y writer (row-major y), and its progress reports when running with the combine
    using YWriter =
        std::conditional_t<writes_y, RowMajorYWriter<tt::CBIndex::c_16, y_columns, y_hidden_tiles>, NoYWriter>;
    constexpr uint32_t y_args = schedule_args + kScheduleArgs;
    YWriter y_writer(writes_y ? moe_block_args::arg(y_args) : 0, writes_y ? moe_block_args::arg(y_args + 1) : 0);
    CombineProgress<> combine_progress;
    combine_progress.init(schedule, local_experts, true);
    uint32_t sub_blocks_written = 0;
    auto poll_y = [&]() {
        if constexpr (writes_y) {
            y_writer.issue(schedule);
            if (y_writer.retire(schedule)) {
                ++sub_blocks_written;
                combine_progress.wrote(schedule.expert[y_writer.done_entry]);
                combine_progress.report(false);
            }
        }
    };

    for (uint32_t block = 0; block < total;) {
        uint32_t count = total - block < batch ? total - block : batch;
        if constexpr (reads_per_load) {
            // a batch must not straddle a load boundary: its reservation could need slots of the next load, which a
            // pinned schedule frees only once a later load retires (a deadlock)
            const uint32_t load_left = blocks_per_expert - block % blocks_per_expert;
            count = count < load_left ? count : load_left;
        } else {
            // a batch must not straddle the ring's end either: block b sits in ring slot b % ring_blocks (as the
            // compute reads it), while a batch's reads land linearly from the write pointer, past the ring's end
            const uint32_t ring_left = ring_blocks - block % ring_blocks;
            count = count < ring_left ? count : ring_left;
        }
        uint32_t landing[batch];
        for (uint32_t i = 0; i < count; ++i) {
            while (!cb_pages_reservable_at_back(in1_cb, (i + 1) * block_tiles)) {
                poll_y();
            }
            cb_reserve_back(in1_cb, (i + 1) * block_tiles);
            landing[i] = destination(block + i, i);
        }
        for (uint32_t i = 0; i < count; ++i) {
            noc_async_read(block_source(expert_block(block + i)), landing[i], block_bytes);
        }
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            poll_y();
        }
        noc_async_read_barrier();
        cb_push_back(in1_cb, count * block_tiles);
        block += count;
    }
    if constexpr (writes_y) {
        while (sub_blocks_written < schedule.total_sub_blocks) {
            invalidate_l1_cache();
            poll_y();
        }
        y_writer.finish();
        noc_async_write_barrier();
        combine_progress.report();    // the experts this core writes nothing of
        noc_async_write_set_trid(0);  // (the next program's plain writes carry ID 0)
    }
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
