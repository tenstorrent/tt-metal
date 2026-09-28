// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("cb_input_id");
    constexpr uint32_t intermediate_cb = get_named_compile_time_arg_val("cb_interm_id");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("cb_compute_output_id");
    constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t slice_B = get_named_compile_time_arg_val("slice_B");

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    // Initialize binary operations - use the same constants consistently.
    // Hardware startup stays with the kernel; the accumulator owns only the op-level init.
    // (upstream renamed binary_op_init_common -> compute_kernel_hw_startup + per-op add_init.)
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);

    // Arm once: hoists add_init out of the chunk loop and asserts tile_granularity fits DEST.
    auto acc = compute_kernel_lib::BlockAccumulate::arm(input_cb_id, intermediate_cb, output_cb, tile_granularity);

    // The same interleaved own/other chunk walk the reader and writer drive, from the shared
    // header — previously this kernel carried its own copy of the interleave. A zero-tile own
    // chunk still runs the full CB protocol (acc.run(0) waits/pops/pushes one granule), which is
    // what keeps it in lockstep with the reader's empty-granule pushes.
    sched::DimZeroChunkWalk walk(slice_B, tile_granularity, start_tiles_read, start_tiles_to_read, direction);

    // Don't reduce on the first slice
    for (uint32_t i = 0; i < ring_size - 1; i++) {
        walk.reset();
        while (walk.next_batch()) {
            while (walk.next_chunk()) {
                acc.run(walk.tiles_this_chunk());
            }
        }
    }
}
