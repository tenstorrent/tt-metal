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
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t slice_B = get_compile_time_arg_val(4);

    uint32_t arg_idx = 0;
    const uint32_t num_total_reduction_steps = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // Hardware startup stays with the kernel; the accumulator owns only the op-level init.
    // (upstream renamed binary_op_init_common -> compute_kernel_hw_startup + per-op add_init.)
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);

    // Arm once: hoists add_init out of the chunk loop and asserts tile_granularity fits DEST.
    auto acc = compute_kernel_lib::BlockAccumulate::arm(input_cb_id, intermediate_cb, output_cb, tile_granularity);

    // The same chunk walk the reader and writer drive (slice_B plays the channel role in the
    // dim-zero family), so the three kernels' chunk boundaries cannot drift. The step count is
    // host-computed, which is why this kernel never reproduces the reader/writer phase logic.
    sched::LineChannelWalk walk(slice_B, tile_granularity, start_tiles_read, start_tiles_to_read);

    for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
        walk.reset();
        while (walk.next_channel()) {
            while (walk.next_chunk()) {
                acc.run(walk.tiles_this_chunk());
            }
        }
    }
}
