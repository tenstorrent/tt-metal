// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(4);
    constexpr uint32_t slice_C = get_compile_time_arg_val(5);
    constexpr uint32_t slice_Wt = get_compile_time_arg_val(6);
    constexpr bool runtime_rows = get_compile_time_arg_val(7);

    uint32_t arg_idx = 0;
    const uint32_t num_total_reduction_steps = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_workers = get_arg_val<uint32_t>(arg_idx++);

    uint32_t tiles_read_begin = start_tiles_read;
    uint32_t tiles_read_end = start_tiles_to_read;
    if constexpr (runtime_rows) {
        const uint32_t output_channel_num_pages =
            get_common_arg_val<uint32_t>(ttnn::ccl::ReduceScatterCommonArgs::input_Ht) * slice_Wt;
        tiles_read_begin = ttnn::runtime_length::balanced_start(worker_id, output_channel_num_pages, num_workers);
        tiles_read_end = ttnn::runtime_length::balanced_end(worker_id, output_channel_num_pages, num_workers);
    }

    // Hardware startup stays with the kernel; the accumulator owns only the op-level init.
    // (upstream renamed binary_op_init_common -> compute_kernel_hw_startup + per-op add_init.)
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);

    // Arm once: hoists add_init out of the chunk loop and asserts tile_granularity fits DEST.
    auto acc = compute_kernel_lib::BlockAccumulate::arm(input_cb_id, intermediate_cb, output_cb, tile_granularity);

    // The same channel/chunk walk the reader and writer drive, so the three kernels' chunk
    // boundaries — and therefore the CB protocol — cannot drift. The step count is host-computed
    // (targets after the first device, plus the final reduction), which is why this kernel never
    // reproduces the reader/writer phase logic.
    sched::LineChannelWalk walk(slice_C, tile_granularity, tiles_read_begin, tiles_read_end);

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
            walk.reset();
            while (walk.next_channel()) {
                while (walk.next_chunk()) {
                    acc.run(walk.tiles_this_chunk());
                }
            }
        }
    }
}
