// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t cb_input_id = get_named_compile_time_arg_val("cb_input_id");
    constexpr uint32_t cb_interm_id = get_named_compile_time_arg_val("cb_interm_id");
    constexpr uint32_t cb_interm2_id = get_named_compile_time_arg_val("cb_interm2_id");
    constexpr uint32_t cb_compute_output_id = get_named_compile_time_arg_val("cb_compute_output_id");
    constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t input_tensor_B = get_named_compile_time_arg_val("input_tensor_B");
    constexpr uint32_t slice_C = get_named_compile_time_arg_val("slice_C");

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    // The ring schedule — the batch/step/channel/chunk walk, the per-step flags and the even/odd
    // chunk split — comes from the shared header, so this kernel, the reader and the writer walk ONE
    // definition instead of three hand-maintained copies of it.
    sched::RingRsSchedule schedule(
        ring_size, input_tensor_B, slice_C, tile_granularity, start_tiles_read, start_tiles_to_read, direction);

    // Hardware startup stays here, verbatim and unchanged — the accumulator deliberately does not own
    // it (the hw startup and the per-op add_init are not interchangeable).
    compute_kernel_hw_startup(cb_interm_id, cb_input_id, cb_compute_output_id);

    // Arm the accumulator once: hoists add_tiles_init out of the chunk loop (the pre-migration kernel
    // re-issued it every chunk) and asserts tile_granularity fits the DEST register — the invariant
    // the host is reproducing when it clamps granularity to fp32_dest_acc_en ? 4 : 8. The operand
    // order matches the pre-migration add_tiles(cb_interm_id, cb_input_id, ...).
    auto acc =
        compute_kernel_lib::BlockAccumulate::arm(cb_interm_id, cb_input_id, cb_compute_output_id, tile_granularity);

    while (schedule.next_batch()) {
        while (schedule.next_step()) {
            // Terminal ring step reduces THREE tensors (input + intermediate + output); every other
            // reducing step adds two.
            const bool reduce_output = schedule.flags().reduce_output;
            while (schedule.next_channel()) {
                while (schedule.next_chunk()) {
                    // Not this worker's parity this step (or the zero-tile chunk): the reader does
                    // not push for it, so there is nothing to reduce.
                    if (schedule.skip() || !schedule.reduce_interm()) {
                        continue;
                    }
                    if (reduce_output) {
                        acc.run_seeded(cb_interm2_id, schedule.tiles_this_chunk());
                    } else {
                        acc.run(schedule.tiles_this_chunk());
                    }
                }
            }
        }
    }
}
