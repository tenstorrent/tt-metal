// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
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

    compute_kernel_hw_startup(cb_interm_id, cb_input_id, cb_compute_output_id);

    // The ring schedule — the batch/step/channel/chunk walk, the per-step flags and the even/odd
    // chunk split — now comes from the shared header, so this kernel, the reader and the writer are
    // walking ONE definition instead of three hand-maintained copies of it.
    sched::RingRsSchedule schedule(
        ring_size, input_tensor_B, slice_C, tile_granularity, start_tiles_read, start_tiles_to_read, direction);

    while (schedule.next_batch()) {
        while (schedule.next_step()) {
            const bool reduce_output = schedule.flags().reduce_output;
            while (schedule.next_channel()) {
                while (schedule.next_chunk()) {
                    // Not this worker's parity this step (or the zero-tile chunk): the reader does
                    // not push for it, so there is nothing to reduce.
                    if (schedule.skip() || !schedule.reduce_interm()) {
                        continue;
                    }
                    const uint32_t tiles_to_read = schedule.tiles_this_chunk();

                    // If reduce_output, add 3 tensors. Else add 2 tensors.
                    if (reduce_output) {
                        cb_wait_front(cb_interm2_id, tile_granularity);
                    }
                    cb_wait_front(cb_input_id, tile_granularity);
                    cb_wait_front(cb_interm_id, tile_granularity);

                    tile_regs_acquire();  // acquire DST registers for MATH thread, resets DST to 0
                    if (reduce_output) {
                        copy_tile_init(cb_interm2_id);
                        for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                            copy_tile(cb_interm2_id, tile_id, tile_id);  // load DST
                        }
                        add_tiles_init(cb_interm_id, cb_input_id, true);  // DST = srcA + srcB + DST
                    } else {
                        add_tiles_init(cb_interm_id, cb_input_id, false);  // DST = srcA + srcB
                    }
                    for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                        add_tiles(cb_interm_id, cb_input_id, tile_id, tile_id, tile_id);
                    }
                    tile_regs_commit();  // release lock on DST by MATH thread, signal the PACK thread

                    if (reduce_output) {
                        cb_pop_front(cb_interm2_id, tile_granularity);
                    }
                    cb_pop_front(cb_input_id, tile_granularity);
                    cb_pop_front(cb_interm_id, tile_granularity);

                    cb_reserve_back(cb_compute_output_id, tile_granularity);
                    tile_regs_wait();  // acquire lock on DST for PACK thread
                    for (uint32_t tile_id = 0; tile_id < tiles_to_read; ++tile_id) {
                        pack_tile(tile_id, cb_compute_output_id, tile_id);  // pack results from DST registers
                                                                            // to output circular buffers
                    }
                    tile_regs_release();  // release lock on DST by PACK thread
                    cb_push_back(cb_compute_output_id, tile_granularity);
                }
            }
        }
    }
}
