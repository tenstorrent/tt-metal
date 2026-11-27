// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#ifdef FP32_SFPU_REDUCTION
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#endif
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(4);
    constexpr uint32_t slice_C = get_compile_time_arg_val(5);

    uint32_t arg_idx = 0;
    const uint32_t num_total_reduction_steps = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    CircularBuffer cb_input(input_cb_id);
    CircularBuffer cb_intermediate(intermediate_cb);
    CircularBuffer cb_output(output_cb);

#ifdef FP32_SFPU_REDUCTION
    compute_kernel_hw_startup(input_cb_id, output_cb);
    copy_init(input_cb_id);
    add_binary_tile_init();
#else
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);
    add_init(input_cb_id, intermediate_cb, false);
#endif

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_input.wait_front(tile_granularity);
                    cb_intermediate.wait_front(tile_granularity);

#ifdef FP32_SFPU_REDUCTION
                    // Keep the communication chunk unchanged, but use only two FP32 DEST slots
                    // per output. The FPU path's tile_granularity may occupy all four slots.
                    cb_output.reserve_back(tile_granularity);
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        tile_regs_acquire();
                        copy_init(input_cb_id);
                        copy_tile(input_cb_id, tile_id, 0);
                        reconfig_data_format_srca(input_cb_id, intermediate_cb);
                        copy_init(intermediate_cb);
                        copy_tile(intermediate_cb, tile_id, 1);
                        add_binary_tile(0, 1, 0);
                        reconfig_data_format_srca(intermediate_cb, input_cb_id);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, output_cb);
                        tile_regs_release();
                    }
                    cb_input.pop_front(tile_granularity);
                    cb_intermediate.pop_front(tile_granularity);
#else
                    tile_regs_acquire();
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                    }
                    tile_regs_commit();

                    cb_input.pop_front(tile_granularity);
                    cb_intermediate.pop_front(tile_granularity);

                    cb_output.reserve_back(tile_granularity);
                    tile_regs_wait();
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        pack_tile(tile_id, output_cb);
                    }
                    tile_regs_release();
#endif
                    cb_output.push_back(tile_granularity);

                    tiles_read += num_pages_to_read;
                }
            }
        }
    }
}
