// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/cb_api.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t cb_input_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t kernel_t = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(3);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(4);
    constexpr uint32_t channels = get_compile_time_arg_val(5);
    constexpr bool is_max_pool = get_compile_time_arg_val(6) == 1;

    constexpr uint32_t window_size = kernel_t * kernel_h * kernel_w;

    // Get number of filter windows from runtime args
    const uint32_t num_windows = get_arg_val<uint32_t>(0);

    // Process every filter window - loop to handle multiple windows
    for (uint32_t window = 0; window < num_windows; window++) {
        // Wait for input window to be available
        cb_wait_front(cb_input_window, window_size);

        // Reserve space for output
        cb_reserve_back(cb_output, 1);

        // Get output tile
        volatile tt_l1_ptr uint16_t* output_tile;
        cb_get_tile(cb_output, 0, (volatile tt_l1_ptr void*)&output_tile);

        // For 1x1x1 kernel (identity operation), just copy first stick
        if (window_size == 1) {
            volatile tt_l1_ptr uint16_t* input_stick;
            cb_get_tile(cb_input_window, 0, (volatile tt_l1_ptr void*)&input_stick);

            // Copy the entire page/stick data as-is from reader
            // The reader already writes the correct data layout
            for (uint32_t i = 0; i < 512; i++) {
                output_tile[i] = input_stick[i];
            }
        } else {
            // Multi-stick max pooling
            volatile tt_l1_ptr uint16_t* first_stick;
            cb_get_tile(cb_input_window, 0, (volatile tt_l1_ptr void*)&first_stick);

            // Initialize output with first stick
            for (uint32_t i = 0; i < channels; i++) {
                output_tile[i] = first_stick[i];
            }

            // Process remaining sticks and find maximum
            for (uint32_t stick_idx = 1; stick_idx < window_size; stick_idx++) {
                volatile tt_l1_ptr uint16_t* current_stick;
                cb_get_tile(cb_input_window, stick_idx, (volatile tt_l1_ptr void*)&current_stick);

                // Compare each element and keep maximum
                for (uint32_t i = 0; i < channels; i++) {
                    // Simple bfloat16 comparison by converting to float
                    union {
                        float f;
                        uint32_t u;
                    } current, max_val;
                    current.u = ((uint32_t)current_stick[i]) << 16;
                    max_val.u = ((uint32_t)output_tile[i]) << 16;

                    if (current.f > max_val.f) {
                        output_tile[i] = current_stick[i];
                    }
                }
            }
        }

        // Pop all input window elements
        for (uint32_t i = 0; i < window_size; i++) {
            cb_pop_front(cb_input_window, 1);
        }

        // Push output back
        cb_push_back(cb_output, 1);
    }
}

}  // namespace NAMESPACE
