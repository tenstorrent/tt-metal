// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/cb_api.h"

namespace NAMESPACE {

// Helper function to extract channel data from stick
// Each stick contains all channels for a single spatial position
inline void extract_channel_data(volatile uint16_t* stick_data, uint32_t channels, float* channel_values) {
    // For bfloat16 format, each channel is 2 bytes
    // Convert bfloat16 to float for easier processing
    union {
        float f;
        uint32_t i;
    } converter;

    for (uint32_t c = 0; c < channels; c++) {
        // Simple bfloat16 to float conversion for RISC-V
        // bfloat16 is just the upper 16 bits of float32
        converter.i = ((uint32_t)stick_data[c]) << 16;
        channel_values[c] = converter.f;
    }
}

// Helper function to find max value among an array of values
inline float find_max_value(float* values, uint32_t count) {
    if (count == 0) {
        return 0.0f;
    }

    float max_val = values[0];
    for (uint32_t i = 1; i < count; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
        }
    }
    return max_val;
}

// Helper function to construct output stick from channel max values
inline void construct_output_stick(float* channel_maxes, uint32_t channels, volatile uint16_t* output_stick) {
    // Convert float max values back to bfloat16 and write to output stick
    union {
        float f;
        uint32_t i;
    } converter;

    for (uint32_t c = 0; c < channels; c++) {
        converter.f = channel_maxes[c];
        // Extract upper 16 bits to get bfloat16
        output_stick[c] = (uint16_t)(converter.i >> 16);
    }
}

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

    // Temporary buffers for processing
    float channel_buffer[32];  // Support up to 32 channels for now
    float window_channels[32 * 8];  // Buffer for channels from up to 8 sticks (2x2x2 window)
    float channel_maxes[32];        // Max values for each channel

    // Bounds check - ensure we don't exceed buffer limits
    uint32_t safe_channels = channels > 32 ? 32 : channels;

    // Process every filter window - loop to handle multiple windows
    for (uint32_t window = 0; window < num_windows; window++) {
        // Wait for input window to be available
        cb_wait_front(cb_input_window, window_size);

        // Reserve space for output
        cb_reserve_back(cb_output, 1);

        // DEBUG: Print window info
        DPRINT << "Processing window " << window << " of " << num_windows << ENDL();

        // SIMPLIFIED STEP: Just copy first input stick to output (like working PASS 3)
        // This ensures CB management is correct before adding max computation

        // Get first stick from input window
        volatile uint16_t* input_stick;
        cb_get_tile(cb_input_window, 0, (volatile void*)&input_stick);

        // Get output tile
        volatile uint16_t* output_tile;
        cb_get_tile(cb_output, 0, (volatile void*)&output_tile);

        // Copy first stick to output
        for (uint32_t i = 0; i < 512; i++) {  // 32x16 tile = 512 elements
            output_tile[i] = input_stick[i];
        }

        // DEBUG: Print first few values
        DPRINT << "Input[0-3]: " << input_stick[0] << " " << input_stick[1] << " " << input_stick[2] << " "
               << input_stick[3] << ENDL();

        // Pop all input window elements
        for (uint32_t i = 0; i < window_size; i++) {
            cb_pop_front(cb_input_window, 1);
        }

        // Push output back
        cb_push_back(cb_output, 1);

        // DEBUG: Print completion
        DPRINT << "Window " << window << " completed successfully" << ENDL();
    }
}

}  // namespace NAMESPACE
