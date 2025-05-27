// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(4);

    // Derived compile-time constants
    constexpr uint32_t total_pages = num_devices * num_pages_per_packet;
    constexpr uint32_t num_device_pairs = num_devices / 2;  // num_devices is always even

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(fabric_receiver_cb_id, fabric_receiver_cb_id, accumulator_cb_id);
    add_tiles_init(fabric_receiver_cb_id, fabric_receiver_cb_id, true);

    // Wait for input data once before beginning processing
    cb_wait_front(fabric_receiver_cb_id, total_pages);

    // Reserve output space once before processing
    cb_reserve_back(accumulator_cb_id, num_pages_per_packet);

    // Process tiles in pairs for efficient addition
    tile_regs_acquire();

    // Pre-calculate page indices for each device pair to avoid repetitive calculations in inner loop
    for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++) {
        // Process pairs of devices (0+1, 2+3, etc.)
        for (uint32_t device_pair = 0; device_pair < num_device_pairs; device_pair++) {
            const uint32_t first_device_id = device_pair * 2;
            const uint32_t second_device_id = first_device_id + 1;

            // Calculate indices once
            const uint32_t first_index = first_device_id * num_pages_per_packet + page_group;
            const uint32_t second_index = second_device_id * num_pages_per_packet + page_group;

            add_tiles(fabric_receiver_cb_id, fabric_receiver_cb_id, first_index, second_index, page_group);
        }
    }

    tile_regs_commit();

    // Pack output tiles
    tile_regs_wait();
    for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++) {
        pack_tile(page_group, accumulator_cb_id, page_group);
    }
    tile_regs_release();

    cb_pop_front(fabric_receiver_cb_id, total_pages);
    cb_push_back(accumulator_cb_id, num_pages_per_packet);
}
}  // namespace NAMESPACE
