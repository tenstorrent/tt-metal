// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint_tensix.h"

// void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     for (uint16_t r = 0; r < 32; ++r) {
//         DPRINT << (uint)r << " : "
//                << TileSlice(
//                       cb_id,
//                       tile_id,
//                       SliceRange{
//                           .h0 = (uint8_t)r,
//                           .h1 = (uint8_t)(r + 1),
//                           .hs = (uint8_t)1,
//                           .w0 = (uint8_t)0,
//                           .w1 = (uint8_t)32,
//                           .ws = (uint8_t)1},
//                       true,
//                       untilize)
//                << ENDL();
//     }
//     DPRINT << ENDL();
// }
namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(4);

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(fabric_receiver_cb_id, fabric_receiver_cb_id, accumulator_cb_id);
    add_tiles_init(fabric_receiver_cb_id, fabric_receiver_cb_id, true);

    // UNPACK(DPRINT << "waiting on cb" << fabric_receiver_cb_id << " num tiles:" << num_devices << ENDL());
    cb_wait_front(fabric_receiver_cb_id, num_devices * num_pages_per_packet);

    // This loop doesn't do anything now that print statements are commented out
    // for (uint32_t i = 0; i < num_devices; i++) {
    //     UNPACK(print_full_tile(fabric_receiver_cb_id, i, true));
    // }

    // Reserve output space once before processing
    cb_reserve_back(accumulator_cb_id, num_pages_per_packet);

    // Process tiles in pairs for efficient addition
    tile_regs_acquire();
    for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++) {
        for (uint32_t device_id = 0; device_id < num_devices; device_id += 2) {
            add_tiles(
                fabric_receiver_cb_id,
                fabric_receiver_cb_id,
                device_id * num_pages_per_packet + page_group,
                (device_id + 1) * num_pages_per_packet + page_group,
                page_group);
            // dprint_tensix_dest_reg(0);
        }
    }
    tile_regs_commit();

    // Pack output tiles
    tile_regs_wait();
    for (uint32_t page_group = 0; page_group < num_pages_per_packet; page_group++) {
        pack_tile(page_group, accumulator_cb_id, page_group);
    }
    tile_regs_release();

    // UNPACK(DPRINT << "popping front from cb " << fabric_receiver_cb_id << " tiles: " << num_devices << ENDL());
    cb_pop_front(fabric_receiver_cb_id, num_devices * num_pages_per_packet);
    cb_push_back(accumulator_cb_id, num_pages_per_packet);
    // DPRINT << "Kernel finished" << ENDL();
}
}  // namespace NAMESPACE
