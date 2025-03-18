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
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(3);
    binary_op_init_common(accumulator_cb_id, accumulator_cb_id, output_tensor_cb_id);
    add_tiles_init(accumulator_cb_id, accumulator_cb_id, true);

    // UNPACK(DPRINT << "waiting on cb" << accumulator_cb_id << " num tiles:" << num_devices << ENDL());
    cb_wait_front(accumulator_cb_id, num_devices);
    for (uint32_t i = 0; i < num_devices; i++) {
        // UNPACK(print_full_tile(accumulator_cb_id, i, true));
    }
    cb_reserve_back(output_tensor_cb_id, tiles_per_core_width_output);

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_devices; i += 2) {
        add_tiles(accumulator_cb_id, accumulator_cb_id, i, i + 1, 0);
        // dprint_tensix_dest_reg(0);
    }
    tile_regs_commit();

    // Pack output tiles
    tile_regs_wait();
    pack_tile(0, output_tensor_cb_id);
    tile_regs_release();
    // UNPACK(DPRINT << "popping front from cb " << accumulator_cb_id << " tiles: " << num_devices << ENDL());
    cb_pop_front(accumulator_cb_id, num_devices);
    cb_push_back(output_tensor_cb_id, tiles_per_core_width_output);
    // DPRINT << "Kernel finished" << ENDL();
}
}  // namespace NAMESPACE
