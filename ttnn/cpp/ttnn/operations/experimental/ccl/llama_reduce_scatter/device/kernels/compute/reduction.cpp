// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t accumulator_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core_width_output = get_compile_time_arg_val(3);
    binary_op_init_common(accumulator_cb_id, accumulator_cb_id, output_tensor_cb_id);
    add_tiles_init(accumulator_cb_id, accumulator_cb_id, true);

    UNPACK(DPRINT << "waiting on cb" << accumulator_cb_id << " num tiles:" << num_devices << ENDL());
    cb_wait_front(accumulator_cb_id, num_devices);

    PACK(DPRINT << "reserving back on cb" << output_tensor_cb_id << " tiles:" << tiles_per_core_width_output << ENDL());
    cb_reserve_back(output_tensor_cb_id, tiles_per_core_width_output);

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_devices; i += 2) {
        add_tiles(accumulator_cb_id, accumulator_cb_id, i, i + 1, 0);
    }
    tile_regs_commit();

    // Pack output tiles
    tile_regs_wait();
    pack_tile(0, output_tensor_cb_id);
    tile_regs_release();

    UNPACK(DPRINT << "popping front from cb " << accumulator_cb_id << " tiles: " << num_devices << ENDL());
    cb_pop_front(accumulator_cb_id, num_devices);
    PACK(DPRINT << "pushing back to cb " << output_tensor_cb_id << " tiles:" << tiles_per_core_width_output << ENDL());
    cb_push_back(output_tensor_cb_id, tiles_per_core_width_output);

    DPRINT << "Kernel finished" << ENDL();
}
}  // namespace NAMESPACE
