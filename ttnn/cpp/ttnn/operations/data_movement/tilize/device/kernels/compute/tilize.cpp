// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    tilize_init(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

    DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
    DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
    DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(cb_id_in0, per_core_block_tile_cnt);
        cb_reserve_back(cb_id_out0, per_core_block_tile_cnt);

        DPRINT_UNPACK({ DPRINT << "cb_id_in0 is: " << TSLICE(cb_id_in0, 0, SliceRange::h0_w0_32()) << ENDL(); });

        tilize_block(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

        DPRINT_UNPACK({ DPRINT << "cb_id_out0 is: " << TSLICE(cb_id_out0, 0, SliceRange::h0_w0_32()) << ENDL(); });

        cb_push_back(cb_id_out0, per_core_block_tile_cnt);
        cb_pop_front(cb_id_in0, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
