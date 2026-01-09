// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "untilize kernel started" << ENDL();
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    untilize_init(src_cb_id);

    DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
    DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
    DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        DPRINT << "Processing block " << b << ENDL();
        cb_wait_front(src_cb_id, per_core_block_tile_cnt);
        cb_reserve_back(out_cb_id, per_core_block_tile_cnt);

        DPRINT_UNPACK({ DPRINT << "src_cb is: " << TSLICE(src_cb_id, 0, SliceRange::h0_w0_32()) << ENDL(); });

        untilize_block(src_cb_id, per_core_block_tile_cnt, out_cb_id);

        DPRINT_PACK({ DPRINT << "out_cb is" << TSLICE(out_cb_id, 0, SliceRange::h0_w0_32()) << ENDL(); });

        cb_push_back(out_cb_id, per_core_block_tile_cnt);
        cb_pop_front(src_cb_id, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
