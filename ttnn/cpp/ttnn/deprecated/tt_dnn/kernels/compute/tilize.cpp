// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

// #include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "tilize kernel started!!!" << ENDL();
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    // UNPACK(( DPRINT << "Block count=" << uint32_t(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt <<
    // ENDL() ));
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

    DPRINT_MATH(DPRINT << "this is the math kernel" << ENDL());
    DPRINT_PACK(DPRINT << "this is the pack kernel" << ENDL());
    DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    DPRINT_DATA0(DPRINT << "this is the data movement kernel on noc 0" << ENDL());
    DPRINT_DATA1(DPRINT << "this is the data movement kernel on noc 1" << ENDL());

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        DPRINT << "Processing block " << b << ENDL();
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        DPRINT_UNPACK({ DPRINT << "cb_in is: " << TSLICE(tt::CBIndex::c_0, 0, SliceRange::h0_w0_32()) << ENDL(); });

        tilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        DPRINT_PACK({ DPRINT << "cb_out is: " << TSLICE(tt::CBIndex::c_16, 0, SliceRange::h0_w0_32()) << ENDL(); });

        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
