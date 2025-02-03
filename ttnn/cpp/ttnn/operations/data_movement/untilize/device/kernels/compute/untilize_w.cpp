// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);
    untilize_init(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // UNPACK(( DPRINT << "Block count=" << uint32_t(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt <<
    // ENDL() ));

    // UNPACK (( DPRINT <<  "per_core_block_cnt: " << per_core_block_cnt << ENDL()));
    // UNPACK (( DPRINT <<  "per_core_block_tile_cnt: " << per_core_block_tile_cnt << ENDL()));
    // UNPACK (( DPRINT <<  "third_dim: " << third_dim << ENDL()));

    for (uint32_t b = 0; b < per_core_block_cnt * per_core_block_tile_cnt * third_dim; ++b) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);

        // for (int32_t r = 0; r < 32; ++r) {
        // SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //     UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_0, 0, sr, false, false) << ENDL() ));
        // }

        untilize_block(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

        // UNPACK (( DPRINT <<  "AFTER UNTILIZE BLOCK " << ENDL()));
        // for (int32_t r = 0; r < 32; ++r) {
        // SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //     UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_16, 0, sr) << ENDL() ));
        // }

        cb_push_back(tt::CBIndex::c_16, 1);
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
}  // namespace NAMESPACE

/*
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);

    pack_untilize_init<1>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < per_core_block_cnt * per_core_block_tile_cnt * third_dim ; ++b) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);

        //for (int32_t r = 0; r < 32; ++r) {
        //SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //    UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_0, 0, sr, false, false) << ENDL() ));
        //}

        pack_untilize_block<1>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

        //for (int32_t r = 0; r < 32; ++r) {
        //SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //    UNPACK(( DPRINT  << TSLICE(tt::CBIndex::c_16, 0, sr) << ENDL() ));
        //}

        cb_push_back(tt::CBIndex::c_16, 1);
        cb_pop_front(tt::CBIndex::c_0, 1);
    }

    pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE

*/
