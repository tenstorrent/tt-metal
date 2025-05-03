// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        pack_tile(0, tt::CBIndex::c_16);

        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);

        release_dst();
    }
}
}  // namespace NAMESPACE
