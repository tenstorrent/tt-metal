// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    CircularBuffer cb_in(tt::CBIndex::c_0);
    CircularBuffer cb_out(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);
    copy_init(tt::CBIndex::c_0);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // Pop tile after tile, copy to DST and pack
        cb_in.wait_front(1);

        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        cb_in.pop_front(1);

        cb_out.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        cb_out.push_back(1);
    }
}
