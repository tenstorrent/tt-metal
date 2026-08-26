// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// eltwise_copy_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t onetile = 1;

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    CircularBuffer cb_in(tt::CBIndex::c_0);
    CircularBuffer cb_out(tt::CBIndex::c_16);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_in.wait_front(onetile);
        cb_out.reserve_back(onetile);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb_in.pop_front(onetile);
        cb_out.push_back(onetile);

        tile_regs_release();
    }
}
