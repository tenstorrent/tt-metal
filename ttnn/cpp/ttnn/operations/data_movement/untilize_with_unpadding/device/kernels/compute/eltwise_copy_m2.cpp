// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp.
// The legacy source is shared, so it is forked here (not edited in place) and ported to
// Metal 2.0 named bindings for untilize_with_unpadding's multi-core sharded factory (the
// unpad-width-16 path uses copy compute purely for a potential data type conversion).
//   tt::CBIndex::c_0  -> dfb::in
//   tt::CBIndex::c_16 -> dfb::out
//   per_core_tile_cnt CTA -> named CTA

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    unary_op_init_common(dfb::in, dfb::out);
    copy_tile_init(dfb::in);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(dfb::in, 1);
        cb_reserve_back(dfb::out, 1);
        copy_tile(dfb::in, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);

        cb_pop_front(dfb::in, 1);
        cb_push_back(dfb::out, 1);

        tile_regs_release();
    }
}
