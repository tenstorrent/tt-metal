// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of eltwise_typecast.cpp (op-private copy). The legacy compute kernel is still consumed
// positionally by the legacy (un-migrated) TypecastShardedProgramFactory and must not be touched, so the
// migrated interleaved / subgrid / row-major-chunked factories carry their own copy here. Only the binding
// mechanism changed: the CB ids come from the DFB binding tokens (dfb::), and per_core_block_cnt /
// per_core_block_dim from named compile-time args (args::). The typecast loop and LLK calls are preserved.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);
    constexpr uint32_t input_cb = dfb::input_cb;
    constexpr uint32_t output_cb = dfb::output_cb;

    init_sfpu(input_cb, output_cb);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(output_cb, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(input_cb, 1);

            copy_tile(input_cb, 0, 0);

            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, output_cb);

            cb_pop_front(input_cb, 1);

            tile_regs_release();
        }
        cb_push_back(output_cb, per_core_block_dim);
    }
}
