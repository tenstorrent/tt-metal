// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PackTileBlock validation kernel — atomic multi-tile pack via `pack_tile_block`.
// Loads N tiles via CopyTile (one tile per iter), but exec is a single block-pack of N tiles.
//
// Strategy: caller-managed (NoWaitNoPop on input, NoReserveNoPush on output) — the
// reader/writer handle CB lifecycle externally. The compute kernel iterates `num_tiles / N_BLOCK`
// times, each iteration doing N_BLOCK copies into consecutive DEST slots, then atomic pack.

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

#ifndef N_BLOCK
#define N_BLOCK 4
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n_block = N_BLOCK;

    const uint32_t per_core_block_count = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t num_tiles = per_core_block_count * per_core_block_dim;

    // For each n_block-sized window, do n_block copies + one block pack.
    // Use caller-managed lifecycle since the chain combinator can't natively
    // express "atomic pack-N-after-N-copies".
    const uint32_t num_blocks = num_tiles / n_block;

    // Initialize once.
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);
    pack_reconfig_data_format(cb_out);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        // Wait for n_block input tiles + reserve n_block output tiles.
        cb_wait_front(cb_in, n_block);
        cb_reserve_back(cb_out, n_block);

        tile_regs_acquire();
        for (uint32_t i = 0; i < n_block; ++i) {
            copy_tile(cb_in, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        // Atomic block pack — n_block consecutive DEST slots → cb_out.
        pack_tile_block(0, cb_out, n_block);
        tile_regs_release();

        cb_pop_front(cb_in, n_block);
        cb_push_back(cb_out, n_block);
    }
}
