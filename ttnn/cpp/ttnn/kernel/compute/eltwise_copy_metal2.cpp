// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of eltwise_copy.cpp.
//
// Per-tile copy from input DFB to output DFB, used for data-format conversion in
// sharded_to_interleaved when the input and output dtypes differ.
//
// Bindings (named, from host KernelSpec):
//   dfb::src                — DFB endpoint (CONSUMER) — input shard data
//   dfb::dst                — DFB endpoint (PRODUCER) — converted output, fed to writer
//   args::per_core_tile_cnt — CTA: number of tiles to copy on this core

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    // dfb::src and dfb::dst implicitly convert to uint32_t for LLK APIs on WH/BH.
    unary_op_init_common(dfb::src, dfb::dst);
    copy_tile_init(dfb::src);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(dfb::src, 1);
        cb_reserve_back(dfb::dst, 1);
        copy_tile(dfb::src, 0, 0);

        pack_tile(0, dfb::dst);

        cb_pop_front(dfb::src, 1);
        cb_push_back(dfb::dst, 1);

        release_dst();
    }
}
