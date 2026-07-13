// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp.
// The shared compute kernel is still used by sibling ops on the legacy ProgramDescriptor
// path (nlp_create_qkv_heads_boltz / _vit, split_query_key_value_and_split_heads), so this
// op's Metal 2.0 Interleaved factory binds a forked copy with named args + DFB handles.

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto NHtWt = get_arg(args::NHtWt);
    transpose_wh_init(dfb::in_k, dfb::out_k);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(dfb::in_k, 1);
        cb_reserve_back(dfb::out_k, 1);

        tile_regs_acquire();
        transpose_wh_tile(dfb::in_k, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out_k);
        tile_regs_release();

        cb_push_back(dfb::out_k, 1);
        cb_pop_front(dfb::in_k, 1);
    }
}
