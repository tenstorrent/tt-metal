// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of transpose_wh_sharded.cpp (cross-op shared with legacy create_qkv_heads{,_from_separate_tensors} /
// split_query_key_value_and_split_heads_sharded). The legacy source stays in place, non-Metal-2.0, for its cross-op
// consumers; only the transpose Metal 2.0 factory binds this fork. Sunset when those consumers migrate.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t NHtWt = get_arg(args::NHtWt);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    // transpose a row-major block:
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile

    uint32_t tile_idx = 0;
    uint32_t tile_idx_N = 0;

    dfb_in.wait_front(NHtWt);
    dfb_out.reserve_back(NHtWt);
    for (uint32_t n = 0; n < N; ++n) {
        tile_idx = tile_idx_N;
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t h = 0; h < Ht; ++h) {
                tile_regs_acquire();
                transpose_tile(dfb::in, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb::out);
                tile_regs_release();
                tile_idx += Wt;
            }
            tile_idx = tile_idx - HtWt + 1;
        }
        tile_idx_N += HtWt;
    }
    dfb_out.push_back(NHtWt);
    dfb_in.pop_front(NHtWt);
}
