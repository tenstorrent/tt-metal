// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose_wh_sharded.cpp (see the legacy copy alongside).
// Compute logic is unchanged; resource access uses Metal 2.0 named handles:
//   - cb_id_in / cb_id_out (legacy CB-index CTAs) -> dfb::in / dfb::out
//   - NHtWt / HtWt / N / Ht / Wt (legacy RTAs 0-4) -> named RTAs
// Forked (not modified in place) because the legacy copy is still bound by the
// create_qkv_heads, create_qkv_heads_from_separate_tensors and
// split_query_key_value_and_split_heads_sharded ops, which are not on Metal 2.0.

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

    constexpr auto dfb_in_id = dfb::in;
    constexpr auto dfb_out_id = dfb::out;

    compute_kernel_hw_startup(dfb_in_id, dfb_out_id);
    transpose_init(dfb_in_id);

    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_out(dfb_out_id);

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
                transpose_tile(dfb_in_id, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_out_id);
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
