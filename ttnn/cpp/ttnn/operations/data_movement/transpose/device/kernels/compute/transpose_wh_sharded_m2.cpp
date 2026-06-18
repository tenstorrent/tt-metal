// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose/device/kernels/compute/transpose_wh_sharded.cpp. The legacy source is
// also used by the (unmigrated) experimental transformer create_qkv_heads* / split_qkv ops; this copy
// is named-binding ported for the transpose WH sharded compute kernel only. Keep the two in sync until
// the legacy copy's last consumer ports. Logic UNCHANGED; only the access
// mechanism moves to named bindings:
//   - input CB c_0  -> dfb::src0 (borrowed input shard; compute consumes)
//   - output CB c_16 -> dfb::out  (borrowed output shard; compute produces)
//   - positional RTAs -> get_arg(args::...)

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t NHtWt = get_arg(args::NHtWt);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);

    constexpr auto cb_id_in = dfb::src0;
    constexpr auto cb_id_out = dfb::out;

    transpose_wh_init(cb_id_in, cb_id_out);

    DataflowBuffer cb_in(cb_id_in);
    DataflowBuffer cb_out(cb_id_out);

    // transpose a row-major block:
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile

    uint32_t tile_idx = 0;
    uint32_t tile_idx_N = 0;

    cb_in.wait_front(NHtWt);
    cb_out.reserve_back(NHtWt);
    for (uint32_t n = 0; n < N; ++n) {
        tile_idx = tile_idx_N;
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t h = 0; h < Ht; ++h) {
                tile_regs_acquire();
                transpose_wh_tile(cb_id_in, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_id_out);
                tile_regs_release();
                tile_idx += Wt;
            }
            tile_idx = tile_idx - HtWt + 1;
        }
        tile_idx_N += HtWt;
    }
    cb_out.push_back(NHtWt);
    cb_in.pop_front(NHtWt);
}
