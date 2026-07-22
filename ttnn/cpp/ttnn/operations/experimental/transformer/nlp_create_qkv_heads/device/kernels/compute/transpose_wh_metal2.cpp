// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of the shared compute donor ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp.
// Forked into this op's directory (rather than modified in place) because the shared original is
// still consumed by legacy (non-ported) siblings: nlp_create_qkv_heads_boltz, nlp_create_qkv_heads_vit,
// split_query_key_value_and_split_heads. The two copies stay behavior-identical; the `_metal2` suffix
// signals the fork and its sunset (delete when the last legacy consumer ports). See METAL2_PORT_REPORT.md.
//
// Named binding tokens: dfb::k_in  = reader->compute input CB (legacy c_0),
//                       dfb::k_out = compute->writer output CB (legacy c_16).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t NHtWt = get_arg(args::NHtWt);
    compute_kernel_hw_startup(dfb::k_in, dfb::k_out);
    transpose_init(dfb::k_in);

    DataflowBuffer cb_in(dfb::k_in);
    DataflowBuffer cb_out(dfb::k_out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(dfb::k_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::k_out);
        tile_regs_release();

        cb_out.push_back(1);
        cb_in.pop_front(1);
    }
}
