// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose/device/kernels/compute/transpose_wh.cpp. The legacy source is also
// used by nlp_create_qkv_heads* / split_query_key_value_and_split_heads / permute_tiled, which have
// not yet migrated; this copy is named-binding ported for the transpose WH (tiled) compute kernel
// only. Keep the two in sync until the legacy copy's last consumer ports.

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto NHtWt = get_arg(args::NHtWt);

    transpose_wh_init(dfb::src0, dfb::out);

    DataflowBuffer cb_in(dfb::src0);
    DataflowBuffer cb_out(dfb::out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_wh_tile(dfb::src0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        cb_out.push_back(1);
        cb_in.pop_front(1);
    }
}
