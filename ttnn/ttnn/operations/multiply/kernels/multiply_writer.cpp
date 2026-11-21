// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// multiply writer, implicit sync in the interleaved regimes (tagged writes; the DM0 ISR acks the ring entries).
//
// flat_stream (regime 0): store_stream — hart w of W writes this cluster's tiles w, w+W, ...
// row_bcast_strips (regime 1): store_block — per block of the shared walker, the block's rows x cols output
//   tiles row-major, in the order compute packs them.
// sharded_local (regime 2): no writes. The out ring is borrowed from the output's resident shard, so the
//   packed tiles are already in place; this kernel is the ring's consumer only to retire them: one wait /
//   pop over the whole shard (explicit sync, the ring listed in disable_dfb_implicit_sync_for).
//
// The kernel's last write carried a ring's id, so it hands the tag register back at the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "multiply_block_walker.hpp"

void kernel_main() {
    constexpr uint32_t regime = get_arg(args::regime);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t block_cols = get_arg(args::block_cols);
    const uint32_t start = get_arg(args::start);
    const uint32_t count = get_arg(args::count);

    DataflowBuffer dfb_out(dfb::out);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);

    if constexpr (regime == 0) {
        for (uint32_t i = get_my_thread_id(); i < count; i += W) {
            noc.async_write<NocOptions::TXN_ID>(dfb_out, acc_out, {}, {.page_id = start + i});
        }
    } else if constexpr (regime == 2) {
        static_assert(regime != 2 || W == 1, "sharded_local retires whole shards: one writer hart");
        dfb_out.wait_front(count);
        dfb_out.pop_front(count);
    } else {
        static_assert(regime != 1 || W == 1, "row_bcast_strips deals whole blocks: one writer hart");
        const uint32_t G = get_arg(args::G);
        const uint32_t Ht = get_arg(args::Ht);
        const uint32_t Wt = get_arg(args::Wt);

        multiply::BlockWalker walker(start, count, G, Ht, Wt, block_cols);
        multiply::Block blk;
        while (walker.next(blk)) {
            uint32_t out_row = (blk.n * Ht + blk.row_begin) * Wt + blk.col_begin;
            for (uint32_t r = 0; r < blk.rows; ++r, out_row += Wt) {
                for (uint32_t j = 0; j < blk.cols; ++j) {
                    noc.async_write<NocOptions::TXN_ID>(dfb_out, acc_out, {}, {.page_id = out_row + j});
                }
            }
        }
    }
    // Firmware waits for the tagged writes to land when the kernel returns; hand the tag register back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
