// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// max_pool2d writer: writer thread w of W. Block operation store_pooled_rows (op_design.md).
//
// The assignment's output rows are contiguous in its owner's output shard, `pitch_bytes` apart, starting at page
// `out_page`: its output stream is that byte range, and stream tile u is bytes [u*tile_bytes, +tile_bytes) of it.
// Each popped entry is therefore one write; the last tile of the assignment is partial and written to its exact
// length, so nothing lands past the assignment's rows. A recomputed (padded-round) tile is written again with the
// same bytes to the same place.
//
// Dealing mirrors the reader: in block b, NEO t owns tiles [b*C*bt + t*bt_b, + bt_b); writer thread w drains NEOs
// w, w+W, ... one entry each in strict rotation (STRIDED ring).
//
// Explicit sync on `pooled_tiles` (the host opts it out of implicit sync): the write length is not an entry. Every
// transfer carries the user transaction id 0, and the hart hands the tag back at the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t out_page = get_arg(args::out_page);
    const uint32_t pos_count = get_arg(args::pos_count);
    const uint32_t T = get_arg(args::num_tiles);
    const uint32_t bt = get_arg(args::block_tiles);
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t pitch_bytes = get_arg(args::pitch_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};
    static_assert(C % W == 0, "W must divide C on a STRIDED ring");

    DataflowBuffer pooled(dfb::pooled_tiles);
    Noc noc;
    const auto out = TensorAccessor(tensor::output);

    const uint32_t thread = get_my_thread_id();
    const uint32_t total_bytes = pos_count * pitch_bytes;
    const uint32_t Q = (T + C - 1) / C;
    const uint32_t num_blocks = (Q + bt - 1) / bt;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t bt_b = Q - b * bt < bt ? Q - b * bt : bt;
        const uint32_t u0 = b * C * bt;
        for (uint32_t i = 0; i < bt_b; ++i) {
            for (uint32_t g = 0; g < C / W; ++g) {
                const uint32_t neo = thread + g * W;
                uint32_t u = u0 + neo * bt_b + i;
                if (u >= T) {
                    u = T - 1;  // padded round: the NEO recomputed the last real tile
                }
                const uint32_t offset = u * tile_bytes;
                const uint32_t len = total_bytes - offset < tile_bytes ? total_bytes - offset : tile_bytes;
                pooled.wait_front(1);
                noc.async_write<NocOptions::TXN_ID>(
                    pooled, out, len, {.offset_bytes = 0}, {.page_id = out_page, .offset_bytes = offset}, user_txn);
                noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                pooled.pop_front(1);
            }
        }
    }
    // Hand the hart back with the user id in its tag register (target page, ring protocols).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
