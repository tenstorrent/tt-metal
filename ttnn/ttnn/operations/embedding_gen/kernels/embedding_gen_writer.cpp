// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// embedding_gen writer: stores each gathered_block entry, already in output layout (op_design.md, store_block).
//
// Raw data-movement API, no kernel-library helper: no helper covers storing an entry as tiles or row slabs at
// page offsets.
//
// Per block, thread t of W taking blocks t, t+W, ...: wait for one entry; issue every write of the block
// (n_cols whole tiles, or n_tok row slabs); one write barrier; pop the entry.
//
// Explicit sync on gathered_block (an entry is drained by several writes): the host lists the ring in
// disable_dfb_implicit_sync_for, and every NoC transfer of this kernel carries user trid 0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "embedding_gen_common.hpp"

void kernel_main() {
    constexpr uint32_t W = get_arg(args::writer_threads);
    constexpr bool out_tiled = get_arg(args::out_tiled) != 0;
    constexpr uint32_t col_unit_bytes = get_arg(args::col_unit_bytes);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};

    const uint32_t start_unit = get_arg(args::start_unit);
    const uint32_t num_units = get_arg(args::num_units);
    const embedding_gen::Geometry geo{
        get_arg(args::S),
        get_arg(args::Dt),
        get_arg(args::block_tokens),
        get_arg(args::block_cols),
        get_arg(args::num_col_chunks),
        get_arg(args::groups_per_batch)};

    DataflowBuffer ring(dfb::gathered_block);
    Noc noc;
    const auto out_acc = TensorAccessor(tensor::out);
    const uint32_t slab_bytes = geo.block_cols * col_unit_bytes;

    for (uint32_t local_block = get_my_thread_id(); local_block < num_units; local_block += W) {
        const embedding_gen::Block blk = geo.block(start_unit + local_block);

        ring.wait_front(1);
        if constexpr (out_tiled) {
            // One block is one output tile-row of `n_cols` tiles: page (b * tile-rows per batch + tg) * Dt + c.
            const uint32_t first_page = (blk.b * geo.groups_per_batch + blk.tg) * geo.Dt + blk.c0;
            for (uint32_t jj = 0; jj < blk.n_cols; ++jj) {
                noc.async_write<NocOptions::TXN_ID>(
                    ring, out_acc, tile_bytes, {.offset_bytes = jj * tile_bytes}, {.page_id = first_page + jj}, user_txn);
            }
        } else {
            // Token k's slab goes to output row b * S + s0 + k, at the chunk's column offset.
            const uint32_t first_row = blk.b * geo.S + blk.s0;
            for (uint32_t k = 0; k < blk.n_tok; ++k) {
                noc.async_write<NocOptions::TXN_ID>(
                    ring,
                    out_acc,
                    blk.n_cols * col_unit_bytes,
                    {.offset_bytes = k * slab_bytes},
                    {.page_id = first_row + k, .offset_bytes = blk.c0 * col_unit_bytes},
                    user_txn);
            }
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
        ring.pop_front(1);
    }

    noc_async_write_set_trid(0, noc.get_noc_id());
}
