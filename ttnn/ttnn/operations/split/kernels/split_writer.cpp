// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// split writer (implicit sync, both regimes): drains cb_output_pages one entry per unit of this core's range,
// writing each entry to its page of the chunk tensor it belongs to. The walk is the reader's (split_walk.hpp),
// so the two agree on which chunk page each entry is. One tagged write per entry; the DM0 interrupt handler
// acks the entry once it is sent.
//
// The chunk tensors are bound as chunk_0 .. chunk_{SPLIT_NUM_CHUNKS-1}; SPLIT_NUM_CHUNKS (<= SPLIT_MAX_CHUNKS) is a define so
// that only bound accessors are named.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "split_walk.hpp"

// SPLIT_MAX_CHUNKS is the host's MAX_CHUNKS (split_program_artifacts.py); this file names accessors chunk_0..chunk_3.
#if !defined(SPLIT_MAX_CHUNKS) || SPLIT_MAX_CHUNKS > 4
#error "split_writer: SPLIT_MAX_CHUNKS must be defined and <= 4 (the accessors this writer names)"
#endif
#if !defined(SPLIT_NUM_CHUNKS) || SPLIT_NUM_CHUNKS < 1 || SPLIT_NUM_CHUNKS > SPLIT_MAX_CHUNKS
#error "split_writer: SPLIT_NUM_CHUNKS must be 1..SPLIT_MAX_CHUNKS"
#endif

void kernel_main() {
    const uint32_t start_unit = get_arg(args::start_unit);
    const uint32_t num_units = get_arg(args::num_units);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer ring(dfb::cb_output_pages);
    Noc noc;
    const auto acc_chunk_0 = TensorAccessor(tensor::chunk_0);
#if SPLIT_NUM_CHUNKS > 1
    const auto acc_chunk_1 = TensorAccessor(tensor::chunk_1);
#endif
#if SPLIT_NUM_CHUNKS > 2
    const auto acc_chunk_2 = TensorAccessor(tensor::chunk_2);
#endif
#if SPLIT_NUM_CHUNKS > 3
    const auto acc_chunk_3 = TensorAccessor(tensor::chunk_3);
#endif

    // store_page: the ring's front entry -> page `page` of chunk `c`.
    auto store_page = [&](uint32_t c, uint32_t page) {
        switch (c) {
#if SPLIT_NUM_CHUNKS > 3
            case 3: noc.async_write<NocOptions::TXN_ID>(ring, acc_chunk_3, {}, {.page_id = page}); break;
#endif
#if SPLIT_NUM_CHUNKS > 2
            case 2: noc.async_write<NocOptions::TXN_ID>(ring, acc_chunk_2, {}, {.page_id = page}); break;
#endif
#if SPLIT_NUM_CHUNKS > 1
            case 1: noc.async_write<NocOptions::TXN_ID>(ring, acc_chunk_1, {}, {.page_id = page}); break;
#endif
            default: noc.async_write<NocOptions::TXN_ID>(ring, acc_chunk_0, {}, {.page_id = page}); break;
        }
    };

#if SPLIT_REGIME == SPLIT_REGIME_PAGE_REMAP
    split::PageRemapWalk walk(get_arg(args::cut_pages), get_arg(args::inner_pages), get_arg(args::chunk_pages));
#elif SPLIT_REGIME == SPLIT_REGIME_TILE_ROW_SHIFT
    split::TileRowShiftWalk walk(
        get_arg(args::h_tiles),
        get_arg(args::w_tiles),
        get_arg(args::split_rows),
        get_arg(args::cut_rows),
        get_arg(args::chunk_tiles),
        get_arg(args::column_tiles));
#else
#error "split_writer: unknown SPLIT_REGIME"
#endif
    // Hart t of W takes units t, t+W, ... of the range, as the ring deals entries.
    walk.seek(start_unit + get_my_thread_id());
    for (uint32_t n = get_my_thread_id(); n < num_units; n += W) {
        store_page(walk.chunk(), walk.output_page());
        walk.advance(W);
    }
    // Firmware waits for the tagged writes to land when the kernel returns. Hand the hart back with the user id
    // in its sticky tag register (see split_reader.cpp).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
