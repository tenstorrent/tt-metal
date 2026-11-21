// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// subtract reader — load_block: this cluster's contiguous tile range [start_id, start_id + num_tiles) of the
// minuend (a) and the subtrahend (b) into their STRIDED rings.
//
// Implicit sync (the target page's protocol for a ring a NoC transfer fills): one read tagged
// NocOptions::TXN_ID per entry, no reserve / barrier / push; the DM0 interrupt handler posts the credits.
// Reader hart p of R reads tiles p, p+R, ... (R | C keeps each hart's turn on the same NEOs).
//
// No kernel-library dataflow helper wraps the Quasar implicit-sync tagged read of an interleaved accessor
// (op_design.md, API Mapping), so the raw tagged call is used, as in tt-metal's Quasar ops.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t reader_threads = get_arg(args::reader_threads);

    DataflowBuffer cb_minuend_tiles(dfb::cb_minuend_tiles);
    DataflowBuffer cb_subtrahend_tiles(dfb::cb_subtrahend_tiles);
    Noc noc;
    const auto minuend = TensorAccessor(tensor::minuend);
    const auto subtrahend = TensorAccessor(tensor::subtrahend);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += reader_threads) {
        noc.async_read<NocOptions::TXN_ID>(minuend, cb_minuend_tiles, {.page_id = start_id + i}, {});
        noc.async_read<NocOptions::TXN_ID>(subtrahend, cb_subtrahend_tiles, {.page_id = start_id + i}, {});
    }
    // No closing barrier: firmware credits the last batch and drains the rings when the kernel returns.
    // Hand the hart's transaction tag back.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
