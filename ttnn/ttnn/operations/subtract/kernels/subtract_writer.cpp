// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// subtract writer — store_block: this cluster's contiguous tile range [start_id, start_id + num_tiles) of the
// difference, drained from its STRIDED ring.
//
// Implicit sync: one write tagged NocOptions::TXN_ID per entry, no wait / barrier / pop; the DM0 interrupt
// handler acks each entry once sent. Writer hart w of W writes tiles w, w+W, ...

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
    constexpr uint32_t writer_threads = get_arg(args::writer_threads);

    DataflowBuffer cb_difference_tiles(dfb::cb_difference_tiles);
    Noc noc;
    const auto difference = TensorAccessor(tensor::difference);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += writer_threads) {
        noc.async_write<NocOptions::TXN_ID>(cb_difference_tiles, difference, {}, {.page_id = start_id + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns. Hand the tag back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
