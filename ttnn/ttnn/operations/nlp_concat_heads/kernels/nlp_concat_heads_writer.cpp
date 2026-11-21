// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_concat_heads writer (regime `tile_relay`): store_block.
//
// The ring holds this core's output pages in output order, so the writer stores them contiguously:
// [start_page, start_page + core_tiles). Implicit sync: one tagged NoC write per tile leaves from the ring's
// cursor and the DM0 interrupt handler acks the entry; no wait / barrier / pop. Firmware waits for the
// writes to land when the kernel returns.
//
// Thread plan: W writer harts on a STRIDED ring; hart w writes this core's pages w, w + W, ...

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t W = get_arg(args::writer_threads);
    const uint32_t core_tiles = get_arg(args::core_tiles);
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t block_tiles = get_arg(args::block_tiles);

    DataflowBuffer cb_relay_tiles(dfb::relay);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);

    uint32_t i = get_my_thread_id();  // this hart's next page, as an offset into the core's range
    for (uint32_t block_start = 0; block_start < core_tiles; block_start += block_tiles) {
        // store_block: the pages of this block that fall to this hart.
        const uint32_t block_end = block_start + block_tiles < core_tiles ? block_start + block_tiles : core_tiles;
        for (; i < block_end; i += W) {
            noc.async_write<NocOptions::TXN_ID>(cb_relay_tiles, acc_out, {}, {.page_id = start_page + i});
        }
    }
    // Hand the hart's sticky tag register back (see the reader).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
