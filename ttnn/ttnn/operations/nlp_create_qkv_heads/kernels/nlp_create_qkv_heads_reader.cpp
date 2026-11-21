// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_create_qkv_heads reader, block operation `load_block` (op_design.md, regime tile_relocate).
//
// This cluster owns the contiguous input page range [start_page, start_page + num_pages), cut into
// blocks_per_core blocks. Every page is read once, by one tagged NoC read straight into the ring
// `cb_input_tiles` (implicit sync: the DM0 interrupt handler posts the credits per transaction-id slice, so
// there is no reserve / barrier / push). The STRIDED ring deals entries round-robin over the reader harts:
// hart t takes the cluster's pages t, t+R, ... — the index runs across block boundaries so the dealing never
// misaligns with the ring.
//
// No helper: the kernel library has no dataflow helper for a tagged accessor-to-ring read (op_design.md,
// "Helpers considered and rejected"); the plain Noc call is the ring's implicit-sync API.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t num_pages = get_arg(args::num_pages);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t blocks_per_core = get_arg(args::blocks_per_core);

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    Noc noc;
    const auto acc_input = TensorAccessor(tensor::input);

    const uint32_t block_pages = (num_pages + blocks_per_core - 1) / blocks_per_core;
    uint32_t i = get_my_thread_id();  // this hart's next page, local to the cluster's range
    for (uint32_t block_end = block_pages; i < num_pages; block_end += block_pages) {
        // load_block: this hart's share of pages [block_end - block_pages, block_end)
        const uint32_t end = block_end < num_pages ? block_end : num_pages;
        for (; i < end; i += R) {
            noc.async_read<NocOptions::TXN_ID>(acc_input, cb_input_tiles, {.page_id = start_page + i}, {});
        }
    }
    // Firmware credits the last partial batch when the kernel returns. Hand the hart's sticky tag back.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
