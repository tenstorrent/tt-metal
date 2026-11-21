// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// concat writer: store_block. Drains cb_output_tiles to the output tensor with implicit sync: one write tagged
// NocOptions::TXN_ID per ring entry, leaving from the ring's cursor; the DM0 interrupt handler acks each entry
// once it is sent, so there is no wait_front / barrier / pop_front. The reader fills the ring in blocks (explicit
// sync in run_assembly, implicit in page_copy); entries reach this side in output page order either way.
//
// Writer thread w of W writes this cluster's output pages start_tile + w, + w + W, ... (STRIDED ring).
//
// No helper: the op has no compute phase, and a whole-page NoC write is the store itself (op_design.md, API
// Mapping, "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile = get_arg(args::start_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += W) {
        noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, acc_out, {}, {.page_id = start_tile + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns. Hand the tag register back with the
    // user id: it is sticky, and a tt-metal program that runs next issues untagged transfers.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
