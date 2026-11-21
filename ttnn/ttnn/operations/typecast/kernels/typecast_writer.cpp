// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// typecast writer (store_block): cb_output_tiles to the output's tiles start_id .. start_id + num_tiles - 1,
// dealt round-robin over the W writer harts (hart w writes tiles w, w+W, ...).
//
// Implicit sync: one tagged write per ring entry; the DM0 interrupt handler acks the entry once sent.

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
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    Noc noc;
    const auto output_accessor = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += W) {
        noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, output_accessor, {}, {.page_id = start_id + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns. Hand the tag register back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
