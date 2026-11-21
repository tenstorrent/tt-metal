// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// div writer (store_block): writer hart w of W drains this cluster's output tiles w, w+W, ... (W | C).
// Implicit sync: one tagged NoC write per ring entry, the DM0 interrupt handler acks the entry once sent.

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

    DataflowBuffer dfb_quotient(dfb::out);
    Noc noc;
    const auto acc_quotient = TensorAccessor(tensor::out);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += W) {
        noc.async_write<NocOptions::TXN_ID>(dfb_quotient, acc_quotient, {}, {.page_id = start_id + i});
    }
    // Firmware waits for the tagged writes to land when the kernel returns; hand the tag back (see reader).
    noc_async_write_set_trid(0, noc.get_noc_id());
}
