// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// sigmoid writer (store_block): writer thread w of W writes this cluster's pages w, w+W, ... of its run
// [start_id, start_id + num_tiles) from cb_output_tiles. The STRIDED ring hands entry k from NEO k mod C, the
// NEO that computed page k.
//
// Implicit sync: one transaction-tagged write per entry; the DM0 interrupt handler acks each entry once sent,
// so there is no wait / barrier / pop. The tag register is handed back as the last statement (see the reader).

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
    // Firmware waits for the tagged writes to land when the kernel returns.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
