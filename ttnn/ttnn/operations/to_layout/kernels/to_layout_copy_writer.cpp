// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_layout identity_copy writer (copy_page_block): one tagged implicit-sync write per page out of
// staged_output (= cb_pages); the DM0 interrupt handler acks the entries.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t num_units = get_arg(args::num_units);
    constexpr uint32_t W = get_arg(args::writer_threads);

    DataflowBuffer staged(dfb::staged_output);
    Noc noc;
    const auto acc = TensorAccessor(tensor::output);

    for (uint32_t i = get_my_thread_id(); i < num_units; i += W) {
        noc.async_write<NocOptions::TXN_ID>(staged, acc, {}, {.page_id = unit_start + i});
    }
    noc_async_write_set_trid(0, noc.get_noc_id());
}
