// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_layout identity_copy reader (copy_page_block): the input already has the requested layout. One tagged
// implicit-sync read per page into staged_input (= cb_pages, a DM -> DM ring whose capacity is
// BUFFER_DEPTH x COPY_BLOCK_PAGES pages); the DM0 interrupt handler posts the credits.

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
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer staged(dfb::staged_input);
    Noc noc;
    const auto acc = TensorAccessor(tensor::input);

    for (uint32_t i = get_my_thread_id(); i < num_units; i += R) {
        noc.async_read<NocOptions::TXN_ID>(acc, staged, {.page_id = unit_start + i}, {});
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
