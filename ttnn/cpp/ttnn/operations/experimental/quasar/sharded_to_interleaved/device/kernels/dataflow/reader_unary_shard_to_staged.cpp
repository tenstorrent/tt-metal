// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Copies this core's input shard into the staging DFB. It walks the block's tiles in row-major order
// and skips the shard's padding, so the writers only see real tiles, and the shard itself needs no
// DFB.
void kernel_main() {
    const uint32_t num_entries = get_arg(args::num_entries);
    const uint32_t block_width = get_arg(args::block_width);  // entries per block row
    const uint32_t shard_width = get_arg(args::shard_width);  // entry slots per shard row
    const uint32_t entry_bytes = get_arg(args::entry_bytes);

    Noc noc;
    DataflowBuffer cb_stage(dfb::stage);
    const uint32_t shard_base = TensorAccessor(tensor::src).get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    // Thread t of N stages block entries t, t + N, ...: the strided ring takes them in that order.
    // Each TXN_ID read fills the next ring entry and posts its credit when it lands.
    for (uint32_t k = get_my_thread_id(); k < num_entries; k += get_num_threads()) {
        const uint32_t slot = (k / block_width) * shard_width + k % block_width;
        noc.async_read<NocOptions::TXN_ID>(
            self_ep, cb_stage, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + slot * entry_bytes}, {});
    }
}
