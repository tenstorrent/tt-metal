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

// Drains the row staging DFB into this core's output shard. The readers fill the ring with one shard
// row per entry, so the shard itself needs no DFB and its size is not bounded by the DFB's per-txn-ID
// entry count.
void kernel_main() {
    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t row_bytes = get_arg(args::padded_block_width_bytes);

    Noc noc;
    DataflowBuffer cb_stage(dfb::stage);
    const uint32_t shard_base = TensorAccessor(tensor::dst).get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    // Thread t of N copies shard rows t, t + N, ...: the strided ring hands it exactly those rows, in
    // order. Each TXN_ID write drains the next staged row and acks it when it lands.
    for (uint32_t h = get_my_thread_id(); h < block_height; h += get_num_threads()) {
        noc.async_write<NocOptions::TXN_ID>(
            cb_stage, self_ep, {}, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + h * row_bytes});
    }
}
