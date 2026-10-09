// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar nearest-neighbour upsample, sharded (height / block) row-major input: CONSUMER side.
//
// SPMD data-movement kernel with the same thread count T as the reader, so ring entry i (= output stick i of
// this core's shard) is produced and consumed by thread i % T. Each entry is written into the local output
// shard with ONE implicit-sync NoC write, addressed through a TensorAccessor over the sharded OUTPUT tensor
// (page = one shard-wide row, block sharding: ncols pages per row):
//
//     noc.async_write<NocOptions::TXN_ID>(stage, output, {}, {.page_id = dst_page});
//
// The write is stamped with a DFB transaction id and the DM0 ISR acks the entry (frees it for the reader)
// when the NoC reports it sent; finish() handles the tail, write_barrier() makes sure every byte has landed
// before the kernel returns.
//
// Compile args: pages_per_row, out_nsticks_per_core
// Runtime args: out_row_start, col (same as the reader)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t pages_per_row = get_arg(args::pages_per_row);
    constexpr uint32_t out_nsticks_per_core = get_arg(args::out_nsticks_per_core);
    const uint32_t out_row_start = get_arg(args::out_row_start);
    const uint32_t col = get_arg(args::col);

    const uint32_t num_threads = get_num_threads();
    const uint32_t thread_id = get_my_thread_id();

    const auto output = TensorAccessor(tensor::output);
    DataflowBuffer stage(dfb::stage);
    Noc noc;

    for (uint32_t i = thread_id; i < out_nsticks_per_core; i += num_threads) {
        const uint32_t dst_page = (out_row_start + i) * pages_per_row + col;
        noc.async_write<NocOptions::TXN_ID>(stage, output, {}, {.page_id = dst_page});
    }
    stage.finish();
    stage.write_barrier(noc);
}
