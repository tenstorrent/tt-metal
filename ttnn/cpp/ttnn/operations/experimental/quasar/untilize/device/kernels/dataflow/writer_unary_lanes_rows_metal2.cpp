// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes untilize's lane-wide row entries to the row-major output with implicit sync. Output entry k
// is row k / num_lanes of lane k % num_lanes (the strided DFB interleaves the lanes' tile counters), and
// thread t of N drains the entries t, t + N, .... With COLUMN_LANES the entry is a segment of a block's
// row at the lane's column offset; otherwise it is a whole row of the lane's block. The output is
// interleaved, or with LOCAL_SHARD this core's output shard.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t lane_rows = get_arg(args::lane_rows);  // rows each lane produces on this core
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t num_lanes = get_arg(args::num_lanes);
    constexpr uint32_t lane_bytes = get_arg(args::lane_bytes);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);
    const auto s = TensorAccessor(tensor::output);
#if LOCAL_SHARD
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);
    const uint32_t shard_base = s.get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];
#endif

    for (uint32_t k = get_my_thread_id(); k < num_lanes * lane_rows; k += get_num_threads()) {
        const uint32_t lane = k % num_lanes;
        const uint32_t q = k / num_lanes;
        const uint32_t m = q / tile_height;
        const uint32_t row =
            (COLUMN_LANES ? first_block + m : first_block + lane + num_lanes * m) * tile_height + q % tile_height;
        const uint32_t offset = COLUMN_LANES ? lane * lane_bytes : 0;
#if LOCAL_SHARD
        noc.async_write<NocOptions::TXN_ID>(
            cb_out, self_ep, {}, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + row * row_bytes + offset});
#else
        noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = row, .offset_bytes = offset});
#endif
    }
}
