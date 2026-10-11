// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "internal/tt-2xx/quasar/cache.h"
#include "experimental/kernel_args.h"

// Feeds tilize's compute threads ("lanes") one row segment per DFB entry with implicit sync. Reader
// thread t serves lane t alone: with as many readers as compute threads the strided DFB pairs producer
// t with consumer t. With COLUMN_LANES lane t takes its lane_bytes-wide column slice of every block (a
// block is tile_height rows); otherwise it takes every num_threads-th block whole. Rows past the input
// are read from a scratch row filled with the pad value, so every entry is one TXN_ID read.
void kernel_main() {
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t lane_bytes = get_arg(args::lane_bytes);

    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_lane_blocks = get_arg(args::num_lane_blocks);
    const uint32_t num_input_rows = get_arg(args::num_input_rows);
    const uint32_t pad_value = get_arg(args::pad_value);  // pad value replicated to fill 32 bits

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const auto s0 = TensorAccessor(tensor::input);
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    const uint32_t lane = get_my_thread_id();
    const uint32_t num_lanes = get_num_threads();
    Scratchpad<uint32_t> scratch(scratch::pad);
    const uint32_t pad_row_addr = scratch.get_base_address() + lane * lane_bytes;
    auto* pad_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_row_addr);
    for (uint32_t i = 0; i < lane_bytes / sizeof(uint32_t); ++i) {
        pad_row[i] = pad_value;
    }
    // CPU stores sit in the L2 cache; flush them to L1 so the NoC reads below see the pad row.
    flush_l2_cache_range(static_cast<uintptr_t>(pad_row_addr), static_cast<size_t>(lane_bytes));

    const uint32_t col_offset_bytes = COLUMN_LANES ? lane * lane_bytes : 0;
    for (uint32_t m = 0; m < num_lane_blocks; ++m) {
        const uint32_t block = COLUMN_LANES ? first_block + m : first_block + lane + num_lanes * m;
        for (uint32_t j = 0; j < tile_height; ++j) {
            const uint32_t row = block * tile_height + j;
            if (row < num_input_rows) {
                noc.async_read<NocOptions::TXN_ID>(s0, cb_in, {.page_id = row, .offset_bytes = col_offset_bytes}, {});
            } else {
                noc.async_read<NocOptions::TXN_ID>(
                    self_ep, cb_in, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_row_addr}, {});
            }
        }
    }
}
