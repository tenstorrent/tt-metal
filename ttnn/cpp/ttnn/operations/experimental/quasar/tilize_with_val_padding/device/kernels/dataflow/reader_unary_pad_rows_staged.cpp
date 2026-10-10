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

// Feeds tilize one sub-block at a time: tile_height rows of sub_block_tiles tiles, one DFB entry per
// row segment. Thread t of N reads the sub-blocks t, t + N, ...; the ALL-pattern DFB gives each thread
// its own contiguous region and lets compute take sub-blocks from the threads in turn. Rows past the
// input are read from a scratch row filled with the pad value, so every entry is one TXN_ID read with
// implicit sync.
void kernel_main() {
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t sub_blocks_per_row = get_arg(args::sub_blocks_per_row);
    constexpr uint32_t row_seg_bytes = get_arg(args::row_seg_bytes);

    const uint32_t num_sub_blocks = get_arg(args::num_sub_blocks);
    const uint32_t start_row = get_arg(args::start_row);  // first padded row of this core's first block
    const uint32_t num_input_rows = get_arg(args::num_input_rows);
    const uint32_t pad_value = get_arg(args::pad_value);  // pad value replicated to fill 32 bits

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const auto s0 = TensorAccessor(tensor::input);
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    const uint32_t thread_id = get_my_thread_id();
    Scratchpad<uint32_t> scratch(scratch::pad);
    const uint32_t pad_row_addr = scratch.get_base_address() + thread_id * row_seg_bytes;
    auto* pad_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pad_row_addr);
    for (uint32_t i = 0; i < row_seg_bytes / sizeof(uint32_t); ++i) {
        pad_row[i] = pad_value;
    }
    // CPU stores sit in the L2 cache; flush them to L1 so the NoC reads below see the pad row.
    flush_l2_cache_range(static_cast<uintptr_t>(pad_row_addr), static_cast<size_t>(row_seg_bytes));

    for (uint32_t s = thread_id; s < num_sub_blocks; s += get_num_threads()) {
        const uint32_t first_row = start_row + (s / sub_blocks_per_row) * tile_height;
        const uint32_t col_offset_bytes = (s % sub_blocks_per_row) * row_seg_bytes;
        for (uint32_t j = 0; j < tile_height; ++j) {
            const uint32_t row = first_row + j;
            if (row < num_input_rows) {
                noc.async_read<NocOptions::TXN_ID>(s0, cb_in, {.page_id = row, .offset_bytes = col_offset_bytes}, {});
            } else {
                noc.async_read<NocOptions::TXN_ID>(
                    self_ep, cb_in, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pad_row_addr}, {});
            }
        }
    }
}
