// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t num_unpadded_output_rows = get_arg_val<uint32_t>(0);
    uint32_t num_padded_tiles_per_batch = get_arg_val<uint32_t>(1);
    uint32_t num_unpadded_rows_per_batch = get_arg_val<uint32_t>(2);
    uint32_t padded_block_row_size_bytes = get_arg_val<uint32_t>(3);
    uint32_t unpadded_block_row_size_bytes = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_untilize_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);

    experimental::CircularBuffer cb_untilize_out(cb_id_untilize_out);
    experimental::CircularBuffer cb_out(cb_id_out);

    cb_out.reserve_back(num_unpadded_output_rows);
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    for (uint32_t b = 0; b < batch; ++b) {
        cb_untilize_out.wait_front(num_padded_tiles_per_batch);
        // Keep legacy NOC API for local L1 reads (sharded kernel)
        uint64_t noc_l1_read_addr = get_noc_addr(cb_untilize_out.get_read_ptr());

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch; ++row) {
            noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            noc_l1_read_addr += padded_block_row_size_bytes;
            l1_write_addr += aligned_page_size;
        }

        noc_async_read_barrier();
        cb_untilize_out.pop_front(num_padded_tiles_per_batch);
    }
    cb_out.push_back(num_unpadded_output_rows);
}
