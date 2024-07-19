// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"


void kernel_main() {
    uint32_t num_unpadded_output_rows       = get_arg_val<uint32_t>(0);
    uint32_t num_padded_tiles_per_batch     = get_arg_val<uint32_t>(1);
    uint32_t num_unpadded_rows_per_batch    = get_arg_val<uint32_t>(2);
    uint32_t padded_block_row_size_bytes    = get_arg_val<uint32_t>(3);
    uint32_t unpadded_block_row_size_bytes  = get_arg_val<uint32_t>(4);
    uint32_t batch                          = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_untilize_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    cb_reserve_back(cb_id_out, num_unpadded_output_rows);
    uint32_t l1_write_addr = get_write_ptr(cb_id_out);

    for(uint32_t b = 0; b < batch; ++b) {
        cb_wait_front(cb_id_untilize_out, num_padded_tiles_per_batch);
        uint64_t noc_l1_read_addr = get_noc_addr(get_read_ptr(cb_id_untilize_out));

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch; ++row) {
            noc_async_read(noc_l1_read_addr, l1_write_addr, unpadded_block_row_size_bytes);
            noc_l1_read_addr += padded_block_row_size_bytes;
            l1_write_addr += unpadded_block_row_size_bytes;
        }

        noc_async_read_barrier();
        cb_pop_front(cb_id_untilize_out, num_padded_tiles_per_batch);
    }
    cb_push_back(cb_id_out, num_unpadded_output_rows);
}
