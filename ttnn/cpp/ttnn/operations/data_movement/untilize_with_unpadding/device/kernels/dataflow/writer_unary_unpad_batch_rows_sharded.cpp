// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t num_unpadded_output_rows = get_arg_val<uint32_t>(0);
    uint32_t num_padded_tiles_per_batch = get_arg_val<uint32_t>(1);
    uint32_t num_unpadded_rows_per_batch = get_arg_val<uint32_t>(2);
    uint32_t padded_block_row_size_bytes = get_arg_val<uint32_t>(3);
    uint32_t unpadded_block_row_size_bytes = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(5);

    constexpr uint32_t dfb_id_untilize_out = get_compile_time_arg_val(0);
    constexpr uint32_t dfb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);

    Noc noc;
    DataflowBuffer dfb_untilize_out(dfb_id_untilize_out);
    DataflowBuffer dfb_out(dfb_id_out);

    dfb_out.reserve_back(num_unpadded_output_rows);
    uint32_t l1_write_addr = dfb_out.get_write_ptr();

    for (uint32_t b = 0; b < batch; ++b) {
        dfb_untilize_out.wait_front(num_padded_tiles_per_batch);
        uint32_t src_addr = dfb_untilize_out.get_read_ptr();

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch; ++row) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                unpadded_block_row_size_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = src_addr},
                {.offset_bytes = 0});
            src_addr += padded_block_row_size_bytes;
            l1_write_addr += aligned_page_size;
        }

        noc.async_read_barrier();
        dfb_untilize_out.pop_front(num_padded_tiles_per_batch);
    }
    dfb_out.push_back(num_unpadded_output_rows);
}
