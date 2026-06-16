// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ported in place to Metal 2.0 named bindings for untilize_with_unpadding's multi-core sharded
// factory (this writer is op-local — only untilize_with_unpadding uses it).
//   cb_id_untilize_out (c_16) -> dfb::out          (untilize compute output, consumed here)
//   cb_id_out          (c_17) -> dfb::sharded_out  (borrowed L1 sharded output, written here)
//   aligned_page_size CTA     -> named CTA
//   runtime args              -> named RTAs

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_unpadded_output_rows = get_arg(args::num_unpadded_output_rows);
    uint32_t num_padded_tiles_per_batch = get_arg(args::num_padded_tiles_per_batch);
    uint32_t num_unpadded_rows_per_batch = get_arg(args::num_unpadded_rows_per_batch);
    uint32_t padded_block_row_size_bytes = get_arg(args::padded_block_row_size_bytes);
    uint32_t unpadded_block_row_size_bytes = get_arg(args::unpadded_block_row_size_bytes);
    uint32_t batch = get_arg(args::batch);

    constexpr uint32_t aligned_page_size = get_arg(args::aligned_page_size);

    Noc noc;
    DataflowBuffer cb_untilize_out(dfb::out);
    DataflowBuffer cb_out(dfb::sharded_out);

    cb_out.reserve_back(num_unpadded_output_rows);
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    for (uint32_t b = 0; b < batch; ++b) {
        cb_untilize_out.wait_front(num_padded_tiles_per_batch);
        uint32_t src_addr = cb_untilize_out.get_read_ptr();

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
        cb_untilize_out.pop_front(num_padded_tiles_per_batch);
    }
    cb_out.push_back(num_unpadded_output_rows);
}
