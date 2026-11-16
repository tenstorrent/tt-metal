// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_padded_tiles_per_batch = get_arg(args::num_padded_tiles_per_batch);
    auto num_unpadded_rows_per_batch = get_arg(args::num_unpadded_rows_per_batch);
    auto padded_block_row_size_bytes = get_arg(args::padded_block_row_size_bytes);
    auto unpadded_block_row_size_bytes = get_arg(args::unpadded_block_row_size_bytes);
    auto batch = get_arg(args::batch);

    constexpr auto aligned_page_size = get_arg(args::aligned_page_size);

    Noc noc;
    // The untilized block the compute kernel packs; drained here row by row.
    DataflowBuffer dfb_untilize_out(dfb::untilize_out);
    // The output shard on this core, which this kernel writes in place; no other kernel accesses it.
    LocalTensorAccessor<uint32_t> out_shard(tensor::out);

    // Byte offset of the next output row within the shard.
    uint32_t write_offset = 0;

    for (uint32_t b = 0; b < batch; ++b) {
        dfb_untilize_out.wait_front(num_padded_tiles_per_batch);
        uint32_t src_addr = dfb_untilize_out.get_read_ptr();

        for (uint32_t row = 0; row < num_unpadded_rows_per_batch; ++row) {
            noc.async_read(
                UnicastEndpoint{},
                out_shard,
                unpadded_block_row_size_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = src_addr},
                {.offset_bytes = write_offset});
            src_addr += padded_block_row_size_bytes;
            write_offset += aligned_page_size;
        }

        noc.async_read_barrier();
        dfb_untilize_out.pop_front(num_padded_tiles_per_batch);
    }
}
