// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t num_blocks = get_arg(args::num_blocks);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t input_Ht = get_arg(args::input_Ht);
    const std::uint32_t cache_HtWt = get_arg(args::cache_HtWt);
    std::uint32_t seq_tile = get_arg(args::seq_tile_start);
    std::uint32_t cache_page = get_arg(args::start_id);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const auto dst = TensorAccessor(tensor::dst);
    const std::uint32_t page_bytes = dfb_out.get_entry_size();

    for (std::uint32_t block = 0; block < num_blocks; ++block) {
        for (std::uint32_t tile = 0; tile < Wt; ++tile) {
            dfb_out.wait_front(1);
            noc.async_write(dfb_out, dst, page_bytes, {}, {.page_id = cache_page + tile});
            noc.async_writes_flushed();
            dfb_out.pop_front(1);
        }
        cache_page += Wt;
        if (++seq_tile == input_Ht) {
            seq_tile = 0;
            cache_page += cache_HtWt - input_Ht * Wt;
        }
    }
    noc.async_write_barrier();
}
