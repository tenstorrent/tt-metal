// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Writes the three per-token-tile outputs (pre, post, comb) back to DRAM.
void kernel_main() {
    const uint32_t num_token_tiles = get_arg(args::num_token_tiles);
    const uint32_t start_tile = get_arg(args::start_tile);  // this core's first output page

    Noc noc;
    DataflowBuffer dfbp(dfb::pre), dfbq(dfb::post), dfbc(dfb::comb);

    const uint32_t pre_page = dfbp.get_entry_size();
    const uint32_t post_page = dfbq.get_entry_size();
    const uint32_t comb_page = dfbc.get_entry_size();

    const auto s_pre = TensorAccessor(tensor::pre);
    const auto s_post = TensorAccessor(tensor::post);
    const auto s_comb = TensorAccessor(tensor::comb);

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        dfbp.wait_front(1);
        noc.async_write(dfbp, s_pre, pre_page, {}, {.page_id = start_tile + t});
        noc.async_write_barrier();
        dfbp.pop_front(1);

        dfbq.wait_front(1);
        noc.async_write(dfbq, s_post, post_page, {}, {.page_id = start_tile + t});
        noc.async_write_barrier();
        dfbq.pop_front(1);

        dfbc.wait_front(1);
        noc.async_write(dfbc, s_comb, comb_page, {}, {.page_id = start_tile + t});
        noc.async_write_barrier();
        dfbc.pop_front(1);
    }
}
