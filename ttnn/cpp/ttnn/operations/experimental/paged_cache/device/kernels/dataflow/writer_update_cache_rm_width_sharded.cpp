// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    const auto cache_addr = get_arg(args::cache_addr);
    const auto my_batch_idx = get_arg(args::my_batch_idx);
    uint32_t update_idx = get_arg(args::update_idx);

    constexpr auto num_cache_cores = get_arg(args::num_cache_cores);
    constexpr auto shard_width_bytes = get_arg(args::shard_width_bytes);
    constexpr auto cache_num_rows = get_arg(args::cache_num_rows);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);

    DataflowBuffer dfb_input(dfb::input);
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif

    dfb_input.reserve_back(1);
    dfb_input.push_back(1);
    const uint32_t src_base = dfb_input.get_read_ptr();

    bool skip_update = false;
#ifdef USE_INDEX_TENSOR
    {
        const auto addrg = TensorAccessor(tensor::index);
        dfb_index.reserve_back(1);
        uint32_t index_cb_wr_ptr = dfb_index.get_write_ptr();
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
        noc.async_read_barrier();
        dfb_index.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        update_idx = index_ptr[my_batch_idx];
        dfb_index.pop_front(1);
        if (update_idx == (uint32_t)-1) {
            skip_update = true;
        }
    }
#endif

    if (!skip_update && update_idx < cache_num_rows) {
        const uint32_t row_offset = update_idx * shard_width_bytes;
        const uint32_t dst = cache_addr + row_offset;
        UnicastEndpoint cache_ep;
        for (uint32_t c = 0; c < num_cache_cores; ++c) {
            const uint32_t noc_x = get_common_vararg(2 * c);
            const uint32_t noc_y = get_common_vararg(2 * c + 1);
            noc.async_write(
                CoreLocalMem<uint32_t>(src_base + c * shard_width_bytes),
                cache_ep,
                shard_width_bytes,
                {},
                {.noc_x = noc_x, .noc_y = noc_y, .addr = dst});
        }
        noc.async_write_barrier();
    }

    dfb_input.pop_front(1);
}
