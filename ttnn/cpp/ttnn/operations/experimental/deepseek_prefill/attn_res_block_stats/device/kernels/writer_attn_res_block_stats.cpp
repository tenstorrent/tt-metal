// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

void kernel_main() {
    constexpr uint32_t kSplitArgs = 0;
    constexpr auto out_args = TensorAccessorArgs<kSplitArgs + 3>();

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer cb_out_obj(cb_out);
    auto out_accessor = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    // The output is [1, C, N, 32], one tile per item, so an item's page is the item index.
    for_each_attn_res_block_stats_item<kSplitArgs>([&](uint32_t item) {
        cb_out_obj.wait_front(1);
        noc.async_write(cb_out_obj, out_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = item});
        noc.async_write_barrier();
        cb_out_obj.pop_front(1);
    });
}
