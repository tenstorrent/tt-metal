// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t num_output_tiles_per_head = get_arg_val<uint32_t>(2);
    const uint32_t cache_batch_start = get_arg_val<uint32_t>(3);
    const uint32_t cache_HtWt = get_arg_val<uint32_t>(4);
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);
    const uint32_t block_start = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto cache_ta_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto cache_s = TensorAccessor(cache_ta_args, cache_addr, tile_bytes);

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t global_block = block_start + b;
        const uint32_t head = global_block / num_output_tiles_per_head;
        const uint32_t t = global_block % num_output_tiles_per_head;

        const uint32_t cache_tile_id = cache_batch_start + head * cache_HtWt + t * Wt;

        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        for (uint32_t w = 0; w < Wt; w++) {
            noc_async_write_tile(cache_tile_id + w, cache_s, l1_read_addr + w * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
    }
}
