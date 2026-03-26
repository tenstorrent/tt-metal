// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(1);

    // compile-time args
    constexpr uint32_t cb_id_in16 = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in16);
    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_dst.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end(); page_iter++) {
            cb_wait_front(cb_id_in16, 1);
            const uint64_t output_page_noc_addr = page_iter->noc_addr();
            uint32_t output_page_read_addr = get_read_ptr(cb_id_in16);

            noc_async_write(output_page_read_addr, output_page_noc_addr, tile_size_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_in16, 1);
        }
    }
}
