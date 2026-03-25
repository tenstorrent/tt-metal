// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(2);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t num_shards = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);

    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto accessor_src = TensorAccessor(src_args, src_addr);
    const auto accessor_dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_dst.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end(); page_iter++) {
            auto output_page_id = page_iter->page_id();
            cb_wait_front(cb_id_in0, 1);
            const uint64_t src_page_noc_addr = accessor_src.get_noc_addr(output_page_id);
            uint32_t output_page_write_addr = get_write_ptr(cb_id_in0);

            noc_async_read(src_page_noc_addr, output_page_write_addr, tile_size_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, 1);
        }
    }
}
