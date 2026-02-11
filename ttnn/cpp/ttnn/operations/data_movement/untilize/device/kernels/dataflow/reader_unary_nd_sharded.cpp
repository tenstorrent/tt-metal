// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(1);

    // compile-time args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_input_block = get_compile_time_arg_val(1);
    constexpr uint32_t num_shards = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_args = TensorAccessorArgs<4>();
    const auto accessor_src = TensorAccessor(src_args, src_addr, tile_size_bytes);
    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = accessor_src.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end();
             page_iter += num_tiles_per_input_block) {
            cb_reserve_back(cb_id_in0, num_tiles_per_input_block);
            uint64_t noc_read_addr = page_iter->noc_addr();
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read(noc_read_addr, l1_write_addr, tile_size_bytes * num_tiles_per_input_block);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, num_tiles_per_input_block);
        }
    }
}
