// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_shard_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(1);
    const uint32_t tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr auto src_args = TensorAccessorArgs<2>();
    const auto accessor_src = TensorAccessor(src_args, src_addr);
    auto shard_pages = accessor_src.shard_pages(start_shard_id);

    CircularBuffer cb_in(cb_id_in0);
    auto page_iter = shard_pages.begin();
    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_in.reserve_back(tiles_per_block);
        uint32_t cb_write_addr = cb_in.get_write_ptr();
        noc_async_read(page_iter->noc_addr(), cb_write_addr, tile_size_bytes * tiles_per_block);
        page_iter += tiles_per_block;
        noc_async_read_barrier();
        cb_in.push_back(tiles_per_block);
    }
}
