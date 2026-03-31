// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    uint32_t rt = 0;
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t num_tiles_core = get_arg_val<uint32_t>(rt++);
    const uint32_t shard_start_tile = get_arg_val<uint32_t>(rt++);
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(dst_args, dst_base_addr, tile_bytes);

    // Each core writes its own global tile range [shard_start_tile ... shard_start_tile + num_tiles_core]
    for (uint32_t t = 0; t < num_tiles_core; ++t) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read = get_read_ptr(cb_id_out);

        const uint32_t global_tile = shard_start_tile + t;
        const uint64_t dst_noc_addr = s0.get_noc_addr(global_tile);

        noc_async_write(l1_read, dst_noc_addr, tile_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_id_out, 1);
    }

    noc_async_write_barrier();
}
