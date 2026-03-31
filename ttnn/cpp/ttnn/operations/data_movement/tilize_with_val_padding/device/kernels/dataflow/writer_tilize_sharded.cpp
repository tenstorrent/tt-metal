// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    // Runtime args
    uint32_t rt_arg_ind = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t num_tiles = get_arg_val<uint32_t>(rt_arg_ind++);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, tile_bytes);

    // Write tiles from CB to sharded output
    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        uint64_t dst_noc_addr = s0.get_noc_addr(tile_id);
        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
        noc_async_writes_flushed();

        cb_pop_front(cb_id_out, 1);
    }

    noc_async_write_barrier();
}
