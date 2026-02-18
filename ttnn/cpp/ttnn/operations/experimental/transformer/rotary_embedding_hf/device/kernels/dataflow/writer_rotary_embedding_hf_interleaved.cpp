// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t output_tile_bytes = get_tile_size(output_cb_id);
    const auto s = TensorAccessor(dst_args, dst_addr, output_tile_bytes);

    uint32_t output_curr_id = start_id;

#ifdef OUT_SHARDED
    cb_wait_front(output_cb_id, num_tiles);
#else
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(output_cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(output_cb_id);
        noc_async_write_tile(output_curr_id, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(output_cb_id, 1);
        output_curr_id++;
    }
#endif
}
