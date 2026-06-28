// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// CT args: [num_o_tiles, ...TensorAccessorArgs(output)]
// RT args: [output_addr, total_tiles]

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t total_tiles = get_arg_val<uint32_t>(1);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_out = tt::CBIndex::c_17;
    uint32_t tile_bytes = get_tile_size(cb_out);
    const auto accessor = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t i = 0; i < total_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
