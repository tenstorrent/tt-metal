// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const uint32_t tile_size = get_tile_size(cb_id);
    const auto d0 = TensorAccessor(dst_args, dst_addr, tile_size);

    for (uint32_t tile_id = start_tile; tile_id < end_tile; tile_id++) {
        cb_wait_front(cb_id, 1);
        noc_async_write_tile(tile_id, d0, get_read_ptr(cb_id));
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
