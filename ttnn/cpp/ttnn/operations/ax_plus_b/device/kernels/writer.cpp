// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "hostdevcommon/kernel_structs.h"

//
// Writer for elemwise y = ax + b
// Assumptions
// - y is stored in cb_16
//
void kernel_main() {
    uint32_t y_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_y = tt::CBIndex::c_16;
    constexpr auto y_args = TensorAccessorArgs<1>();

    auto y_accessor = TensorAccessor(y_args, y_addr, get_tile_size(cb_id_y));

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_y, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_y);
        noc_async_write_page(i, y_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_y, 1);
    }
}
