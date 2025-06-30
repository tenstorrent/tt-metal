// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    auto tensor_args = make_tensor_accessor_args<1, 0>();

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    auto tensor_accessor = make_tensor_accessor_from_args(tensor_args, dst_addr, tile_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t curr_noc_addr = tensor_accessor.get_noc_addr(i);
        noc_async_write(l1_read_addr, curr_noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
#endif
}
