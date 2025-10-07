// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    uint32_t dst_addr1 = get_arg_val<uint32_t>(0);
    uint32_t dst_addr2 = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst1_args = TensorAccessorArgs<1>();
    constexpr auto dst2_args = TensorAccessorArgs<dst1_args.next_compile_time_args_offset()>();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s1 = TensorAccessor(dst1_args, dst_addr1, tile_bytes);
    const auto s2 = TensorAccessor(dst2_args, dst_addr2, tile_bytes);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onetile);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        if (dst_addr1 != 0) {
            noc_async_write_tile(i, s1, l1_read_addr);
            noc_async_write_barrier();
        }

        if (dst_addr2 != 0) {
            noc_async_write_tile(i, s2, l1_read_addr);
            noc_async_write_barrier();
        }

        cb_pop_front(cb_id_out, onetile);
    }
}
