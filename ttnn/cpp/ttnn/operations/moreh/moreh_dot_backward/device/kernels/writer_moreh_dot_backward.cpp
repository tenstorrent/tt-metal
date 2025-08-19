// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t dst0_addr = get_arg_val<uint32_t>(2);
    uint32_t dst1_addr = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out1 = get_compile_time_arg_val(1);
    constexpr auto dst0_args = TensorAccessorArgs<2>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t dst0_tile_bytes = get_tile_size(cb_id_out0);

    const uint32_t dst1_tile_bytes = get_tile_size(cb_id_out1);

    const auto s0 = TensorAccessor(dst0_args, dst0_addr, dst0_tile_bytes);

    const auto s1 = TensorAccessor(dst1_args, dst1_addr, dst1_tile_bytes);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        if (has_input_grad) {
            cb_wait_front(cb_id_out0, onetile);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            noc_async_write_tile(i, s0, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, onetile);
        }

        if (has_other_grad) {
            cb_wait_front(cb_id_out1, onetile);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out1);
            noc_async_write_tile(i, s1, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out1, onetile);
        }
    }
}
