// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

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

    const auto s0 = TensorAccessor(dst0_args, dst0_addr);

    const auto s1 = TensorAccessor(dst1_args, dst1_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out0(cb_id_out0);
    experimental::CircularBuffer cb_out1(cb_id_out1);
    const auto out0_tile_bytes = get_tile_size(cb_id_out0);
    const auto out1_tile_bytes = get_tile_size(cb_id_out1);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        if (has_input_grad) {
            cb_out0.wait_front(onetile);
            noc.async_write(cb_out0, s0, out0_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            cb_out0.pop_front(onetile);
        }

        if (has_other_grad) {
            cb_out1.wait_front(onetile);
            noc.async_write(cb_out1, s1, out1_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
            cb_out1.pop_front(onetile);
        }
    }
}
