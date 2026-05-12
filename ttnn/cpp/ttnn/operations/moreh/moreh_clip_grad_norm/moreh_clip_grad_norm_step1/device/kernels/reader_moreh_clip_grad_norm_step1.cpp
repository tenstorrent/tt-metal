// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto decimal = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_decimal = cb_id++;
    const auto cb_id_mask_h_w = cb_id++;

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr);

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_decimal, decimal);
    fill_cb_with_value(cb_id_one, scaler.u);
    generate_mask_h_w_if_needed(cb_id_mask_h_w, origin_h, origin_w);

    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_input(cb_id_input);
    const auto input_tile_bytes = get_tile_size(cb_id_input);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_input.reserve_back(onetile);
        noc.async_read(s, cb_input, input_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_input.push_back(onetile);
    }

}  // void kernel_main()
