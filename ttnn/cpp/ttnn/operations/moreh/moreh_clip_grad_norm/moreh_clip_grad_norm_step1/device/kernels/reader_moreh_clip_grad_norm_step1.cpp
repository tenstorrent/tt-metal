// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

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

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr, input_tile_bytes);

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_decimal, decimal);
    fill_cb_with_value(cb_id_one, scaler.u);
    generate_mask_h_w_if_needed(cb_id_mask_h_w, origin_h, origin_w);

    constexpr uint32_t onetile = 1;

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_input, onetile);
        noc_async_read_tile(tile_idx, s, input_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_input, onetile);
    }

}  // void kernel_main()
