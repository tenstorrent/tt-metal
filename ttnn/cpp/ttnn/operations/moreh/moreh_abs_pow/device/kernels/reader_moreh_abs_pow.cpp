// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto decimal = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_decimal = cb_id++;
    const auto cb_id_mask_w = cb_id++;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr, input_tile_bytes);

    Scalar one;
    one.f = 1.0f;
    fill_cb_with_value(cb_id_one, one.u);
    fill_cb_with_value(cb_id_decimal, decimal);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        generate_mask_w(cb_id_mask_w, mask_w);
    }

    const auto start_tile_idx = tile_offset;
    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const auto tile_idx = start_tile_idx + row_idx * Wt + col_idx;
            cb_reserve_back(cb_id_input, 1);
            noc_async_read_tile(tile_idx, s, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, 1);
        }
    }

}  // void kernel_main()
