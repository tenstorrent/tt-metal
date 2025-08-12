// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_mask_h = cb_id++;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr, input_tile_bytes);

    Scalar one;
    one.f = 1.0f;
    fill_cb_with_value(cb_id_one, one.u);

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        generate_mask_h(cb_id_mask_h, mask_h);
    }

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);

    auto start_output_tile_idx = tile_offset;
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        const auto inner_idx = start_output_tile_idx % Wt;
        const auto outer_idx = start_output_tile_idx / Wt;

        auto input_tile_idx = outer_idx * Ht * Wt + inner_idx;
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            cb_reserve_back(cb_id_input, 1);
            noc_async_read_tile(input_tile_idx, s, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, 1);
            input_tile_idx += Wt;
        }

        start_output_tile_idx++;
    }

}  // void kernel_main()
