// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto clip_coef_clamped_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_clip_coef_clamped = cb_id++;

    constexpr uint32_t onetile = 1;

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_addrg = TensorAccessor(input_args, input_addr, input_tile_bytes);

    // clip_coef_clamped
    const uint32_t clip_coef_clamped_tile_bytes = get_tile_size(cb_id_clip_coef_clamped);
    constexpr auto coef_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto coef_addrg = TensorAccessor(coef_args, clip_coef_clamped_addr, clip_coef_clamped_tile_bytes);

    // clip_coef_clamped
    const auto clip_coef_clamped_l1_write_ptr = get_write_ptr(cb_id_clip_coef_clamped);
    cb_reserve_back(cb_id_clip_coef_clamped, onetile);
    noc_async_read_tile(0, coef_addrg, clip_coef_clamped_l1_write_ptr);
    noc_async_read_barrier();
    cb_push_back(cb_id_clip_coef_clamped, onetile);

    // input
    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_input, onetile);
        noc_async_read_tile(tile_idx, input_addrg, input_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_input, onetile);
    }

}  // void kernel_main()
