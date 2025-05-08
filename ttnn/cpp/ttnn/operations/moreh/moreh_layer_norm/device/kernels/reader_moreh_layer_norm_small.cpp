// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const auto beta_addr = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto scaler = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);
    const auto mask_h = get_arg_val<uint32_t>(i++);
    const auto mask_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_eps = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_mask_h = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_mask_w = tt::CBIndex::c_6;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t block_size = get_compile_time_arg_val(3);

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

#ifdef GAMMA_HAS_VALUE
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamma_data_format = get_dataformat(cb_id_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> gamm_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};
#endif

#ifdef BETA_HAS_VALUE
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto beta_data_format = get_dataformat(cb_id_beta);
    const InterleavedAddrGenFast<beta_is_dram> beta_addrg = {
        .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};
#endif

    fill_cb_with_value(cb_id_scaler, scaler);
    fill_cb_with_value(cb_id_eps, eps);

#ifdef DO_MASK_H
    generate_mask_h(cb_id_mask_h, mask_h);
#endif

#ifdef DO_MASK_W
    generate_mask_w(cb_id_mask_w, mask_w);
#endif

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    uint32_t input_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        cb_reserve_back(cb_id_input, num_inner);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            input_tile_idx = tile_offset + outer_idx * num_inner + inner_idx;
            noc_async_read_tile(input_tile_idx, input_addrg, input_l1_write_ptr + inner_idx * input_tile_bytes);
        }  // num_inner loop
        noc_async_read_barrier();
        cb_push_back(cb_id_input, num_inner);

        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
#ifdef GAMMA_HAS_VALUE
            cb_reserve_back(cb_id_gamma, block_size);
            auto gamma_l1_write_addr = get_write_ptr(cb_id_gamma);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_read_tile(inner_idx + r, gamm_addrg, gamma_l1_write_addr);
                gamma_l1_write_addr += gamma_tile_bytes;
            }  // block_size loop
            noc_async_read_barrier();
            cb_push_back(cb_id_gamma, block_size);
#endif

#ifdef BETA_HAS_VALUE
            cb_reserve_back(cb_id_beta, block_size);
            auto beta_l1_write_addr = get_write_ptr(cb_id_beta);
            for (uint32_t r = 0; r < block_size; r++) {
                noc_async_read_tile(inner_idx + r, beta_addrg, beta_l1_write_addr);
                beta_l1_write_addr += beta_tile_bytes;
            }  // block_size loop
            noc_async_read_barrier();
            cb_push_back(cb_id_beta, block_size);
#endif
        }  // num_inner loop
        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
