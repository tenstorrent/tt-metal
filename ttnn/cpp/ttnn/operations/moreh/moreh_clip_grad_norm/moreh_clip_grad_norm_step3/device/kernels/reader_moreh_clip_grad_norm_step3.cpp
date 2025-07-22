// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto clip_coef_clamped_addr = get_arg_val<uint32_t>(i++);
    const bool clip_coef_clamped_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_clip_coef_clamped = cb_id++;

    constexpr uint32_t onetile = 1;

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // clip_coef_clamped
    const uint32_t clip_coef_clamped_tile_bytes = get_tile_size(cb_id_clip_coef_clamped);
    const auto clip_coef_clamped_data_format = get_dataformat(cb_id_clip_coef_clamped);

    const InterleavedAddrGenFast<true> dram_clip_coef_clamped_addrg = {
        .bank_base_address = clip_coef_clamped_addr,
        .page_size = clip_coef_clamped_tile_bytes,
        .data_format = clip_coef_clamped_data_format};
    const InterleavedAddrGenFast<false> l1_clip_coef_clamped_addrg = {
        .bank_base_address = clip_coef_clamped_addr,
        .page_size = clip_coef_clamped_tile_bytes,
        .data_format = clip_coef_clamped_data_format};

    // clip_coef_clamped
    const auto clip_coef_clamped_l1_write_ptr = get_write_ptr(cb_id_clip_coef_clamped);
    cb_reserve_back(cb_id_clip_coef_clamped, onetile);
    if (clip_coef_clamped_is_dram) {
        noc_async_read_tile(0, dram_clip_coef_clamped_addrg, clip_coef_clamped_l1_write_ptr);
    } else {
        noc_async_read_tile(0, l1_clip_coef_clamped_addrg, clip_coef_clamped_l1_write_ptr);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_clip_coef_clamped, onetile);

    // input
    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_reserve_back(cb_id_input, onetile);
        if (input_is_dram) {
            noc_async_read_tile(tile_idx, dram_input_addrg, input_l1_write_ptr);
        } else {
            noc_async_read_tile(tile_idx, l1_input_addrg, input_l1_write_ptr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_input, onetile);
    }

}  // void kernel_main()
