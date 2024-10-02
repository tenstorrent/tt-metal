// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto gamma_grad_addr = get_arg_val<uint32_t>(i++);
    const bool gamma_grad_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool gamma_grad_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto beta_grad_addr = get_arg_val<uint32_t>(i++);
    const bool beta_grad_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool beta_grad_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_channels_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto batch = get_arg_val<uint32_t>(i++);

    const auto HtWt = num_inner_tiles / batch;

    uint32_t cb_id{16};
    const auto cb_id_gamma_grad = cb_id++;
    const auto cb_id_beta_grad = cb_id++;

    // gamma_grad
    const uint32_t gamma_grad_tile_bytes = get_tile_size(cb_id_gamma_grad);
    const auto gamma_grad_data_format = get_dataformat(cb_id_gamma_grad);

    const InterleavedAddrGenFast<true> dram_gamma_grad_addrg = {
        .bank_base_address = gamma_grad_addr,
        .page_size = gamma_grad_tile_bytes,
        .data_format = gamma_grad_data_format};

    const InterleavedAddrGenFast<false> l1_gamma_grad_addrg = {
        .bank_base_address = gamma_grad_addr,
        .page_size = gamma_grad_tile_bytes,
        .data_format = gamma_grad_data_format};

    // beta_grad
    const uint32_t beta_grad_tile_bytes = get_tile_size(cb_id_beta_grad);
    const auto beta_grad_data_format = get_dataformat(cb_id_beta_grad);

    const InterleavedAddrGenFast<true> dram_beta_grad_addrg = {
        .bank_base_address = beta_grad_addr, .page_size = beta_grad_tile_bytes, .data_format = beta_grad_data_format};

    const InterleavedAddrGenFast<false> l1_beta_grad_addrg = {
        .bank_base_address = beta_grad_addr, .page_size = beta_grad_tile_bytes, .data_format = beta_grad_data_format};

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto gamma_grad_l1_read_ptr = get_read_ptr(cb_id_gamma_grad);
    const auto beta_grad_l1_read_ptr = get_read_ptr(cb_id_beta_grad);

    for (uint32_t outer_idx = 0; outer_idx < num_channels_per_core; ++outer_idx) {
        auto c_idx = outer_idx + (tile_offset / HtWt);

        // gamma_grad, beta_grad (1, 1, 1, C)
        const auto gamma_beta_c_idx = c_idx;
        const auto gamma_beta_tile_idx = gamma_beta_c_idx / TILE_W;
        const auto gamma_beta_w_idx_in_tile = gamma_beta_c_idx % TILE_W;
        const auto tilized_gamma_beta_idx_in_tile = get_tilized_idx(0, gamma_beta_w_idx_in_tile, TILE_H, TILE_W);

        if (gamma_grad_has_value) {
            // gamma_grad (1, 1, 1, C)
            const auto gamma_grad_dtype_bytes = gamma_grad_tile_bytes / (TILE_H * TILE_W);
            cb_wait_front(cb_id_gamma_grad, onetile);
            if (tilized_gamma_beta_idx_in_tile != 0) {
                auto gamma_grad_ptr = reinterpret_cast<uint16_t *>(gamma_grad_l1_read_ptr);
                gamma_grad_ptr[tilized_gamma_beta_idx_in_tile] = gamma_grad_ptr[0];
            }
            const auto gamma_grad_noc_addr = gamma_grad_is_dram
                                                 ? get_noc_addr(gamma_beta_tile_idx, dram_gamma_grad_addrg)
                                                 : get_noc_addr(gamma_beta_tile_idx, l1_gamma_grad_addrg);
            noc_async_write(
                gamma_grad_l1_read_ptr + tilized_gamma_beta_idx_in_tile * gamma_grad_dtype_bytes,
                gamma_grad_noc_addr + tilized_gamma_beta_idx_in_tile * gamma_grad_dtype_bytes,
                gamma_grad_dtype_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_gamma_grad, onetile);
        }

        if (beta_grad_has_value) {
            // beta_grad (1, 1, 1, C)
            const auto beta_grad_dtype_bytes = beta_grad_tile_bytes / (TILE_H * TILE_W);
            cb_wait_front(cb_id_beta_grad, onetile);
            if (tilized_gamma_beta_idx_in_tile != 0) {
                auto beta_grad_ptr = reinterpret_cast<uint16_t *>(beta_grad_l1_read_ptr);
                beta_grad_ptr[tilized_gamma_beta_idx_in_tile] = beta_grad_ptr[0];
            }
            const auto beta_grad_noc_addr = beta_grad_is_dram ? get_noc_addr(gamma_beta_tile_idx, dram_beta_grad_addrg)
                                                              : get_noc_addr(gamma_beta_tile_idx, l1_beta_grad_addrg);
            noc_async_write(
                beta_grad_l1_read_ptr + tilized_gamma_beta_idx_in_tile * beta_grad_dtype_bytes,
                beta_grad_noc_addr + tilized_gamma_beta_idx_in_tile * beta_grad_dtype_bytes,
                beta_grad_dtype_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_beta_grad, onetile);
        }

    }  // outer_idx loop

}  // void kernel_main()
