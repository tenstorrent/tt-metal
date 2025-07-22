// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto mean_addr = get_arg_val<uint32_t>(i++);
    const bool mean_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool mean_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto rstd_addr = get_arg_val<uint32_t>(i++);
    const bool rstd_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool rstd_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);

    const auto num_groups = get_arg_val<uint32_t>(i++);
    const auto block_size = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;
    const auto cb_id_mean = cb_id++;
    const auto cb_id_rstd = cb_id++;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // mean
    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_data_format = get_dataformat(cb_id_mean);

    const InterleavedAddrGenFast<true> dram_mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    const InterleavedAddrGenFast<false> l1_mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    // rstd
    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_data_format = get_dataformat(cb_id_rstd);

    const InterleavedAddrGenFast<true> dram_rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    const InterleavedAddrGenFast<false> l1_rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto start_mean_rstd_idx = tile_offset / num_inner_tiles;

    const auto output_l1_read_ptr = get_read_ptr(cb_id_output);
    uint32_t output_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        // mean, rstd (1, 1, N, num_groups)
        // mean_rstd_tile_idx = n * num_groups + g
        const auto mean_rstd_idx = start_mean_rstd_idx + outer_idx;
        const auto mean_rstd_n_idx = mean_rstd_idx / num_groups;
        const auto mean_rstd_g_idx = mean_rstd_idx % num_groups;

        const auto mean_rstd_tile_h_idx = mean_rstd_n_idx / TILE_H;
        const auto mean_rstd_tile_w_idx = mean_rstd_g_idx / TILE_W;

        const auto mean_rstd_h_idx_in_tile = mean_rstd_n_idx % TILE_H;
        const auto mean_rstd_w_idx_in_tile = mean_rstd_g_idx % TILE_W;

        const auto mean_rstd_Wt = (num_groups + TILE_W - 1) / TILE_W;

        const auto mean_rstd_tile_idx = mean_rstd_tile_h_idx * mean_rstd_Wt + mean_rstd_tile_w_idx;

        const auto tilized_mean_rstd_idx_in_tile =
            get_tilized_idx(mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile, TILE_H, TILE_W);

        // mean (1, 1, N, num_groups)
        if (mean_has_value) {
            const auto mean_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
            const auto mean_l1_read_ptr = get_read_ptr(cb_id_mean);
            cb_wait_front(cb_id_mean, onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                auto mean_ptr = reinterpret_cast<uint16_t*>(mean_l1_read_ptr);
                mean_ptr[tilized_mean_rstd_idx_in_tile] = mean_ptr[0];
            }
            const auto mean_noc_addr = mean_is_dram ? get_noc_addr(mean_rstd_tile_idx, dram_mean_addrg)
                                                    : get_noc_addr(mean_rstd_tile_idx, l1_mean_addrg);
            noc_async_write(
                mean_l1_read_ptr + tilized_mean_rstd_idx_in_tile * mean_dtype_bytes,
                mean_noc_addr + tilized_mean_rstd_idx_in_tile * mean_dtype_bytes,
                mean_dtype_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_mean, onetile);
        }

        // rstd (1, 1, N, num_groups)
        if (rstd_has_value) {
            const auto rstd_dtype_bytes = rstd_tile_bytes / (TILE_H * TILE_W);
            const auto rstd_l1_read_ptr = get_read_ptr(cb_id_rstd);
            cb_wait_front(cb_id_rstd, onetile);
            if (tilized_mean_rstd_idx_in_tile != 0) {
                auto rstd_ptr = reinterpret_cast<uint16_t*>(rstd_l1_read_ptr);
                rstd_ptr[tilized_mean_rstd_idx_in_tile] = rstd_ptr[0];
            }
            const auto rstd_noc_addr = rstd_is_dram ? get_noc_addr(mean_rstd_tile_idx, dram_rstd_addrg)
                                                    : get_noc_addr(mean_rstd_tile_idx, l1_rstd_addrg);
            noc_async_write(
                rstd_l1_read_ptr + tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes,
                rstd_noc_addr + tilized_mean_rstd_idx_in_tile * rstd_dtype_bytes,
                rstd_dtype_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_id_rstd, onetile);
        }

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            // output (N, C, H, W)
            cb_wait_front(cb_id_output, block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                output_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                if (output_is_dram) {
                    noc_async_write_tile(
                        output_tile_idx, dram_output_addrg, output_l1_read_ptr + r * output_tile_bytes);
                } else {
                    noc_async_write_tile(output_tile_idx, l1_output_addrg, output_l1_read_ptr + r * output_tile_bytes);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_output, block_size);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
