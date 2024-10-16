// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto output_grad_addr = get_arg_val<uint32_t>(i++);
    const bool output_grad_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto mean_addr = get_arg_val<uint32_t>(i++);
    const bool mean_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto rstd_addr = get_arg_val<uint32_t>(i++);
    const bool rstd_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const bool gamma_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool gamma_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto num_channels = get_arg_val<uint32_t>(i++);
    const auto num_groups = get_arg_val<uint32_t>(i++);

    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_output_grad = cb_id++;
    const auto cb_id_input = cb_id++;
    const auto cb_id_mean = cb_id++;
    const auto cb_id_rstd = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_n_recip_n = cb_id++;
    const auto cb_id_gamma = cb_id++;
    const auto cb_id_mask_h_w = cb_id++;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_H : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_W : TILE_W;

    const auto Ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto Wt = (origin_w + TILE_W - 1) / TILE_W;

    const auto C = num_channels;

    union {
        float f;
        uint32_t u;
    } scalar;
    scalar.f = 1.0f;
    fill_cb_with_value(cb_id_one, scalar.u);

    const auto n = static_cast<float>((num_channels / num_groups) * origin_h * origin_w);
    scalar.f = n;
    fill_cb_with_value(cb_id_n_recip_n, scalar.u);
    scalar.f = 1.0f / n;
    fill_cb_with_value(cb_id_n_recip_n, scalar.u);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w, get_tile_size(cb_id_mask_h_w));
    }

    // output_grad
    const uint32_t output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto output_grad_data_format = get_dataformat(cb_id_output_grad);

    const InterleavedAddrGenFast<true> dram_output_grad_addrg = {.bank_base_address = output_grad_addr,
                                                                 .page_size = output_grad_tile_bytes,
                                                                 .data_format = output_grad_data_format};

    const InterleavedAddrGenFast<false> l1_output_grad_addrg = {.bank_base_address = output_grad_addr,
                                                                .page_size = output_grad_tile_bytes,
                                                                .data_format = output_grad_data_format};

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

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

    // gamma
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamma_data_format = get_dataformat(cb_id_gamma);

    const InterleavedAddrGenFast<true> dram_gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    const InterleavedAddrGenFast<false> l1_gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    const auto mean_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
    const auto rstd_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
    const auto start_mean_rstd_idx = tile_offset / num_inner_tiles;

    const auto output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    const auto mean_l1_write_ptr = get_write_ptr(cb_id_mean);
    const auto rstd_l1_write_ptr = get_write_ptr(cb_id_rstd);
    const auto gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);

    uint32_t mean_rstd_idx, mean_rstd_n_idx, mean_rstd_g_idx;
    uint32_t mean_rstd_tile_h_idx, mean_rstd_tile_w_idx;
    uint32_t mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile;
    uint32_t mean_rstd_Wt, mean_rstd_tile_idx, tilized_mean_rstd_idx_in_tile;

    uint32_t input_tile_idx;
    uint32_t output_grad_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        // mean, rstd (1, 1, N, num_groups)
        // mean_rstd_idx = n * num_groups + g
        mean_rstd_idx = start_mean_rstd_idx + outer_idx;
        mean_rstd_n_idx = mean_rstd_idx / num_groups;
        mean_rstd_g_idx = mean_rstd_idx % num_groups;

        mean_rstd_tile_h_idx = mean_rstd_n_idx / TILE_H;
        mean_rstd_tile_w_idx = mean_rstd_g_idx / TILE_W;

        mean_rstd_h_idx_in_tile = mean_rstd_n_idx % TILE_H;
        mean_rstd_w_idx_in_tile = mean_rstd_g_idx % TILE_W;

        mean_rstd_Wt = (num_groups + TILE_W - 1) / TILE_W;

        mean_rstd_tile_idx = mean_rstd_tile_h_idx * mean_rstd_Wt + mean_rstd_tile_w_idx;

        tilized_mean_rstd_idx_in_tile =
            get_tilized_idx(mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile, TILE_H, TILE_W);

        // mean (1, 1, N, num_groups)
        cb_reserve_back(cb_id_mean, onetile);
        if (mean_is_dram) {
            noc_async_read_tile(mean_rstd_tile_idx, dram_mean_addrg, mean_l1_write_ptr);
        } else {
            noc_async_read_tile(mean_rstd_tile_idx, l1_mean_addrg, mean_l1_write_ptr);
        }
        noc_async_read_barrier();
        if (tilized_mean_rstd_idx_in_tile != 0) {
            auto mean_ptr = reinterpret_cast<uint16_t *>(mean_l1_write_ptr);
            mean_ptr[0] = mean_ptr[tilized_mean_rstd_idx_in_tile];
        }
        cb_push_back(cb_id_mean, onetile);

        // rstd (1, 1, N, num_groups)
        cb_reserve_back(cb_id_rstd, onetile);
        if (rstd_is_dram) {
            noc_async_read_tile(mean_rstd_tile_idx, dram_rstd_addrg, rstd_l1_write_ptr);
        } else {
            noc_async_read_tile(mean_rstd_tile_idx, l1_rstd_addrg, rstd_l1_write_ptr);
        }
        noc_async_read_barrier();
        if (tilized_mean_rstd_idx_in_tile != 0) {
            auto rstd_ptr = reinterpret_cast<uint16_t *>(rstd_l1_write_ptr);
            rstd_ptr[0] = rstd_ptr[tilized_mean_rstd_idx_in_tile];
        }
        cb_push_back(cb_id_rstd, onetile);

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // input (N, C, H, W)
            input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            cb_reserve_back(cb_id_input, onetile);
            if (input_is_dram) {
                noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr);
            } else {
                noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);

            // output_grad (N, C, H, W)
            output_grad_tile_idx = input_tile_idx;
            cb_reserve_back(cb_id_output_grad, onetile);
            if (output_grad_is_dram) {
                noc_async_read_tile(output_grad_tile_idx, dram_output_grad_addrg, output_grad_l1_write_ptr);
            } else {
                noc_async_read_tile(output_grad_tile_idx, l1_output_grad_addrg, output_grad_l1_write_ptr);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, C)
                const auto gamma_c_idx = (input_tile_idx / (Ht * Wt)) % C;
                const auto gamma_tile_idx = gamma_c_idx / TILE_W;
                const auto gamma_w_idx_in_tile = gamma_c_idx % TILE_W;
                const auto tilized_gamma_idx_in_tile = get_tilized_idx(0, gamma_w_idx_in_tile, TILE_H, TILE_W);
                cb_reserve_back(cb_id_gamma, onetile);
                if (gamma_is_dram) {
                    noc_async_read_tile(gamma_tile_idx, dram_gamma_addrg, gamma_l1_write_ptr);
                } else {
                    noc_async_read_tile(gamma_tile_idx, l1_gamma_addrg, gamma_l1_write_ptr);
                }
                noc_async_read_barrier();
                if (tilized_gamma_idx_in_tile != 0) {
                    auto gamma_ptr = reinterpret_cast<uint16_t *>(gamma_l1_write_ptr);
                    gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
                }
                cb_push_back(cb_id_gamma, onetile);
            }
        }  // inner_idx loop

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // output_grad (N, C, H, W)
            output_grad_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            cb_reserve_back(cb_id_output_grad, onetile);
            if (output_grad_is_dram) {
                noc_async_read_tile(output_grad_tile_idx, dram_output_grad_addrg, output_grad_l1_write_ptr);
            } else {
                noc_async_read_tile(output_grad_tile_idx, l1_output_grad_addrg, output_grad_l1_write_ptr);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, C)
                const auto gamma_c_idx = (output_grad_tile_idx / (Ht * Wt)) % C;
                const auto gamma_tile_idx = gamma_c_idx / TILE_W;
                const auto gamma_w_idx_in_tile = gamma_c_idx % TILE_W;
                const auto tilized_gamma_idx_in_tile = get_tilized_idx(0, gamma_w_idx_in_tile, TILE_H, TILE_W);
                cb_reserve_back(cb_id_gamma, onetile);
                if (gamma_is_dram) {
                    noc_async_read_tile(gamma_tile_idx, dram_gamma_addrg, gamma_l1_write_ptr);
                } else {
                    noc_async_read_tile(gamma_tile_idx, l1_gamma_addrg, gamma_l1_write_ptr);
                }
                noc_async_read_barrier();
                if (tilized_gamma_idx_in_tile != 0) {
                    auto gamma_ptr = reinterpret_cast<uint16_t *>(gamma_l1_write_ptr);
                    gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
                }
                cb_push_back(cb_id_gamma, onetile);
            }

            // input (N, C, H, W)
            input_tile_idx = output_grad_tile_idx;
            cb_reserve_back(cb_id_input, onetile);
            if (input_is_dram) {
                noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr);
            } else {
                noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
