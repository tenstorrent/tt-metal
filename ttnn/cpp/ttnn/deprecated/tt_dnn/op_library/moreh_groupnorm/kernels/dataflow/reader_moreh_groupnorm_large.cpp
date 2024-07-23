// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const bool gamma_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool gamma_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto beta_addr = get_arg_val<uint32_t>(i++);
    const bool beta_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const bool beta_has_value = get_arg_val<uint32_t>(i++) == 1;

    const auto scaler = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto num_channels = get_arg_val<uint32_t>(i++);

    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);
    const auto block_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto Ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto Wt = (origin_w + TILE_W - 1) / TILE_W;

    const auto HtWt = Ht * Wt;

    const auto C = num_channels;

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_scaler = cb_id++;
    const auto cb_id_eps = cb_id++;
    const auto cb_id_gamma = cb_id++;
    const auto cb_id_beta = cb_id++;
    const auto cb_id_mask_h = cb_id++;
    const auto cb_id_mask_w = cb_id++;

    fill_cb_with_value(cb_id_scaler, scaler);
    fill_cb_with_value(cb_id_eps, eps);

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_H : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_W : TILE_W;

    if (do_mask_h) {
        generate_mask_h(cb_id_mask_h, mask_h);
    }
    if (do_mask_w) {
        generate_mask_w(cb_id_mask_w, mask_w);
    }

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // gamma
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamma_data_format = get_dataformat(cb_id_gamma);

    const InterleavedAddrGenFast<true> dram_gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    const InterleavedAddrGenFast<false> l1_gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    // beta
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto beta_data_format = get_dataformat(cb_id_beta);

    const InterleavedAddrGenFast<true> dram_beta_addrg = {
        .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};

    const InterleavedAddrGenFast<false> l1_beta_addrg = {
        .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
    uint32_t input_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        // For E[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            cb_reserve_back(cb_id_input, block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                if (input_is_dram) {
                    noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                } else {
                    noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, block_size);
        }  // inner_idx loop

        // For Var[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            cb_reserve_back(cb_id_input, block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                if (input_is_dram) {
                    noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                } else {
                    noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, block_size);
        }  // inner_idx loop

        // For (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
            cb_reserve_back(cb_id_input, block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                if (input_is_dram) {
                    noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                } else {
                    noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr + r * input_tile_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, block_size);

            // input (N, C, H, W)
            // input_tile_idx = n * C * Ht * Wt + c * Ht * Wt + h * Wt + w
            // n * C + c = input_tile_idx / (Ht * Wt)
            // c = (input_tile_idx / (Ht * Wt)) % C
            // gamma (1, 1, 1, C)
            if (gamma_has_value) {
                uint32_t gamma_tile_idx;
                const auto gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);
                cb_reserve_back(cb_id_gamma, block_size);
                for (uint32_t r = 0; r < block_size; r++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                    gamma_tile_idx = get_gamma_beta_tile_idx(input_tile_idx, HtWt, C, TILE_W);
                    if (gamma_is_dram) {
                        noc_async_read_tile(
                            gamma_tile_idx, dram_gamma_addrg, gamma_l1_write_ptr + r * gamma_tile_bytes);
                    } else {
                        noc_async_read_tile(gamma_tile_idx, l1_gamma_addrg, gamma_l1_write_ptr + r * gamma_tile_bytes);
                    }
                }
                noc_async_read_barrier();

                uint32_t tilized_gamma_idx_in_tile;
                for (uint32_t q = 0; q < block_size; q++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + q;
                    tilized_gamma_idx_in_tile =
                        get_tilized_gamma_beta_idx_in_tile(input_tile_idx, HtWt, C, TILE_H, TILE_W);
                    if (tilized_gamma_idx_in_tile != 0) {
                        auto gamma_ptr = reinterpret_cast<uint16_t *>(gamma_l1_write_ptr + q * gamma_tile_bytes);
                        gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
                    }
                }
                cb_push_back(cb_id_gamma, block_size);
            }

            // beta (1, 1, 1, C)
            if (beta_has_value) {
                uint32_t beta_tile_idx;
                const auto beta_l1_write_ptr = get_write_ptr(cb_id_beta);
                cb_reserve_back(cb_id_beta, block_size);
                for (uint32_t r = 0; r < block_size; r++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                    beta_tile_idx = get_gamma_beta_tile_idx(input_tile_idx, HtWt, C, TILE_W);
                    if (beta_is_dram) {
                        noc_async_read_tile(beta_tile_idx, dram_beta_addrg, beta_l1_write_ptr + r * beta_tile_bytes);
                    } else {
                        noc_async_read_tile(beta_tile_idx, l1_beta_addrg, beta_l1_write_ptr + r * beta_tile_bytes);
                    }
                }
                noc_async_read_barrier();

                uint32_t tilized_beta_idx_in_tile;
                for (uint32_t q = 0; q < block_size; q++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + q;
                    tilized_beta_idx_in_tile =
                        get_tilized_gamma_beta_idx_in_tile(input_tile_idx, HtWt, C, TILE_H, TILE_W);
                    if (tilized_beta_idx_in_tile != 0) {
                        auto beta_ptr = reinterpret_cast<uint16_t *>(beta_l1_write_ptr + q * beta_tile_bytes);
                        beta_ptr[0] = beta_ptr[tilized_beta_idx_in_tile];
                    }
                }
                cb_push_back(cb_id_beta, block_size);
            }
        }  // inner_idx loop
    }      // outer_idx loop

}  // void kernel_main()
