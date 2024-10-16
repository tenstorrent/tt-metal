// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

template <typename T>
void read_mean_rstd(uint32_t cb_id,
                    uint32_t tile_offset,
                    uint32_t normalized_dims,
                    uint32_t outer_idx,
                    uint32_t height,
                    uint32_t width,
                    uint32_t Ht,
                    uint32_t Wt,
                    T addrg) {
    constexpr uint32_t onetile = 1;

    const uint32_t cb_tile_bytes = get_tile_size(cb_id);
    const auto cb_dtype_bytes = cb_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    cb_reserve_back(cb_id, onetile);

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_write_addr);
    if (normalized_dims == 1) {
        for (uint32_t src_h = 0; src_h < 2; src_h++) {
            auto tile_idx = tile_offset + outer_idx;

            auto wt = tile_idx % Wt;
            auto nh = tile_idx / Wt;
            auto h = nh % height;
            auto n = nh / height;

            auto w = src_h * FACE_HEIGHT;

            auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w);

            auto ht = h / TILE_HEIGHT;
            auto noc_id = n * Ht * Wt + ht * Wt + wt;

            auto src_idx = get_tilized_idx(0, src_h * FACE_WIDTH);

            auto dst_noc_addr = get_noc_addr(noc_id, addrg);

            noc_async_read(dst_noc_addr + tilized_idx * cb_dtype_bytes,
                           l1_write_addr + src_idx * cb_dtype_bytes,
                           cb_dtype_bytes * FACE_HEIGHT);

            noc_async_read_barrier();
        }

        // rotate data
        for (uint32_t i = 0; i < 16; i++) {
            l1_ptr[i * FACE_WIDTH] = l1_ptr[i];
            l1_ptr[i * FACE_WIDTH + 256 * 2] = l1_ptr[i + 256];
        }
    } else {
        auto idx = tile_offset + outer_idx;

        auto w = idx % width;
        auto nh = idx / width;
        auto h = nh % height;
        auto n = nh / height;

        auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w % TILE_WIDTH);

        auto wt = w / TILE_WIDTH;
        auto ht = h / TILE_HEIGHT;

        auto noc_id = n * Ht * Wt + ht * Wt + wt;

        auto dst_noc_addr = get_noc_addr(noc_id, addrg);
        noc_async_read(
            dst_noc_addr + tilized_idx * cb_dtype_bytes, l1_write_addr + tilized_idx * cb_dtype_bytes, cb_dtype_bytes);

        noc_async_read_barrier();
        if (idx != 0) {
            l1_ptr[0] = l1_ptr[tilized_idx];
        }
    }

    cb_push_back(cb_id, onetile);
}

void kernel_main() {
    const auto output_grad_addr = get_arg_val<uint32_t>(0);
    const auto input_addr = get_arg_val<uint32_t>(1);
    const auto mean_addr = get_arg_val<uint32_t>(2);
    const auto rstd_addr = get_arg_val<uint32_t>(3);
    const auto gamma_addr = get_arg_val<uint32_t>(4);

    const auto num_rows_per_core = get_arg_val<uint32_t>(5);
    const auto num_inner = get_arg_val<uint32_t>(6);
    const auto tile_offset = get_arg_val<uint32_t>(7);
    const auto n = get_arg_val<uint32_t>(8);
    const auto recip_n = get_arg_val<uint32_t>(9);
    const auto mask_h = get_arg_val<uint32_t>(10);
    const auto mask_w = get_arg_val<uint32_t>(11);
    const auto normalized_dims = get_arg_val<uint32_t>(12);
    const auto mean_rstd_height = get_arg_val<uint32_t>(13);
    const auto mean_rstd_width = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_id_output_grad = 0;
    constexpr uint32_t cb_id_input = 1;
    constexpr uint32_t cb_id_mean = 2;
    constexpr uint32_t cb_id_rstd = 3;
    constexpr uint32_t cb_id_scaler = 4;
    constexpr uint32_t cb_id_n_recip_n = 5;
    constexpr uint32_t cb_id_gamma = 6;
    constexpr uint32_t cb_id_mask_h_w = 7;

    const uint32_t output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto output_grad_data_format = get_dataformat(cb_id_output_grad);

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const uint32_t mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto mean_data_format = get_dataformat(cb_id_mean);

    const uint32_t rstd_tile_bytes = get_tile_size(cb_id_rstd);
    const auto rstd_data_format = get_dataformat(cb_id_rstd);

    constexpr bool output_grad_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool mean_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool rstd_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool do_mask_h = get_compile_time_arg_val(6) == 1;
    constexpr bool do_mask_w = get_compile_time_arg_val(7) == 1;

    const InterleavedAddrGenFast<output_grad_is_dram> output_grad_addrg = {.bank_base_address = output_grad_addr,
                                                                           .page_size = output_grad_tile_bytes,
                                                                           .data_format = output_grad_data_format};

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<mean_is_dram> mean_addrg = {
        .bank_base_address = mean_addr, .page_size = mean_tile_bytes, .data_format = mean_data_format};

    const InterleavedAddrGenFast<rstd_is_dram> rstd_addrg = {
        .bank_base_address = rstd_addr, .page_size = rstd_tile_bytes, .data_format = rstd_data_format};

    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamma_data_format = get_dataformat(cb_id_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> gamma_addrg = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_scaler, scaler.u);
    fill_cb_with_value(cb_id_n_recip_n, n);
    fill_cb_with_value(cb_id_n_recip_n, recip_n);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    auto mean_rstd_Ht = (mean_rstd_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    auto mean_rstd_Wt = (mean_rstd_width + TILE_WIDTH - 1) / TILE_WIDTH;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        uint32_t mean_rstd_tile_offset = tile_offset / num_inner;

        // mean
        read_mean_rstd(cb_id_mean,
                       mean_rstd_tile_offset,
                       normalized_dims,
                       outer_idx,
                       mean_rstd_height,
                       mean_rstd_width,
                       mean_rstd_Ht,
                       mean_rstd_Wt,
                       mean_addrg);

        // rstd
        read_mean_rstd(cb_id_rstd,
                       mean_rstd_tile_offset,
                       normalized_dims,
                       outer_idx,
                       mean_rstd_height,
                       mean_rstd_width,
                       mean_rstd_Ht,
                       mean_rstd_Wt,
                       rstd_addrg);

        // For Sum[dy] and Sum[y * dy]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            // input (N, C, H, W)
            const uint32_t input_l1_write_ptr = get_write_ptr(cb_id_input);
            cb_reserve_back(cb_id_input, onetile);
            noc_async_read_tile(offs + inner_idx + tile_offset, input_addrg, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);

            // output_grad (N, C, H, W)
            const uint32_t output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
            cb_reserve_back(cb_id_output_grad, onetile);
            noc_async_read_tile(offs + inner_idx + tile_offset, output_grad_addrg, output_grad_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, W)
                const uint32_t gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);
                cb_reserve_back(cb_id_gamma, onetile);
                noc_async_read_tile(inner_idx, gamma_addrg, gamma_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_gamma, onetile);
            }  // gamma_has_value
        }  // num_inner loop

        // For ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            // output_grad (N, C, H, W)
            const uint32_t output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
            cb_reserve_back(cb_id_output_grad, onetile);
            noc_async_read_tile(offs + inner_idx + tile_offset, output_grad_addrg, output_grad_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_output_grad, onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, W)
                const uint32_t gamma_l1_write_ptr = get_write_ptr(cb_id_gamma);
                cb_reserve_back(cb_id_gamma, onetile);
                noc_async_read_tile(inner_idx, gamma_addrg, gamma_l1_write_ptr);
                noc_async_read_barrier();
                cb_push_back(cb_id_gamma, onetile);
            }  // gamma_has_value

            // input (N, C, H, W)
            const uint32_t input_l1_write_ptr = get_write_ptr(cb_id_input);
            cb_reserve_back(cb_id_input, onetile);
            noc_async_read_tile(offs + inner_idx + tile_offset, input_addrg, input_l1_write_ptr);
            noc_async_read_barrier();
            cb_push_back(cb_id_input, onetile);
        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
