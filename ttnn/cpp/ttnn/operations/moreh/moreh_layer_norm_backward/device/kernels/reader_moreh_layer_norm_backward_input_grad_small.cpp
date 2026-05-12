// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

template <typename T>
void read_mean_rstd(
    const experimental::Noc& noc,
    uint32_t cb_id,
    uint32_t tile_offset,
    uint32_t normalized_dims,
    uint32_t outer_idx,
    uint32_t height,
    uint32_t width,
    uint32_t Ht,
    uint32_t Wt,
    T addrg) {
    using namespace tt::constants;
    constexpr uint32_t onetile = 1;

    experimental::CircularBuffer cb(cb_id);
    const uint32_t cb_tile_bytes = get_tile_size(cb_id);
    const auto cb_dtype_bytes = cb_tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    cb.reserve_back(onetile);

    uint32_t l1_write_addr = cb.get_write_ptr();
    experimental::CoreLocalMem<volatile uint16_t> l1_ptr(l1_write_addr);
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

            noc.async_read(
                addrg,
                cb,
                cb_dtype_bytes * FACE_HEIGHT,
                {.page_id = noc_id, .offset_bytes = tilized_idx * cb_dtype_bytes},
                {.offset_bytes = src_idx * cb_dtype_bytes});

            noc.async_read_barrier();
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

        noc.async_read(
            addrg,
            cb,
            cb_dtype_bytes,
            {.page_id = noc_id, .offset_bytes = tilized_idx * cb_dtype_bytes},
            {.offset_bytes = tilized_idx * cb_dtype_bytes});

        noc.async_read_barrier();
        if (idx != 0) {
            l1_ptr[0] = l1_ptr[tilized_idx];
        }
    }

    cb.push_back(onetile);
}

void kernel_main() {
    using namespace tt::constants;
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

    constexpr bool gamma_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool do_mask_h = get_compile_time_arg_val(1) == 1;
    constexpr bool do_mask_w = get_compile_time_arg_val(2) == 1;
    constexpr auto output_grad_args = TensorAccessorArgs<3>();
    constexpr auto input_args = TensorAccessorArgs<output_grad_args.next_compile_time_args_offset()>();
    constexpr auto mean_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<rstd_args.next_compile_time_args_offset()>();

    const auto output_grad_addrg = TensorAccessor(output_grad_args, output_grad_addr);
    const auto input_addrg = TensorAccessor(input_args, input_addr);
    const auto mean_addrg = TensorAccessor(mean_args, mean_addr);
    const auto rstd_addrg = TensorAccessor(rstd_args, rstd_addr);

    const auto gamma_addrg = TensorAccessor(gamma_args, gamma_addr);

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

    experimental::Noc noc;
    experimental::CircularBuffer cb_output_grad(cb_id_output_grad);
    experimental::CircularBuffer cb_input(cb_id_input);
    experimental::CircularBuffer cb_gamma(cb_id_gamma);
    const auto output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto input_tile_bytes = get_tile_size(cb_id_input);
    const auto gamma_tile_bytes = get_tile_size(cb_id_gamma);

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        uint32_t mean_rstd_tile_offset = tile_offset / num_inner;

        // mean
        read_mean_rstd(
            noc,
            cb_id_mean,
            mean_rstd_tile_offset,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            mean_rstd_Ht,
            mean_rstd_Wt,
            mean_addrg);

        // rstd
        read_mean_rstd(
            noc,
            cb_id_rstd,
            mean_rstd_tile_offset,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            mean_rstd_Ht,
            mean_rstd_Wt,
            rstd_addrg);

        // input (N, C, H, W)
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            cb_input.reserve_back(onetile);
            noc.async_read(
                input_addrg,
                cb_input,
                input_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_input.push_back(onetile);
        }  // inner_idx loop

        // output_grad (N, C, H, W)
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            cb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                cb_output_grad,
                output_grad_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_output_grad.push_back(onetile);

            if (gamma_has_value) {
                // gamma (1, 1, 1, W)
                cb_gamma.reserve_back(onetile);
                noc.async_read(gamma_addrg, cb_gamma, gamma_tile_bytes, {.page_id = inner_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_gamma.push_back(onetile);
            }  // gamma_has_value

        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
