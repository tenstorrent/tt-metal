// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

void kernel_main() {
    int i{0};
    const auto output_grad_addr = get_arg_val<uint32_t>(i++);
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto mean_addr = get_arg_val<uint32_t>(i++);
    const auto rstd_addr = get_arg_val<uint32_t>(i++);

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_channels_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto num_channels = get_arg_val<uint32_t>(i++);
    const auto num_groups = get_arg_val<uint32_t>(i++);

    const auto origin_h = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(0) == 1;
    constexpr auto output_grad_args = TensorAccessorArgs<1>();
    constexpr auto input_args = TensorAccessorArgs<output_grad_args.next_compile_time_args_offset()>();
    constexpr auto mean_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();

    uint32_t cb_id{0};
    const auto cb_id_output_grad = cb_id++;
    const auto cb_id_input = cb_id++;
    const auto cb_id_mean = cb_id++;
    const auto cb_id_rstd = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_mask_h = cb_id++;
    const auto cb_id_mask_w = cb_id++;

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? origin_h % TILE_H : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? origin_w % TILE_W : TILE_W;

    const auto Ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto Wt = (origin_w + TILE_W - 1) / TILE_W;

    const auto HtWt = Ht * Wt;
    const auto N = num_inner_tiles / HtWt;

    const auto C = num_channels;
    const auto CHtWt = C * HtWt;
    const auto NHtWt = N * HtWt;

    union {
        float f;
        uint32_t u;
    } one;
    one.f = 1.0f;
    fill_cb_with_value(cb_id_one, one.u);

    if (do_mask_h) {
        generate_mask_h(cb_id_mask_h, mask_h);
    }

    if (do_mask_w) {
        generate_mask_w(cb_id_mask_w, mask_w);
    }

    // output_grad
    const auto output_grad_addrg = TensorAccessor(output_grad_args, output_grad_addr);

    // input
    const auto input_addrg = TensorAccessor(input_args, input_addr);

    // mean
    const auto mean_addrg = TensorAccessor(mean_args, mean_addr);

    // rstd
    const auto rstd_addrg = TensorAccessor(rstd_args, rstd_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_output_grad(cb_id_output_grad);
    experimental::CircularBuffer cb_input(cb_id_input);
    experimental::CircularBuffer cb_mean(cb_id_mean);
    experimental::CircularBuffer cb_rstd(cb_id_rstd);

    const auto output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto input_tile_bytes = get_tile_size(cb_id_input);
    const auto mean_tile_bytes = get_tile_size(cb_id_mean);
    const auto rstd_tile_bytes = get_tile_size(cb_id_rstd);

    const auto mean_l1_write_ptr = cb_mean.get_write_ptr();
    const auto rstd_l1_write_ptr = cb_rstd.get_write_ptr();

    uint32_t mean_rstd_n_idx, mean_rstd_g_idx;
    uint32_t mean_rstd_tile_h_idx, mean_rstd_tile_w_idx;
    uint32_t mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile;
    uint32_t mean_rstd_Wt, mean_rstd_tile_idx, tilized_mean_rstd_idx_in_tile;

    uint32_t input_tile_idx;
    uint32_t output_grad_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_channels_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < NHtWt; ++inner_idx) {
            auto n_idx = inner_idx / HtWt;
            auto c_idx = outer_idx;
            auto htwt_idx = inner_idx % HtWt;

            // output_grad (N, C, H, W)
            output_grad_tile_idx = n_idx * CHtWt + c_idx * HtWt + htwt_idx + tile_offset;
            cb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                cb_output_grad,
                output_grad_tile_bytes,
                {.page_id = output_grad_tile_idx},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_output_grad.push_back(onetile);

            if (gamma_grad_has_value) {
                // input (N, C, H, W)
                input_tile_idx = output_grad_tile_idx;
                cb_input.reserve_back(onetile);
                noc.async_read(
                    input_addrg, cb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_input.push_back(onetile);

                // mean, rstd (1, 1, N, num_groups)
                // mean_rstd_idx = n * num_groups + g
                mean_rstd_n_idx = n_idx;
                mean_rstd_g_idx = c_idx % num_groups;

                mean_rstd_tile_h_idx = mean_rstd_n_idx / TILE_H;
                mean_rstd_tile_w_idx = mean_rstd_g_idx / TILE_W;

                mean_rstd_h_idx_in_tile = mean_rstd_n_idx % TILE_H;
                mean_rstd_w_idx_in_tile = mean_rstd_g_idx % TILE_W;

                mean_rstd_Wt = (num_groups + TILE_W - 1) / TILE_W;

                mean_rstd_tile_idx = mean_rstd_tile_h_idx * mean_rstd_Wt + mean_rstd_tile_w_idx;

                tilized_mean_rstd_idx_in_tile =
                    get_tilized_idx(mean_rstd_h_idx_in_tile, mean_rstd_w_idx_in_tile, TILE_H, TILE_W);

                // mean (1, 1, N, num_groups)
                cb_mean.reserve_back(onetile);
                noc.async_read(
                    mean_addrg, cb_mean, mean_tile_bytes, {.page_id = mean_rstd_tile_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if (tilized_mean_rstd_idx_in_tile != 0) {
                    experimental::CoreLocalMem<uint16_t> mean_ptr(mean_l1_write_ptr);
                    mean_ptr[0] = mean_ptr[tilized_mean_rstd_idx_in_tile];
                }
                cb_mean.push_back(onetile);

                // rstd (1, 1, N, num_groups)
                cb_rstd.reserve_back(onetile);
                noc.async_read(
                    rstd_addrg, cb_rstd, rstd_tile_bytes, {.page_id = mean_rstd_tile_idx}, {.offset_bytes = 0});
                noc.async_read_barrier();
                if (tilized_mean_rstd_idx_in_tile != 0) {
                    experimental::CoreLocalMem<uint16_t> rstd_ptr(rstd_l1_write_ptr);
                    rstd_ptr[0] = rstd_ptr[tilized_mean_rstd_idx_in_tile];
                }
                cb_rstd.push_back(onetile);
            }  // gamma_grad_has_value

        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
