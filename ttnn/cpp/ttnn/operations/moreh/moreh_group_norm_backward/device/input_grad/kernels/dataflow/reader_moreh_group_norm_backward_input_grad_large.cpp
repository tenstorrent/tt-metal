// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto tile_offset = get_arg(args::tile_offset);
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);
    const auto num_channels = get_arg(args::num_channels);
    const auto num_groups = get_arg(args::num_groups);

    const auto origin_h = get_arg(args::origin_h);
    const auto origin_w = get_arg(args::origin_w);

    // GAMMA_HAS_VALUE / DO_MASK_H / DO_MASK_W arrive as preprocessor defines rather than as
    // arguments, because each selects whether the host binds a resource; a name the host did not bind
    // does not exist in this build, and even a discarded `if constexpr` branch would still look it up.

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
    // The shared compute kernel consumes this buffer as its reduce scaler, so the binding carries the
    // kernel's name for it; this op fills it with 1.0.
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_n_recip_n(dfb::n_recip_n);
    fill_cb_with_value(dfb_scaler, scalar.u);

    const auto n = static_cast<float>((num_channels / num_groups) * origin_h * origin_w);
    scalar.f = n;
    fill_cb_with_value(dfb_n_recip_n, scalar.u);
    scalar.f = 1.0f / n;
    fill_cb_with_value(dfb_n_recip_n, scalar.u);

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    DataflowBuffer dfb_mask_h_w(dfb::mask_h_w);
    generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w, dfb_mask_h_w.get_tile_size());
#endif

    // output_grad
    const auto output_grad_addrg = TensorAccessor(tensor::output_grad);

    // input
    const auto input_addrg = TensorAccessor(tensor::input);

    // mean
    const auto mean_addrg = TensorAccessor(tensor::mean);

    // rstd
    const auto rstd_addrg = TensorAccessor(tensor::rstd);

#ifdef GAMMA_HAS_VALUE
    // gamma
    const auto gamma_addrg = TensorAccessor(tensor::gamma);
#endif

    const auto start_mean_rstd_idx = tile_offset / num_inner_tiles;

    Noc noc;
    DataflowBuffer dfb_output_grad(dfb::output_grad);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_mean(dfb::mean);
    DataflowBuffer dfb_rstd(dfb::rstd);
#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma(dfb::gamma);
#endif

    const auto output_grad_tile_bytes = dfb_output_grad.get_tile_size();
    const auto input_tile_bytes = dfb_input.get_tile_size();
    const uint32_t mean_tile_bytes = dfb_mean.get_tile_size();
    const auto rstd_tile_bytes = dfb_rstd.get_tile_size();
#ifdef GAMMA_HAS_VALUE
    const auto gamma_tile_bytes = dfb_gamma.get_tile_size();
#endif

    const auto mean_dtype_bytes = mean_tile_bytes / (TILE_H * TILE_W);
    const auto rstd_dtype_bytes = rstd_tile_bytes / (TILE_H * TILE_W);

    const auto mean_l1_write_ptr = dfb_mean.get_write_ptr();
    const auto rstd_l1_write_ptr = dfb_rstd.get_write_ptr();
#ifdef GAMMA_HAS_VALUE
    const auto gamma_l1_write_ptr = dfb_gamma.get_write_ptr();
#endif

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
        dfb_mean.reserve_back(onetile);
        noc.async_read(mean_addrg, dfb_mean, mean_tile_bytes, {.page_id = mean_rstd_tile_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        if (tilized_mean_rstd_idx_in_tile != 0) {
            CoreLocalMem<uint16_t> mean_ptr(mean_l1_write_ptr);
            mean_ptr[0] = mean_ptr[tilized_mean_rstd_idx_in_tile];
        }
        dfb_mean.push_back(onetile);

        // rstd (1, 1, N, num_groups)
        dfb_rstd.reserve_back(onetile);
        noc.async_read(rstd_addrg, dfb_rstd, rstd_tile_bytes, {.page_id = mean_rstd_tile_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        if (tilized_mean_rstd_idx_in_tile != 0) {
            CoreLocalMem<uint16_t> rstd_ptr(rstd_l1_write_ptr);
            rstd_ptr[0] = rstd_ptr[tilized_mean_rstd_idx_in_tile];
        }
        dfb_rstd.push_back(onetile);

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // input (N, C, H, W)
            input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            dfb_input.reserve_back(onetile);
            noc.async_read(input_addrg, dfb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);

            // output_grad (N, C, H, W)
            output_grad_tile_idx = input_tile_idx;
            dfb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                dfb_output_grad,
                output_grad_tile_bytes,
                {.page_id = output_grad_tile_idx},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_output_grad.push_back(onetile);

#ifdef GAMMA_HAS_VALUE
            // gamma (1, 1, 1, C)
            const auto gamma_c_idx = (input_tile_idx / (Ht * Wt)) % C;
            const auto gamma_tile_idx = gamma_c_idx / TILE_W;
            const auto gamma_w_idx_in_tile = gamma_c_idx % TILE_W;
            const auto tilized_gamma_idx_in_tile = get_tilized_idx(0, gamma_w_idx_in_tile, TILE_H, TILE_W);
            dfb_gamma.reserve_back(onetile);
            noc.async_read(gamma_addrg, dfb_gamma, gamma_tile_bytes, {.page_id = gamma_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if (tilized_gamma_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> gamma_ptr(gamma_l1_write_ptr);
                gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
            }
            dfb_gamma.push_back(onetile);
#endif  // GAMMA_HAS_VALUE
        }  // inner_idx loop

        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // output_grad (N, C, H, W)
            output_grad_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            dfb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                dfb_output_grad,
                output_grad_tile_bytes,
                {.page_id = output_grad_tile_idx},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_output_grad.push_back(onetile);

#ifdef GAMMA_HAS_VALUE
            // gamma (1, 1, 1, C)
            const auto gamma_c_idx = (output_grad_tile_idx / (Ht * Wt)) % C;
            const auto gamma_tile_idx = gamma_c_idx / TILE_W;
            const auto gamma_w_idx_in_tile = gamma_c_idx % TILE_W;
            const auto tilized_gamma_idx_in_tile = get_tilized_idx(0, gamma_w_idx_in_tile, TILE_H, TILE_W);
            dfb_gamma.reserve_back(onetile);
            noc.async_read(gamma_addrg, dfb_gamma, gamma_tile_bytes, {.page_id = gamma_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if (tilized_gamma_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> gamma_ptr(gamma_l1_write_ptr);
                gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
            }
            dfb_gamma.push_back(onetile);
#endif  // GAMMA_HAS_VALUE

            // input (N, C, H, W)
            input_tile_idx = output_grad_tile_idx;
            dfb_input.reserve_back(onetile);
            noc.async_read(input_addrg, dfb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
