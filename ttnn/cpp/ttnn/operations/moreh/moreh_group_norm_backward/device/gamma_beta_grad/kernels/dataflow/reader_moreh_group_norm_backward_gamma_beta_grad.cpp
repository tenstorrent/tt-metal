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
    const auto num_channels_per_core = get_arg(args::num_channels_per_core);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);
    const auto num_channels = get_arg(args::num_channels);
    const auto num_groups = get_arg(args::num_groups);

    const auto origin_h = get_arg(args::origin_h);
    const auto origin_w = get_arg(args::origin_w);

    // GAMMA_GRAD_HAS_VALUE / DO_MASK_H / DO_MASK_W arrive as preprocessor defines rather than as
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
    // The shared compute kernel consumes this buffer as its reduce scaler, so the binding carries the
    // kernel's name for it; this op fills it with 1.0.
    DataflowBuffer dfb_scaler(dfb::scaler);
    fill_cb_with_value(dfb_scaler, one.u);

#ifdef DO_MASK_H
    DataflowBuffer dfb_mask_h(dfb::mask_h);
    generate_mask_h(dfb_mask_h, mask_h);
#endif

#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w(dfb::mask_w);
    generate_mask_w(dfb_mask_w, mask_w);
#endif

    // output_grad
    const auto output_grad_addrg = TensorAccessor(tensor::output_grad);

    // input
    const auto input_addrg = TensorAccessor(tensor::input);

    // mean
    const auto mean_addrg = TensorAccessor(tensor::mean);

    // rstd
    const auto rstd_addrg = TensorAccessor(tensor::rstd);

    Noc noc;
    DataflowBuffer dfb_output_grad(dfb::output_grad);
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_mean(dfb::mean);
    DataflowBuffer dfb_rstd(dfb::rstd);

    const auto output_grad_tile_bytes = dfb_output_grad.get_tile_size();
    const auto input_tile_bytes = dfb_input.get_tile_size();
    const auto mean_tile_bytes = dfb_mean.get_tile_size();
    const auto rstd_tile_bytes = dfb_rstd.get_tile_size();

    const auto mean_l1_write_ptr = dfb_mean.get_write_ptr();
    const auto rstd_l1_write_ptr = dfb_rstd.get_write_ptr();

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
            dfb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                dfb_output_grad,
                output_grad_tile_bytes,
                {.page_id = output_grad_tile_idx},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_output_grad.push_back(onetile);

#ifdef GAMMA_GRAD_HAS_VALUE
            // input (N, C, H, W)
            input_tile_idx = output_grad_tile_idx;
            dfb_input.reserve_back(onetile);
            noc.async_read(input_addrg, dfb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);

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
#endif  // GAMMA_GRAD_HAS_VALUE

        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
