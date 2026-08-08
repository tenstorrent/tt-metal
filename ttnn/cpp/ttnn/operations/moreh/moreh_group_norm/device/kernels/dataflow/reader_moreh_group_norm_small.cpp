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
    const auto scaler = get_arg(args::scaler);
    const auto eps = get_arg(args::eps);

    const auto tile_offset = get_arg(args::tile_offset);
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);
    const auto num_channels = get_arg(args::num_channels);

    const auto origin_h = get_arg(args::origin_h);
    const auto origin_w = get_arg(args::origin_w);
    const auto block_size = get_arg(args::block_size);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const auto Ht = (origin_h + TILE_H - 1) / TILE_H;
    const auto Wt = (origin_w + TILE_W - 1) / TILE_W;

    const auto HtWt = Ht * Wt;

    const auto C = num_channels;

    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_eps(dfb::eps);
    fill_cb_with_value(dfb_scaler, scaler);
    fill_cb_with_value(dfb_eps, eps);

#ifdef DO_MASK_H
    {
        const auto mask_h = origin_h % TILE_H;
        DataflowBuffer dfb_mask_h(dfb::mask_h);
        generate_mask_h(dfb_mask_h, mask_h);
    }
#endif
#ifdef DO_MASK_W
    {
        const auto mask_w = origin_w % TILE_W;
        DataflowBuffer dfb_mask_w(dfb::mask_w);
        generate_mask_w(dfb_mask_w, mask_w);
    }
#endif

    Noc noc;

    // input
    DataflowBuffer dfb_input(dfb::input);
    const uint32_t input_tile_bytes = dfb_input.get_tile_size();
    const auto input_addrg = TensorAccessor(tensor::input);

    // gamma
#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma(dfb::gamma);
    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const auto gamma_addrg = TensorAccessor(tensor::gamma);
#endif

    // beta
#ifdef BETA_HAS_VALUE
    DataflowBuffer dfb_beta(dfb::beta);
    const uint32_t beta_tile_bytes = dfb_beta.get_tile_size();
    const auto beta_addrg = TensorAccessor(tensor::beta);
#endif

    uint32_t input_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        dfb_input.reserve_back(num_inner_tiles);
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            dfb_input.reserve_back(num_inner_tiles);
            input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            noc.async_read(
                input_addrg,
                dfb_input,
                input_tile_bytes,
                {.page_id = input_tile_idx},
                {.offset_bytes = inner_idx * input_tile_bytes});
        }  // inner_idx loop
        noc.async_read_barrier();
        dfb_input.push_back(num_inner_tiles);

        // input (N, C, H, W)
        // input_tile_idx = n * C * Ht * Wt + c * Ht * Wt + h * Wt + w
        // n * C + c = input_tile_idx / (Ht * Wt)
        // c = (input_tile_idx / (Ht * Wt)) % C
        // gamma (1, 1, 1, C)
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; inner_idx += block_size) {
#ifdef GAMMA_HAS_VALUE
            {
                uint32_t gamma_tile_idx;
                const auto gamma_l1_write_ptr = dfb_gamma.get_write_ptr();
                dfb_gamma.reserve_back(block_size);
                for (uint32_t r = 0; r < block_size; r++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                    gamma_tile_idx = get_gamma_beta_tile_idx(input_tile_idx, HtWt, C, TILE_W);
                    noc.async_read(
                        gamma_addrg,
                        dfb_gamma,
                        gamma_tile_bytes,
                        {.page_id = gamma_tile_idx},
                        {.offset_bytes = r * gamma_tile_bytes});
                }
                noc.async_read_barrier();

                uint32_t tilized_gamma_idx_in_tile;
                for (uint32_t q = 0; q < block_size; q++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + q;
                    tilized_gamma_idx_in_tile =
                        get_tilized_gamma_beta_idx_in_tile(input_tile_idx, HtWt, C, TILE_H, TILE_W);
                    if (tilized_gamma_idx_in_tile != 0) {
                        CoreLocalMem<uint16_t> gamma_ptr(gamma_l1_write_ptr + q * gamma_tile_bytes);
                        gamma_ptr[0] = gamma_ptr[tilized_gamma_idx_in_tile];
                    }
                }
                dfb_gamma.push_back(block_size);
            }
#endif

            // beta (1, 1, 1, C)
#ifdef BETA_HAS_VALUE
            {
                uint32_t beta_tile_idx;
                const auto beta_l1_write_ptr = dfb_beta.get_write_ptr();
                dfb_beta.reserve_back(block_size);
                for (uint32_t r = 0; r < block_size; r++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + r;
                    beta_tile_idx = get_gamma_beta_tile_idx(input_tile_idx, HtWt, C, TILE_W);
                    noc.async_read(
                        beta_addrg,
                        dfb_beta,
                        beta_tile_bytes,
                        {.page_id = beta_tile_idx},
                        {.offset_bytes = r * beta_tile_bytes});
                }
                noc.async_read_barrier();

                uint32_t tilized_beta_idx_in_tile;
                for (uint32_t q = 0; q < block_size; q++) {
                    input_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx + q;
                    tilized_beta_idx_in_tile =
                        get_tilized_gamma_beta_idx_in_tile(input_tile_idx, HtWt, C, TILE_H, TILE_W);
                    if (tilized_beta_idx_in_tile != 0) {
                        CoreLocalMem<uint16_t> beta_ptr(beta_l1_write_ptr + q * beta_tile_bytes);
                        beta_ptr[0] = beta_ptr[tilized_beta_idx_in_tile];
                    }
                }
                dfb_beta.push_back(block_size);
            }
#endif
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
