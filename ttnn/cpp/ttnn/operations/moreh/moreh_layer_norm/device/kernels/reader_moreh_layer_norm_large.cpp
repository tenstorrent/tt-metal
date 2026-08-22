// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner = get_arg(args::num_inner);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto scaler = get_arg(args::scaler);
    const auto eps = get_arg(args::eps);
    const auto mask_h = get_arg(args::mask_h);
    const auto mask_w = get_arg(args::mask_w);

    constexpr auto block_size = get_arg(args::block_size);

    Noc noc;

    // Input DFB (bound as dfb::input); tile metadata now comes off the DFB object (whitelist rule 7).
    DataflowBuffer dfb_input(dfb::input);
    const uint32_t input_tile_bytes = dfb_input.get_tile_size();
    const auto input_data_format = dfb_input.get_dataformat();

    const auto input_addrg = TensorAccessor(tensor::input);

#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma(dfb::gamma);
    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const auto gamm_addrg = TensorAccessor(tensor::gamma);
#endif

#ifdef BETA_HAS_VALUE
    DataflowBuffer dfb_beta(dfb::beta);
    const uint32_t beta_tile_bytes = dfb_beta.get_tile_size();
    const auto beta_addrg = TensorAccessor(tensor::beta);
#endif

    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_eps(dfb::eps);
    fill_cb_with_value(dfb_scaler, scaler);
    fill_cb_with_value(dfb_eps, eps);

#ifdef DO_MASK_H
    {
        DataflowBuffer dfb_mask_h(dfb::mask_h);
        generate_mask_h(dfb_mask_h, mask_h);
    }
#endif

#ifdef DO_MASK_W
    {
        DataflowBuffer dfb_mask_w(dfb::mask_w);
        generate_mask_w(dfb_mask_w, mask_w);
    }
#endif

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        // For E[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    dfb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            dfb_input.push_back(block_size);
        }  // num_inner loop

        // For x - E[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    dfb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            dfb_input.push_back(block_size);
        }  // num_inner loop

        // For (x - E[x]) * (1.0/(sqrt(Var[x] + eps)))
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            dfb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    dfb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            dfb_input.push_back(block_size);

#ifdef GAMMA_HAS_VALUE
            dfb_gamma.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    gamm_addrg,
                    dfb_gamma,
                    gamma_tile_bytes,
                    {.page_id = inner_idx + r},
                    {.offset_bytes = r * gamma_tile_bytes});
            }
            noc.async_read_barrier();
            dfb_gamma.push_back(block_size);
#endif

#ifdef BETA_HAS_VALUE
            dfb_beta.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    beta_addrg,
                    dfb_beta,
                    beta_tile_bytes,
                    {.page_id = inner_idx + r},
                    {.offset_bytes = r * beta_tile_bytes});
            }
            noc.async_read_barrier();
            dfb_beta.push_back(block_size);
#endif
        }  // num_inner loop
        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
