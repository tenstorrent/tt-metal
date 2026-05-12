// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t i = 0;
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const auto beta_addr = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto scaler = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);
    const auto mask_h = get_arg_val<uint32_t>(i++);
    const auto mask_w = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_eps = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_4;
    constexpr uint32_t cb_id_mask_h = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_mask_w = tt::CBIndex::c_6;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto input_addrg = TensorAccessor(input_args, input_addr);

#ifdef GAMMA_HAS_VALUE
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto gamm_addrg = TensorAccessor(gamma_args, gamma_addr);
#endif

#ifdef BETA_HAS_VALUE
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto beta_addrg = TensorAccessor(beta_args, beta_addr);
#endif

    fill_cb_with_value(cb_id_scaler, scaler);
    fill_cb_with_value(cb_id_eps, eps);

#ifdef DO_MASK_H
    generate_mask_h(cb_id_mask_h, mask_h);
#endif

#ifdef DO_MASK_W
    generate_mask_w(cb_id_mask_w, mask_w);
#endif

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_input(cb_id_input);
#ifdef GAMMA_HAS_VALUE
    experimental::CircularBuffer cb_gamma(cb_id_gamma);
#endif
#ifdef BETA_HAS_VALUE
    experimental::CircularBuffer cb_beta(cb_id_beta);
#endif

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        // For E[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    cb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            cb_input.push_back(block_size);
        }  // num_inner loop

        // For x - E[x]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    cb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            cb_input.push_back(block_size);
        }  // num_inner loop

        // For (x - E[x]) * (1.0/(sqrt(Var[x] + eps)))
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_input.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    input_addrg,
                    cb_input,
                    input_tile_bytes,
                    {.page_id = offs + inner_idx + r + tile_offset},
                    {.offset_bytes = r * input_tile_bytes});
            }
            noc.async_read_barrier();
            cb_input.push_back(block_size);

#ifdef GAMMA_HAS_VALUE
            cb_gamma.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    gamm_addrg,
                    cb_gamma,
                    gamma_tile_bytes,
                    {.page_id = inner_idx + r},
                    {.offset_bytes = r * gamma_tile_bytes});
            }
            noc.async_read_barrier();
            cb_gamma.push_back(block_size);
#endif

#ifdef BETA_HAS_VALUE
            cb_beta.reserve_back(block_size);
            for (uint32_t r = 0; r < block_size; r++) {
                noc.async_read(
                    beta_addrg,
                    cb_beta,
                    beta_tile_bytes,
                    {.page_id = inner_idx + r},
                    {.offset_bytes = r * beta_tile_bytes});
            }
            noc.async_read_barrier();
            cb_beta.push_back(block_size);
#endif
        }  // num_inner loop
        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
