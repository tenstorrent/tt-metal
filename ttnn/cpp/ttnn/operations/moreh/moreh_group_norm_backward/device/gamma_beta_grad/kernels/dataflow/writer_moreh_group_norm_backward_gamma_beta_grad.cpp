// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    int i{0};
    const auto gamma_grad_addr = get_arg_val<uint32_t>(i++);
    const auto beta_grad_addr = get_arg_val<uint32_t>(i++);

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_channels_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto batch = get_arg_val<uint32_t>(i++);

    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool beta_grad_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto gamma_grad_args = TensorAccessorArgs<2>();
    constexpr auto beta_grad_args = TensorAccessorArgs<gamma_grad_args.next_compile_time_args_offset()>();

    const auto HtWt = num_inner_tiles / batch;

    uint32_t cb_id{16};
    const auto cb_id_gamma_grad = cb_id++;
    const auto cb_id_beta_grad = cb_id++;

    // gamma_grad
    const uint32_t gamma_grad_tile_bytes = get_tile_size(cb_id_gamma_grad);
    const auto gamma_grad_addrg = TensorAccessor(gamma_grad_args, gamma_grad_addr);

    // beta_grad
    const uint32_t beta_grad_tile_bytes = get_tile_size(cb_id_beta_grad);
    const auto beta_grad_addrg = TensorAccessor(beta_grad_args, beta_grad_addr);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    Noc noc;
    DataflowBuffer dfb_gamma_grad(cb_id_gamma_grad);
    DataflowBuffer dfb_beta_grad(cb_id_beta_grad);

    const auto gamma_grad_l1_read_ptr = dfb_gamma_grad.get_read_ptr();
    const auto beta_grad_l1_read_ptr = dfb_beta_grad.get_read_ptr();

    for (uint32_t outer_idx = 0; outer_idx < num_channels_per_core; ++outer_idx) {
        auto c_idx = outer_idx + (tile_offset / HtWt);

        // gamma_grad, beta_grad (1, 1, 1, C)
        const auto gamma_beta_c_idx = c_idx;
        const auto gamma_beta_tile_idx = gamma_beta_c_idx / TILE_W;
        const auto gamma_beta_w_idx_in_tile = gamma_beta_c_idx % TILE_W;
        const auto tilized_gamma_beta_idx_in_tile = get_tilized_idx(0, gamma_beta_w_idx_in_tile, TILE_H, TILE_W);

        if (gamma_grad_has_value) {
            // gamma_grad (1, 1, 1, C)
            const auto gamma_grad_dtype_bytes = gamma_grad_tile_bytes / (TILE_H * TILE_W);
            dfb_gamma_grad.wait_front(onetile);
            if (tilized_gamma_beta_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> gamma_grad_ptr(gamma_grad_l1_read_ptr);
                gamma_grad_ptr[tilized_gamma_beta_idx_in_tile] = gamma_grad_ptr[0];
            }
            noc.async_write(
                dfb_gamma_grad,
                gamma_grad_addrg,
                gamma_grad_dtype_bytes,
                {.offset_bytes = tilized_gamma_beta_idx_in_tile * gamma_grad_dtype_bytes},
                {.page_id = gamma_beta_tile_idx,
                 .offset_bytes = tilized_gamma_beta_idx_in_tile * gamma_grad_dtype_bytes});
            noc.async_write_barrier();
            dfb_gamma_grad.pop_front(onetile);
        }

        if (beta_grad_has_value) {
            // beta_grad (1, 1, 1, C)
            const auto beta_grad_dtype_bytes = beta_grad_tile_bytes / (TILE_H * TILE_W);
            dfb_beta_grad.wait_front(onetile);
            if (tilized_gamma_beta_idx_in_tile != 0) {
                CoreLocalMem<uint16_t> beta_grad_ptr(beta_grad_l1_read_ptr);
                beta_grad_ptr[tilized_gamma_beta_idx_in_tile] = beta_grad_ptr[0];
            }
            noc.async_write(
                dfb_beta_grad,
                beta_grad_addrg,
                beta_grad_dtype_bytes,
                {.offset_bytes = tilized_gamma_beta_idx_in_tile * beta_grad_dtype_bytes},
                {.page_id = gamma_beta_tile_idx,
                 .offset_bytes = tilized_gamma_beta_idx_in_tile * beta_grad_dtype_bytes});
            noc.async_write_barrier();
            dfb_beta_grad.pop_front(onetile);
        }

    }  // outer_idx loop

}  // void kernel_main()
