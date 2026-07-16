// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto gamma_grad_addr = get_arg_val<uint32_t>(0);
    const auto beta_grad_addr = get_arg_val<uint32_t>(1);

    const auto num_cols_per_core = get_arg_val<uint32_t>(2);
    const auto tile_offset = get_arg_val<uint32_t>(3);

    constexpr bool gamma_grad_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool beta_grad_has_value = get_compile_time_arg_val(1) == 1;
    constexpr auto gamma_grad_args = TensorAccessorArgs<2>();
    constexpr auto beta_grad_args = TensorAccessorArgs<gamma_grad_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_gamma_grad = 16;
    constexpr uint32_t cb_id_beta_grad = 17;

    const auto gamma_grad_addrg = TensorAccessor(gamma_grad_args, gamma_grad_addr);
    const auto beta_grad_addrg = TensorAccessor(beta_grad_args, beta_grad_addr);

    constexpr uint32_t onetile = 1;

    const auto start_tile_idx = tile_offset;

    Noc noc;
    DataflowBuffer dfb_gamma_grad(cb_id_gamma_grad);
    DataflowBuffer dfb_beta_grad(cb_id_beta_grad);
    const auto gamma_grad_tile_bytes = get_tile_size(cb_id_gamma_grad);
    const auto beta_grad_tile_bytes = get_tile_size(cb_id_beta_grad);

    for (uint32_t w_idx = 0; w_idx < num_cols_per_core; w_idx++) {
        if (gamma_grad_has_value) {
            // gamma_grad (1, 1, 1, W)
            dfb_gamma_grad.wait_front(onetile);
            noc.async_write(
                dfb_gamma_grad,
                gamma_grad_addrg,
                gamma_grad_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = w_idx + start_tile_idx});
            noc.async_write_barrier();
            dfb_gamma_grad.pop_front(onetile);
        }  // gamma_grad_has_value

        if (beta_grad_has_value) {
            // beta_grad (1, 1, 1, W)
            dfb_beta_grad.wait_front(onetile);
            noc.async_write(
                dfb_beta_grad,
                beta_grad_addrg,
                beta_grad_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = w_idx + start_tile_idx});
            noc.async_write_barrier();
            dfb_beta_grad.pop_front(onetile);
        }  // beta_grad_has_value

    }  // num_cols_per_core loop
}  // void kernel_main()
