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
    const auto batch = get_arg(args::batch);

    // GAMMA_GRAD_HAS_VALUE / BETA_GRAD_HAS_VALUE arrive as preprocessor defines rather than as
    // arguments: each selects whether the host binds the matching output tensor and its buffer, and
    // an optional output that is absent has nothing to bind at all.

    const auto HtWt = num_inner_tiles / batch;

#ifdef GAMMA_GRAD_HAS_VALUE
    // gamma_grad
    const auto gamma_grad_addrg = TensorAccessor(tensor::gamma_grad);
#endif

#ifdef BETA_GRAD_HAS_VALUE
    // beta_grad
    const auto beta_grad_addrg = TensorAccessor(tensor::beta_grad);
#endif

    constexpr uint32_t onetile = 1;

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    Noc noc;
#ifdef GAMMA_GRAD_HAS_VALUE
    DataflowBuffer dfb_gamma_grad(dfb::gamma_grad);
    const uint32_t gamma_grad_tile_bytes = dfb_gamma_grad.get_tile_size();
    const auto gamma_grad_l1_read_ptr = dfb_gamma_grad.get_read_ptr();
#endif
#ifdef BETA_GRAD_HAS_VALUE
    DataflowBuffer dfb_beta_grad(dfb::beta_grad);
    const uint32_t beta_grad_tile_bytes = dfb_beta_grad.get_tile_size();
    const auto beta_grad_l1_read_ptr = dfb_beta_grad.get_read_ptr();
#endif

    for (uint32_t outer_idx = 0; outer_idx < num_channels_per_core; ++outer_idx) {
        auto c_idx = outer_idx + (tile_offset / HtWt);

        // gamma_grad, beta_grad (1, 1, 1, C)
        const auto gamma_beta_c_idx = c_idx;
        const auto gamma_beta_tile_idx = gamma_beta_c_idx / TILE_W;
        const auto gamma_beta_w_idx_in_tile = gamma_beta_c_idx % TILE_W;
        const auto tilized_gamma_beta_idx_in_tile = get_tilized_idx(0, gamma_beta_w_idx_in_tile, TILE_H, TILE_W);

#ifdef GAMMA_GRAD_HAS_VALUE
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
            {.page_id = gamma_beta_tile_idx, .offset_bytes = tilized_gamma_beta_idx_in_tile * gamma_grad_dtype_bytes});
        noc.async_write_barrier();
        dfb_gamma_grad.pop_front(onetile);
#endif  // GAMMA_GRAD_HAS_VALUE

#ifdef BETA_GRAD_HAS_VALUE
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
            {.page_id = gamma_beta_tile_idx, .offset_bytes = tilized_gamma_beta_idx_in_tile * beta_grad_dtype_bytes});
        noc.async_write_barrier();
        dfb_beta_grad.pop_front(onetile);
#endif  // BETA_GRAD_HAS_VALUE

    }  // outer_idx loop

}  // void kernel_main()
