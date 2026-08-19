// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_cols_per_core = get_arg(args::num_cols_per_core);
    const auto tile_offset = get_arg(args::tile_offset);

    // GAMMA_GRAD_HAS_VALUE / BETA_GRAD_HAS_VALUE arrive as preprocessor defines rather than as
    // arguments: each selects whether the host binds the matching output tensor and its buffer, and
    // an optional output that is absent has nothing to bind at all.

#ifdef GAMMA_GRAD_HAS_VALUE
    const auto gamma_grad_addrg = TensorAccessor(tensor::gamma_grad);
#endif
#ifdef BETA_GRAD_HAS_VALUE
    const auto beta_grad_addrg = TensorAccessor(tensor::beta_grad);
#endif

    constexpr uint32_t onetile = 1;

    const auto start_tile_idx = tile_offset;

    Noc noc;
#ifdef GAMMA_GRAD_HAS_VALUE
    DataflowBuffer dfb_gamma_grad(dfb::gamma_grad);
    const auto gamma_grad_tile_bytes = dfb_gamma_grad.get_tile_size();
#endif
#ifdef BETA_GRAD_HAS_VALUE
    DataflowBuffer dfb_beta_grad(dfb::beta_grad);
    const auto beta_grad_tile_bytes = dfb_beta_grad.get_tile_size();
#endif

    for (uint32_t w_idx = 0; w_idx < num_cols_per_core; w_idx++) {
#ifdef GAMMA_GRAD_HAS_VALUE
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
#endif  // GAMMA_GRAD_HAS_VALUE

#ifdef BETA_GRAD_HAS_VALUE
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
#endif  // BETA_GRAD_HAS_VALUE

    }  // num_cols_per_core loop
}  // void kernel_main()
