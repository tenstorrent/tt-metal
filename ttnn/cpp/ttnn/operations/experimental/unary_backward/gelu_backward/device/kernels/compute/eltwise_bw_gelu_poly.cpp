// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative.
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1).

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    // GELU backward: grad_in = grad_out * GELU'(input).
    //
    // Faithful 1:1 of the original per-tile loop: EltwiseShape::tiles(num_tiles)
    // (block_size = 1) streams one tile at a time. Per-element inits
    // (gelu_derivative_tile_init, mul_binary_tile_init, copy_tile_to_dst_init_short)
    // are hoisted once at chain entry; InputLifecycle::Streaming waits/pops 1 per
    // iter and OutputLifecycle::Streaming reserves/pushes 1 per iter — matching the
    // original cb_wait_front/cb_pop_front/cb_push_back of 1.
    const auto shape = compute_kernel_lib::EltwiseShape::tiles(num_tiles);

    compute_kernel_lib::eltwise_chain(
        shape,
        compute_kernel_lib::CopyTile<
            cb_grad_out,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::GeluDerivative<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D1>{},
        compute_kernel_lib::
            MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_grad_in,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
