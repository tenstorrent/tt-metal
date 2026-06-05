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
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    // GELU backward: grad_in = grad_out * GELU'(input).
    //
    // 1D shape with explicit block size: EltwiseShape::tiles(n, block_size) emits
    // a chain that walks n tiles in chunks of block_size each. Per-element inits
    // (gelu_derivative_tile_init, mul_binary_tile_init, copy_tile_to_dst_init_short)
    // are hoisted once at chain entry; per-chunk InputLifecycle::Chunked lifecycle waits / pops
    // block_size tiles at a time. PF picks per_core_block_size as the largest
    // power-of-2 divisor of per_core_tile_cnt (<= 8).
    //
    // Lifecycles:
    //   cb_grad_out / cb_input  InputLifecycle::Chunked + Block (per-chunk wait+pop of per_core_block_size tiles)
    //   cb_grad_in              OutputLifecycle::Chunked + Block (per-chunk reserve+push)
    const auto shape = compute_kernel_lib::EltwiseShape::tiles(per_core_tile_cnt, per_core_block_size);

    compute_kernel_lib::eltwise_chain(
        shape,
        compute_kernel_lib::CopyTile<
            cb_grad_out,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::Chunked,
            compute_kernel_lib::CopyTileReconfig::None,
            compute_kernel_lib::OperandKind::Block>{},
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::InputLifecycle::Chunked,
            compute_kernel_lib::CopyTileReconfig::None,
            compute_kernel_lib::OperandKind::Block>{},
        compute_kernel_lib::GeluDerivative<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D1>{},
        compute_kernel_lib::
            MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_grad_in,
            compute_kernel_lib::OutputLifecycle::Chunked,
            compute_kernel_lib::PackTileReconfig::None>{});
}
