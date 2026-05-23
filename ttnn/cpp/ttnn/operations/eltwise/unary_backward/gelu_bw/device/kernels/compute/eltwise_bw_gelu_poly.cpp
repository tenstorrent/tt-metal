// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // GeluDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    binary_op_init_common(cb_grad_out, cb_input, cb_grad_in);

    // GELU backward: grad_in = grad_out * GELU'(input).
    //   D0 = grad_out, D1 = input -> GeluDerivative<D1>, MulBinary<D0, D1, D0>.
    // Original comment notes that GELU derivative uses extra DEST as scratch
    // during polynomial evaluation, so the multi-tile batching isn't possible
    // there. The chain's per-tile iteration handles that constraint naturally —
    // each iteration acquires fresh DEST, runs derivative, and packs result.
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        compute_kernel_lib::eltwise_chain(
            per_core_block_size,
            compute_kernel_lib::CopyTile<
                cb_grad_out,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::GeluDerivative<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D1>{},
            compute_kernel_lib::
                MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_grad_in,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
    }
}
