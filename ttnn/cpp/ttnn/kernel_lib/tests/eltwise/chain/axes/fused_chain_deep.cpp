// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Deep single-DEST chain used by the upfront-Bulk block-size benchmark:
//   out = relu(sigmoid(tanh(exp(A + B) * C)))
//
// Every compute element reads/writes D0; the chain therefore needs one DEST slot per block lane,
// exactly like the shallow chain. Exp, Tanh and Sigmoid deliberately have distinct SFPU init
// sequences. That makes the benchmark cover a realistic chain where grouping work into a block
// amortizes several setup sequences rather than only a simple binary operation.
//
// CT args: [n, block_size, life, batch]. `life` is retained so the Python fixture shares the
// same descriptor builder as fused_chain.cpp; this kernel supports only the Bulk path.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t life = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    static_assert(life == 0, "fused_chain_deep is an upfront-Bulk-only benchmark kernel");

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    for (uint32_t off = 0; off < n; off += batch) {
        eltwise_chain(
            IterationShape::tiles(batch).block_size(block_size),
            BinaryFpu<
                BinaryFpuOp::Add,
                input(cb_a, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                input(cb_b, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)>{},
            Exp<>{},
            DestReuseBinary<
                BinaryFpuOp::Mul,
                input(cb_c, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block),
                DestReuseType::DEST_TO_SRCA>{},
            Tanh<>{},
            Sigmoid<>{},
            Relu<>{},
            PackTile<output(cb_out, ReservePolicy::Upfront, PushPolicy::AtEnd)>{});
    }
}
