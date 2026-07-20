// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Avoids catastrophic cancellation in the naive 1 - tanh²(x) formula.

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // TanhDerivative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    const auto shape = ckl::EltwiseShape::tiles(per_core_tile_cnt, per_core_block_size);

    ckl::eltwise_chain(
        shape,
        ckl::CopyTile<
            cb_grad_out,
            ckl::Dst::D0,
            ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled)>{},
        ckl::CopyTile<
            cb_input,
            ckl::Dst::D1,
            ckl::input(ckl::InputLifecycle::Chunked, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled)>{},
        ckl::TanhDerivative<ckl::Approx::Exact, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<cb_grad_in, ckl::output(ckl::OutputLifecycle::Chunked, ckl::DataFormatReconfig::Disabled)>{});
}
