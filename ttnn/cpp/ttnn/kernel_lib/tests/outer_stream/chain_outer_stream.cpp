// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Functional validation for InputLifecycle::OuterStream — streamed outer-axis broadcast.
//
// BinaryFpu(cb_a, cb_b) -> PackTile(cb_out) over grid(Ht, Wt):
//   cb_a: Streaming + Scalar — full Ht*Wt walk, one tile per (ht, wt), popped per tile.
//   cb_b: OuterStream + Scalar — ONE tile per row: waited at row entry, re-read across the row's Wt
//         cols, popped at row exit. Producer feeds a shallow 2-deep CB (O(1) L1) instead of the
//         Ht-deep window a Bulk+Col operand needs.
// Net: out[ht*Wt + wt] = cb_a[ht*Wt + wt] + cb_b[ht].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(Ht, Wt),
        ckl::BinaryFpu<
            cb_a,
            cb_b,
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::input(ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
            ckl::input(ckl::InputLifecycle::OuterStream, ckl::DataFormatReconfig::Disabled)>{},
        ckl::PackTile<cb_out>{});
}
