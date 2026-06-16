// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Functional validation for InputLifecycle::OuterStream — the streamed outer-axis broadcast.
//
// Chain: BinaryFpu(cb_a, cb_b) -> PackTile(cb_out) over grid(Ht, Wt).
//   cb_a: InputLifecycle::Streaming + OperandKind::Scalar — full Ht*Wt walk, one tile per
//         (ht, wt), read at the front, popped per tile.
//   cb_b: InputLifecycle::OuterStream + OperandKind::Scalar — ONE tile per row: waited at row
//         entry, re-read at the front across that row's Wt cols, popped at row exit. The
//         producer feeds a shallow (2-deep) CB one tile per row — O(1) L1, vs the Ht-deep
//         resident window a Bulk+Col operand would require.
// Net: out[ht*Wt + wt] = cb_a[ht*Wt + wt] + cb_b[ht].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::grid(Ht, Wt),
        compute_kernel_lib::BinaryFpu<
            cb_a,
            cb_b,
            compute_kernel_lib::BinaryFpuOp::Add,
            compute_kernel_lib::BroadcastDim::None,
            compute_kernel_lib::InputLifecycle::Streaming,    // cb_a: one tile per (ht, wt)
            compute_kernel_lib::InputLifecycle::OuterStream,  // cb_b: one tile per row
            compute_kernel_lib::BinaryDataFormatReconfig::None,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar,     // cb_a reads the front
            compute_kernel_lib::OperandKind::Scalar>{},  // cb_b reads the front (advances per row)
        compute_kernel_lib::PackTile<cb_out>{});
}
