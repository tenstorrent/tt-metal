// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Inter-tile index test (the OperandKind *index* axis, not broadcast).
// 2D grid(Ht, Wt) add: out(ht,wt) = A(ht,wt) + B tile selected by B's index mode.
//   A: Block -> tile ht*Wt + wt (full walk)
//   B: Row   -> tile wt (one per COLUMN, Wt tiles)    Col -> tile ht (one per ROW, Ht tiles)
// BroadcastDim::None (plain tile add), so the only variable is which B tile is read — a Row<->Col
// swap reads a different tile and fails PCC. Row/Col need a non-streaming lifecycle (Bulk).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t mode = get_compile_time_arg_val(2);  // 1 = Col index, 2 = Row index

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    if constexpr (mode == 2) {  // Row index on B
        eltwise_chain(
            EltwiseShape::grid(Ht, Wt),
            BinaryFpu<
                input(cb_a, InputLifecycle::Bulk, OperandKind::Block),
                input(cb_b, InputLifecycle::Bulk, OperandKind::Row),
                BinaryFpuOp::Add,
                BroadcastDim::None>{},
            PackTile<output(cb_out, OutputLifecycle::Bulk)>{});
    } else {  // Col index on B
        eltwise_chain(
            EltwiseShape::grid(Ht, Wt),
            BinaryFpu<
                input(cb_a, InputLifecycle::Bulk, OperandKind::Block),
                input(cb_b, InputLifecycle::Bulk, OperandKind::Col),
                BinaryFpuOp::Add,
                BroadcastDim::None>{},
            PackTile<output(cb_out, OutputLifecycle::Bulk)>{});
    }
}
