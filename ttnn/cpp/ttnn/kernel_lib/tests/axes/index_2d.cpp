// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Inter-tile index test (G3 / OK-03, OK-04 — the OperandKind *index* axis, not broadcast).
//
// 2D grid(Ht, Wt) add: out tile (ht,wt) = A tile (ht,wt) + B tile selected by B's index mode.
//   A: OperandKind::Block -> reads tile ht*Wt + wt (the full walk).
//   B: OperandKind::Row   -> reads tile wt (one tile per COLUMN, B has Wt tiles), or
//      OperandKind::Col   -> reads tile ht (one tile per ROW,    B has Ht tiles).
// BroadcastDim::None: a plain element-wise tile add — the only variable is which B tile is read,
// so this isolates inter-tile index selection. A Row<->Col index swap reads a different B tile and
// fails PCC. Row/Col require a non-streaming lifecycle (Bulk).

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
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                InputLifecycle::Bulk,
                InputLifecycle::Bulk,
                BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Row>{},
            PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
    } else {  // Col index on B
        eltwise_chain(
            EltwiseShape::grid(Ht, Wt),
            BinaryFpu<
                cb_a,
                cb_b,
                BinaryFpuOp::Add,
                BroadcastDim::None,
                InputLifecycle::Bulk,
                InputLifecycle::Bulk,
                BinaryDataFormatReconfig::Input,
                Dst::D0,
                OperandKind::Block,
                OperandKind::Col>{},
            PackTile<cb_out, OutputLifecycle::Bulk, PackTileReconfig::Output>{});
    }
}
