// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_lhs, cb_rhs, cb_out);

    // Bcast-H: reader supplies cb_rhs tile wrapped at Wt boundary, so per-tile
    // we stream cb_lhs + cb_rhs (both pop=1) and bcast op (row direction).
    // CHAIN_BCAST_OP / CHAIN_BCAST_DIM are macro-defined by bcast_op_utils
    // (BinaryFpuOp::{Add,Sub,Mul} / BroadcastDim::Row).
    //
    // Reconfig: original used init_bcast at boot then plain bcast op +
    // plain pack_tile — no per-iter reconfig. BinaryDataFormatReconfig::None +
    // PackTileReconfig::None preserve that.
    for (uint32_t b = 0; b < B; b++) {
        for (uint32_t h = 0; h < Ht; h++) {
            compute_kernel_lib::eltwise_chain(
                Wt,
                compute_kernel_lib::BinaryFpu<
                    cb_lhs,
                    cb_rhs,
                    CHAIN_BCAST_OP,
                    CHAIN_BCAST_DIM,
                    compute_kernel_lib::BinaryDataFormatReconfig::None,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    cb_out,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutStreaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }
    }
}
