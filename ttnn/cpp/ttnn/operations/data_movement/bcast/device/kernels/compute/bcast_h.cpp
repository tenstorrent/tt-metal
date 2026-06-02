// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Bcast-H: original `init_bcast<BCAST_LLKOP, BCAST_DIM>` preserved as the
    // big init (hw_configure + pack_dest_init + sync_init + per-op MOP). Chain's
    // BinaryFpu uses CHAIN_BCAST_OP / CHAIN_BCAST_DIM (helper-lib types emitted
    // by bcast_op_utils alongside the LLK enum macros).
    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_lhs, cb_rhs, cb_out);

    // Reader supplies cb_rhs wrapped at Wt boundary, so per-tile we stream
    // cb_lhs + cb_rhs and apply the bcast op. Flat 1D chain over total tiles
    // (B*Ht*Wt) — bcast_h is tile-by-tile, no need for 2D shape.
    //
    // Reconfig: original used init_bcast at boot then plain bcast op + plain
    // pack_tile — no per-iter reconfig. BinaryDataFormatReconfig::None +
    // PackTileReconfig::None preserve that.
    compute_kernel_lib::eltwise_chain(
        B * Ht * Wt,
        compute_kernel_lib::BinaryFpu<
            cb_lhs,
            cb_rhs,
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM,
            compute_kernel_lib::BinaryDataFormatReconfig::None,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::InputLifecycle::Streaming,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::OperandKind::Scalar>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
