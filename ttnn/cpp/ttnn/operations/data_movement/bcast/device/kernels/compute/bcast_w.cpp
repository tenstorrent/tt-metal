// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Standard hw-config big init only. The chain's BinaryFpu emits the bcast MOP
    // (eltwise_chain.inl:209) each run, so init_bcast's per-op MOP was redundant;
    // compute_kernel_hw_startup provides the hw_configure + pack init.
    compute_kernel_hw_startup(cb_lhs, cb_rhs, cb_out);

    // bcast_w: one cb_rhs tile per row, broadcast (COL) across that row's Wt cb_lhs tiles.
    // The B*Ht row loop folds INTO the chain via grid(B*Ht, Wt): cb_rhs is a streamed
    // outer-axis broadcast (InputLifecycle::OuterStream + OperandKind::Scalar) — the chain
    // waits 1 at each row entry, re-reads it at the front across the row's cols, and pops 1
    // at row exit, exactly the original's per-row cb_wait_front(cb_rhs,1)/cb_pop_front(cb_rhs,1).
    // cb_lhs streams one tile per (row,col). CBs stay shallow (2-deep) — O(1) L1, no
    // reader/host change; nc1 batch-broadcast is handled by the reader's page reuse, so the
    // compute just consumes the B*Ht cb_rhs tiles in order. CHAIN_BCAST_DIM (COL) does the
    // intra-tile column broadcast. Folding the loop in hoists the per-op bcast init out of
    // the per-row path (emitted once, not B*Ht times).
    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(B * Ht, Wt),
        ckl::BinaryFpu<
            cb_lhs,
            cb_rhs,
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM,
            ckl::InputLifecycle::Streaming,    // cb_lhs: one tile per (row,col)
            ckl::InputLifecycle::OuterStream,  // cb_rhs: streamed broadcast, one per row
            ckl::BinaryDataFormatReconfig::None,
            ckl::Dst::D0,
            ckl::OperandKind::Scalar,     // cb_lhs reads the front
            ckl::OperandKind::Scalar>{},  // cb_rhs reads the front (advances per row)
        ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
