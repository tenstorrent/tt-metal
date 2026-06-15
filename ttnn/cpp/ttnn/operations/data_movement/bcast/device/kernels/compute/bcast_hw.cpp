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

    // Bcast-HW (scalar bcast). Original `init_bcast<BCAST_LLKOP, BCAST_DIM>`
    // preserved as the big init. Chain's BinaryFpu uses CHAIN_BCAST_OP /
    // CHAIN_BCAST_DIM (helper-lib types emitted by bcast_op_utils).
    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_lhs, cb_rhs, cb_out);

    // BCAST_SCALAR flips the cb_rhs lifecycle:
    //   defined  -> cb_rhs is a single scalar tile held for the whole walk: read at the front
    //               (OperandKind::Scalar) for every output tile and never popped. The chain owns
    //               the wait via InputLifecycle::HeldStream (cb_wait_front(1) per iter, idempotent
    //               since never popped) — folding in what was a manual external wait_front +
    //               CallerManaged. Matches the original's wait-once / hold / never-pop held tile.
    //   undefined -> cb_rhs is waited+popped each iter (InputLifecycle::Streaming).
#ifdef BCAST_SCALAR
    constexpr auto rhs_lifecycle = compute_kernel_lib::InputLifecycle::HeldStream;
#else
    constexpr auto rhs_lifecycle = compute_kernel_lib::InputLifecycle::Streaming;
#endif

    // Flat 1D chain over total tiles (B*Ht*Wt) — bcast_hw is tile-by-tile,
    // no need for 2D shape.
    compute_kernel_lib::eltwise_chain(
        B * Ht * Wt,
        compute_kernel_lib::BinaryFpu<
            cb_lhs,
            cb_rhs,
            CHAIN_BCAST_OP,
            CHAIN_BCAST_DIM,
            compute_kernel_lib::InputLifecycle::Streaming,
            rhs_lifecycle,
            compute_kernel_lib::BinaryDataFormatReconfig::None>{},
        compute_kernel_lib::PackTile<
            cb_out,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
