// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t onetile = 1;
    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);

    constexpr auto cb_lhs = tt::CBIndex::c_0;
    constexpr auto cb_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    CircularBuffer cb_rhs_obj(cb_rhs);

    // Bcast-W: original `init_bcast<BCAST_LLKOP, BCAST_DIM>` preserved as the
    // big init. Chain's BinaryFpu uses CHAIN_BCAST_OP / CHAIN_BCAST_DIM
    // (helper-lib types emitted by bcast_op_utils).
    init_bcast<BCAST_LLKOP, BCAST_DIM>(cb_lhs, cb_rhs, cb_out);

    // bcast_w: 1 cb_rhs scalar per row, broadcast across Wt cb_lhs tiles.
    // cb_rhs is held across the inner Wt loop and popped once at end-of-row.
    // Outer B*Ht loop flattened. HeldStream gives chain-emitted per-iter
    // cb_wait_front(1) without popping; explicit pop_front after the chain
    // matches the original's cb_pop_front(cb_rhs, 1) at end of h iter.
    for (uint32_t row = 0; row < B * Ht; ++row) {
        compute_kernel_lib::eltwise_chain(
            Wt,
            compute_kernel_lib::BinaryFpu<
                cb_lhs,
                cb_rhs,
                CHAIN_BCAST_OP,
                CHAIN_BCAST_DIM,
                compute_kernel_lib::BinaryDataFormatReconfig::None,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_out,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
        cb_rhs_obj.pop_front(onetile);
    }
}
