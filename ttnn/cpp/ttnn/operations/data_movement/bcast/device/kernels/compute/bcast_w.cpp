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
    CircularBuffer cb_rhs_obj(cb_rhs);

    compute_kernel_hw_startup(cb_lhs, cb_rhs, cb_out);

    // Bcast-W: per-row of Wt tiles, cb_rhs is held across the inner loop
    // and popped at the end (1 scaler/row). cb_lhs is streamed (1 tile per
    // inner iter). HeldStream on cb_rhs gives chain-emitted cb_wait_front(1)
    // per iter without popping; explicit pop_front after the chain matches
    // the original's `cb_pop_front(cb_rhs, 1)` at end of h iter.
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
}
