// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // How many rows (tiles) in n_heads dimension

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    // Get the trans_mat
    cb_reserve_back(trans_mat_cb, onetile);
    cb_push_back(trans_mat_cb, onetile);
    cb_wait_front(trans_mat_cb, onetile);

    // Get the sin/cos matrices
    // TODO: To parallelize across multiple batch, this should be in a batch loop
    cb_reserve_back(sin_cb, Wt);
    cb_reserve_back(cos_cb, Wt);

    cb_push_back(sin_cb, Wt);
    cb_push_back(cos_cb, Wt);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(sin_interm_cb, Wt);
        cb_reserve_back(cos_interm_cb, Wt);
        cb_reserve_back(out_cb, Wt);

        // Get the input
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // Do the computation

        // rotated = x @ trans_mat
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();
        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        // sin_interim = rotated * sin (bcast ROW).
        // rotated InputLifecycle::Bulk + Block; sin InputLifecycle::HeldBulk + Block (pre-waited line 42, popped at
        // 106). sin_interm OutputLifecycle::Bulk + Block. Reconfig Input + None (no explicit pack_reconfig).
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
            compute_kernel_lib::BinaryFpu<
                rotated_in_interm_cb,
                sin_cb,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Row,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::InputLifecycle::HeldBulk,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block>{},
            compute_kernel_lib::PackTile<
                sin_interm_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::None>{});

        // cos_interim = x * cos (bcast ROW). Same pattern as sin_interim.
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
            compute_kernel_lib::BinaryFpu<
                in_cb,
                cos_cb,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Row,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::InputLifecycle::HeldBulk,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block>{},
            compute_kernel_lib::PackTile<
                cos_interm_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::None>{});

        // out = cos_interim + sin_interim. Both InputLifecycle::Bulk + Block, out_cb OutputLifecycle::Bulk + Block.
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
            compute_kernel_lib::BinaryFpu<
                cos_interm_cb,
                sin_interm_cb,
                compute_kernel_lib::BinaryFpuOp::Add,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::InputLifecycle::Bulk,
                compute_kernel_lib::OperandKind::Block,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Block>{},
            compute_kernel_lib::PackTile<
                out_cb,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutputLifecycle::Bulk,
                compute_kernel_lib::PackTileReconfig::None>{});
    }

    // Done with the sin/cos matrices, so remove from CB
    cb_pop_front(sin_cb, Wt);
    cb_pop_front(cos_cb, Wt);

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
