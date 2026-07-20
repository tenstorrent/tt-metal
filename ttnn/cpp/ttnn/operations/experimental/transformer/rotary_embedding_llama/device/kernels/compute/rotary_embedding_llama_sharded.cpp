// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

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

    CircularBuffer in_cb_obj(in_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);
    CircularBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    matmul_init(in_cb, trans_mat_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    // Get the trans_mat
    trans_mat_cb_obj.reserve_back(onetile);
    trans_mat_cb_obj.push_back(onetile);
    trans_mat_cb_obj.wait_front(onetile);

    // Get the sin/cos matrices
    // TODO: To parallelize across multiple batch, this should be in a batch loop
    sin_cb_obj.reserve_back(Wt);
    cos_cb_obj.reserve_back(Wt);

    sin_cb_obj.push_back(Wt);
    cos_cb_obj.push_back(Wt);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        rotated_in_interm_cb_obj.reserve_back(Wt);
        sin_interm_cb_obj.reserve_back(Wt);
        cos_interm_cb_obj.reserve_back(Wt);
        out_cb_obj.reserve_back(Wt);

        // Get the input
        in_cb_obj.reserve_back(Wt);
        in_cb_obj.push_back(Wt);
        in_cb_obj.wait_front(Wt);

        // Do the computation

        // rotated = x @ trans_mat
        matmul_init(in_cb, trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();
        rotated_in_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.wait_front(Wt);

        ckl::mul<
            rotated_in_interm_cb,
            sin_cb,
            sin_interm_cb,
            ckl::BroadcastDim::Row,
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::output(ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>(
            ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));

        ckl::mul<
            in_cb,
            cos_cb,
            cos_interm_cb,
            ckl::BroadcastDim::Row,
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block),
            ckl::output(ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>(
            ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));

        ckl::add<
            cos_interm_cb,
            sin_interm_cb,
            out_cb,
            ckl::BroadcastDim::None,
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::input(ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
            ckl::output(ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>(
            ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
    }

    // Done with the sin/cos matrices, so remove from CB
    sin_cb_obj.pop_front(Wt);
    cos_cb_obj.pop_front(Wt);

    // Done with the transformation matrix, so remove from CB
    trans_mat_cb_obj.pop_front(onetile);
}
