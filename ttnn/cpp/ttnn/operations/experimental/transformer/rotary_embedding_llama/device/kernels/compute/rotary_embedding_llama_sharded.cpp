// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

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
    // Magic CB indices are gone: each buffer is a named DFB binding. The local
    // aliases keep the LLK/FIFO call sites readable; each is the dfb:: handle.
    constexpr auto in_dfb = dfb::input;
    constexpr auto cos_dfb = dfb::cos;
    constexpr auto sin_dfb = dfb::sin;
    constexpr auto trans_mat_dfb = dfb::trans_mat;

    constexpr auto rotated_in_interm_dfb = dfb::rotated_interm;
    constexpr auto cos_interm_dfb = dfb::cos_interm;
    constexpr auto sin_interm_dfb = dfb::sin_interm;
    constexpr auto out_dfb = dfb::out;
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto Ht = get_arg(args::Ht);  // How many rows (tiles) in n_heads dimension
    constexpr auto bulk_block_input = [](auto cb) {
        return ckl::input(
            cb,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto held_block_input = [](auto cb) {
        return ckl::input(
            cb,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::None,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](auto cb) {
        return ckl::output(cb, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };

    DataflowBuffer in_dfb_obj(in_dfb);
    DataflowBuffer cos_dfb_obj(cos_dfb);
    DataflowBuffer sin_dfb_obj(sin_dfb);
    DataflowBuffer trans_mat_dfb_obj(trans_mat_dfb);
    DataflowBuffer rotated_in_interm_dfb_obj(rotated_in_interm_dfb);
    DataflowBuffer cos_interm_dfb_obj(cos_interm_dfb);
    DataflowBuffer sin_interm_dfb_obj(sin_interm_dfb);
    DataflowBuffer out_dfb_obj(out_dfb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_dfb, trans_mat_dfb, out_dfb);
    matmul_init(in_dfb, trans_mat_dfb);
    compute_kernel_hw_startup(rotated_in_interm_dfb, sin_dfb, sin_interm_dfb);  // General Init for all binary ops

    // Get the trans_mat
    trans_mat_dfb_obj.reserve_back(onetile);
    trans_mat_dfb_obj.push_back(onetile);
    trans_mat_dfb_obj.wait_front(onetile);

    // Get the sin/cos matrices
    // TODO: To parallelize across multiple batch, this should be in a batch loop
    sin_dfb_obj.reserve_back(Wt);
    cos_dfb_obj.reserve_back(Wt);

    sin_dfb_obj.push_back(Wt);
    cos_dfb_obj.push_back(Wt);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        rotated_in_interm_dfb_obj.reserve_back(Wt);
        sin_interm_dfb_obj.reserve_back(Wt);
        cos_interm_dfb_obj.reserve_back(Wt);
        out_dfb_obj.reserve_back(Wt);

        // Get the input
        in_dfb_obj.reserve_back(Wt);
        in_dfb_obj.push_back(Wt);
        in_dfb_obj.wait_front(Wt);

        // Do the computation

        // rotated = x @ trans_mat
        matmul_init(in_dfb, trans_mat_dfb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_dfb, trans_mat_dfb, j, 0, j);
            pack_tile(j, rotated_in_interm_dfb, j);
        }
        REL();
        rotated_in_interm_dfb_obj.push_back(Wt);
        mul_bcast_rows_init(rotated_in_interm_dfb, sin_dfb);
        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
            ckl::BinaryFpu<
                bulk_block_input(rotated_in_interm_dfb),
                held_block_input(sin_dfb),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Row>{},
            ckl::PackTile<bulk_output(sin_interm_dfb)>{});

        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
            ckl::BinaryFpu<
                ckl::input(
                    in_dfb,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Disabled),
                held_block_input(cos_dfb),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::Row>{},
            ckl::PackTile<bulk_output(cos_interm_dfb)>{});

        ckl::add<
            bulk_block_input(cos_interm_dfb),
            bulk_block_input(sin_interm_dfb),
            bulk_output(out_dfb),
            ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
    }

    // Done with the sin/cos matrices, so remove from CB
    sin_dfb_obj.pop_front(Wt);
    cos_dfb_obj.pop_front(Wt);

    // Done with the transformation matrix, so remove from CB
    trans_mat_dfb_obj.pop_front(onetile);
}
