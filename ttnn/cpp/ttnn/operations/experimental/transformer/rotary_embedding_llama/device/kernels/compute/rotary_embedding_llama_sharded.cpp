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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

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

    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto Ht = get_arg(args::Ht);  // How many rows (tiles) in n_heads dimension
    constexpr auto bulk_block_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto held_block_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](auto dfb_id) {
        return ckl::output(dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };

    DataflowBuffer in_dfb_obj(dfb::input);
    DataflowBuffer cos_dfb_obj(dfb::cos);
    DataflowBuffer sin_dfb_obj(dfb::sin);
    DataflowBuffer trans_mat_dfb_obj(dfb::trans_mat);
    DataflowBuffer rotated_in_interm_dfb_obj(dfb::rotated_interm);
    DataflowBuffer cos_interm_dfb_obj(dfb::cos_interm);
    DataflowBuffer sin_interm_dfb_obj(dfb::sin_interm);
    DataflowBuffer out_dfb_obj(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::input, dfb::trans_mat, dfb::out);
    matmul_init(dfb::input, dfb::trans_mat);
    compute_kernel_hw_startup(dfb::rotated_interm, dfb::sin, dfb::sin_interm);  // General Init for all binary ops

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
        matmul_init(dfb::input, dfb::trans_mat);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(dfb::input, dfb::trans_mat, j, 0, j);
            pack_tile(j, dfb::rotated_interm, j);
        }
        REL();
        rotated_in_interm_dfb_obj.push_back(Wt);
        mul_bcast_rows_init(dfb::rotated_interm, dfb::sin);
        // sin_interim = rotated * sin
        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                bulk_block_input(dfb::rotated_interm),
                ckl::input(held_block_input(dfb::sin), ckl::BroadcastDim::Row)>{},
            ckl::PackTile<bulk_output(dfb::sin_interm)>{});

        // cos_interim = x * cos
        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::input,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::AtEnd,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Disabled),
                ckl::input(held_block_input(dfb::cos), ckl::BroadcastDim::Row)>{},
            ckl::PackTile<bulk_output(dfb::cos_interm)>{});

        // out = cos_interim + sin_interim
        ckl::add<bulk_block_input(dfb::cos_interm), bulk_block_input(dfb::sin_interm), bulk_output(dfb::out)>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt));
    }

    // Done with the sin/cos matrices, so remove from DFB
    sin_dfb_obj.pop_front(Wt);
    cos_dfb_obj.pop_front(Wt);

    // Done with the transformation matrix, so remove from DFB
    trans_mat_dfb_obj.pop_front(onetile);
}
