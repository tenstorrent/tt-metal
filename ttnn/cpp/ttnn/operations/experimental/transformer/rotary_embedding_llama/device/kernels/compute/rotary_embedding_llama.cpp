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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
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
    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr uint32_t onetile = 1;

    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);
    constexpr auto bulk_block_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](auto dfb_id) {
        return ckl::output(dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto sin_cos_input = [](auto dfb_id) {
        return ckl::input(
            dfb_id,
            RELOAD_IMPL == 0 ? ckl::WaitPolicy::None : ckl::WaitPolicy::Upfront,
            RELOAD_IMPL == 0 ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled,
            RELOAD_IMPL == 0 ? ckl::TileOffset::Set : ckl::TileOffset::Unset);
    };

    DataflowBuffer in_dfb_obj(dfb::input);
    DataflowBuffer cos_dfb_obj(dfb::cos);
    DataflowBuffer sin_dfb_obj(dfb::sin);
    DataflowBuffer trans_mat_dfb_obj(dfb::trans_mat);
    DataflowBuffer rotated_in_interm_dfb_obj(dfb::rotated_interm);
    DataflowBuffer cos_interm_dfb_obj(dfb::cos_interm);
    DataflowBuffer sin_interm_dfb_obj(dfb::sin_interm);
    DataflowBuffer out_dfb_obj(dfb::out);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::input, dfb::trans_mat, dfb::out);
    // Start from the state at the end of each iteration so same-format reconfigurations compile out.
    // TODO(#52395): compute_kernel_hw_startup is a call-once API and should be the kernel's first Tensix-engine call,
    // but here it follows another engine op (init_sfpu / a prior startup); see the issue.
    compute_kernel_hw_startup(dfb::cos_interm, dfb::sin_interm, dfb::out);

    // Get the trans_mat
    trans_mat_dfb_obj.wait_front(onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_dfb_obj.wait_front(my_cos_sin_tiles);
            cos_dfb_obj.wait_front(my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input dfb_id wait and reserve
                in_dfb_obj.wait_front(Wt);
#if RELOAD_IMPL == 1
                sin_dfb_obj.wait_front(Wt);
                cos_dfb_obj.wait_front(Wt);
#endif

                rotated_in_interm_dfb_obj.reserve_back(Wt);
                sin_interm_dfb_obj.reserve_back(Wt);
                cos_interm_dfb_obj.reserve_back(Wt);
                out_dfb_obj.reserve_back(Wt);

                // rotated = x @ trans_mat
                // Matmul uses SrcOrder::Reverse: trans_mat is SrcA and input is SrcB.
                reconfig_data_format(dfb::cos_interm, dfb::trans_mat, dfb::sin_interm, dfb::input);
                pack_reconfig_data_format(dfb::out, dfb::rotated_interm);
                matmul_init(dfb::input, dfb::trans_mat);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(dfb::input, dfb::trans_mat, j, in1_index, j);
                    pack_tile(j, dfb::rotated_interm, j);
                }
                REL();
                rotated_in_interm_dfb_obj.push_back(Wt);
                reconfig_data_format(dfb::trans_mat, dfb::rotated_interm, dfb::input, dfb::sin);
                pack_reconfig_data_format(dfb::rotated_interm, dfb::sin_interm);
                mul_init(dfb::rotated_interm, dfb::sin);
                ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        bulk_block_input(dfb::rotated_interm),
                        sin_cos_input(dfb::sin)>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<bulk_output(dfb::sin_interm)>{});

                reconfig_data_format(dfb::rotated_interm, dfb::input, dfb::sin, dfb::cos);
                pack_reconfig_data_format(dfb::sin_interm, dfb::cos_interm);
                ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb::input,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::AtEnd,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Disabled),
                        sin_cos_input(dfb::cos)>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<bulk_output(dfb::cos_interm)>{});

                reconfig_data_format(dfb::input, dfb::cos_interm, dfb::cos, dfb::sin_interm);
                pack_reconfig_data_format(dfb::cos_interm, dfb::out);
                ckl::add<bulk_block_input(dfb::cos_interm), bulk_block_input(dfb::sin_interm), bulk_output(dfb::out)>(
                    ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt));

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_dfb_obj.pop_front(my_cos_sin_tiles);
            cos_dfb_obj.pop_front(my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from DFB
    trans_mat_dfb_obj.pop_front(onetile);
}
