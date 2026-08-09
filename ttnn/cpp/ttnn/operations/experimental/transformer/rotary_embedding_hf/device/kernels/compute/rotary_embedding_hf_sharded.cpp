// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_dfb_id = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // Total rows (tiles) owned by this core
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(10);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t half_Wt = Wt / 2;
    (void)Ht;
    constexpr auto bulk_block_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto held_block_input = [](uint32_t dfb_id) {
        return ckl::input(
            dfb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::None,
            ckl::OperandKind::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](uint32_t dfb_id) {
        return ckl::output(dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto rotated_input = bulk_block_input(rotated_in_interm_dfb_id);
    constexpr auto in_input = ckl::input(
        in_dfb_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Disabled);
    constexpr auto sin_input = held_block_input(sin_dfb_id);
    constexpr auto cos_input = held_block_input(cos_dfb_id);
    constexpr auto sin_interm_input = bulk_block_input(sin_interm_dfb_id);
    constexpr auto cos_interm_input = bulk_block_input(cos_interm_dfb_id);
    constexpr auto sin_output = bulk_output(sin_interm_dfb_id);
    constexpr auto cos_output = bulk_output(cos_interm_dfb_id);
    constexpr auto rotary_output = bulk_output(out_dfb_id);

    DataflowBuffer dfb_in(in_dfb_id);
    DataflowBuffer dfb_cos(cos_dfb_id);
    DataflowBuffer dfb_sin(sin_dfb_id);
    DataflowBuffer dfb_scalar(scalar_dfb_id);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb_id);
    DataflowBuffer dfb_cos_interm(cos_interm_dfb_id);
    DataflowBuffer dfb_sin_interm(sin_interm_dfb_id);
    DataflowBuffer dfb_out(out_dfb_id);

    compute_kernel_hw_startup(in_dfb_id, sin_dfb_id, sin_interm_dfb_id);

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar DFB and push it.
    dfb_scalar.wait_front(onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the DFB.
        dfb_sin.reserve_back(Wt);
        dfb_cos.reserve_back(Wt);
        dfb_sin.push_back(Wt);
        dfb_cos.push_back(Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            dfb_rotated_in_interm.reserve_back(Wt);
            dfb_sin_interm.reserve_back(Wt);
            dfb_cos_interm.reserve_back(Wt);
            dfb_out.reserve_back(Wt);

            // Get the input
            dfb_in.reserve_back(Wt);
            dfb_in.push_back(Wt);
            dfb_in.wait_front(Wt);

            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(half_Wt, /*block_size=*/half_Wt),
                ckl::BinaryFpu<
                    ckl::input(
                        in_dfb_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Block,
                        ckl::TileOffset::Set),
                    ckl::input(scalar_dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Scalar>{half_Wt, 0u},
                ckl::CopyTile<
                    ckl::input(in_dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block),
                    ckl::Dst::D1>{},
                ckl::PackTile<
                    ckl::output(
                        rotated_in_interm_dfb_id,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::TileOffset::Set),
                    ckl::Dst::D0>{0u},
                ckl::PackTile<
                    ckl::output(
                        rotated_in_interm_dfb_id,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::TileOffset::Set),
                    ckl::Dst::D1>{half_Wt});
            dfb_rotated_in_interm.push_back(Wt);

            mul_bcast_rows_init(rotated_in_interm_dfb_id, sin_dfb_id);
            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                ckl::BinaryFpu<rotated_input, sin_input, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                ckl::PackTile<sin_output>{});

            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                ckl::BinaryFpu<in_input, cos_input, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                ckl::PackTile<cos_output>{});

            ckl::add<cos_interm_input, sin_interm_input, rotary_output, ckl::BroadcastDim::None>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
        }

        dfb_sin.pop_front(Wt);
        dfb_cos.pop_front(Wt);
    }

    // Done with the scalar, so remove from DFB
    dfb_scalar.pop_front(onetile);
}
