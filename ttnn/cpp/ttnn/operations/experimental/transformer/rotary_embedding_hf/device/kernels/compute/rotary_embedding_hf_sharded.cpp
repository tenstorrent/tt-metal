// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // Total rows (tiles) owned by this core
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(10);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t half_Wt = Wt / 2;
    (void)Ht;
    constexpr auto bulk_block_input = [](uint32_t cb) {
        return ckl::input(cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto held_block_input = [](uint32_t cb) {
        return ckl::input(
            cb, ckl::InputLifecycle::HeldBulk, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::OutputLifecycle::ReserveNonePushEnd, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto rotated_input = bulk_block_input(rotated_in_interm_cb);
    constexpr auto in_input =
        ckl::input(in_cb, ckl::InputLifecycle::DeferredPop, ckl::OperandKind::Block, ckl::DataFormatReconfig::Disabled);
    constexpr auto sin_input = held_block_input(sin_cb);
    constexpr auto cos_input = held_block_input(cos_cb);
    constexpr auto sin_interm_input = bulk_block_input(sin_interm_cb);
    constexpr auto cos_interm_input = bulk_block_input(cos_interm_cb);
    constexpr auto sin_output = bulk_output(sin_interm_cb);
    constexpr auto cos_output = bulk_output(cos_interm_cb);
    constexpr auto rotary_output = bulk_output(out_cb);

    CircularBuffer in_cb_obj(in_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer scalar_cb_obj(scalar_cb);
    CircularBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    compute_kernel_hw_startup(in_cb, sin_cb, sin_interm_cb);

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar CB and push it.
    scalar_cb_obj.wait_front(onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the CB.
        sin_cb_obj.reserve_back(Wt);
        cos_cb_obj.reserve_back(Wt);
        sin_cb_obj.push_back(Wt);
        cos_cb_obj.push_back(Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            rotated_in_interm_cb_obj.reserve_back(Wt);
            sin_interm_cb_obj.reserve_back(Wt);
            cos_interm_cb_obj.reserve_back(Wt);
            out_cb_obj.reserve_back(Wt);

            // Get the input
            in_cb_obj.reserve_back(Wt);
            in_cb_obj.push_back(Wt);
            in_cb_obj.wait_front(Wt);

            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(half_Wt, /*block_size=*/half_Wt),
                ckl::BinaryFpu<
                    ckl::input(
                        in_cb, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block, ckl::TileOffset::Set),
                    ckl::input(scalar_cb, ckl::InputLifecycle::CallerManaged),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Scalar>{half_Wt, 0u},
                ckl::CopyTile<
                    ckl::input(in_cb, ckl::InputLifecycle::CallerManaged, ckl::OperandKind::Block),
                    ckl::Dst::D1>{},
                ckl::PackTile<
                    ckl::output(rotated_in_interm_cb, ckl::OutputLifecycle::CallerManaged, ckl::TileOffset::Set),
                    ckl::Dst::D0>{0u},
                ckl::PackTile<
                    ckl::output(rotated_in_interm_cb, ckl::OutputLifecycle::CallerManaged, ckl::TileOffset::Set),
                    ckl::Dst::D1>{half_Wt});
            rotated_in_interm_cb_obj.push_back(Wt);

            mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
            ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                ckl::BinaryFpu<rotated_input, sin_input, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                ckl::PackTile<sin_output>{});

            ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                ckl::BinaryFpu<in_input, cos_input, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                ckl::PackTile<cos_output>{});

            ckl::add<cos_interm_input, sin_interm_input, rotary_output, ckl::BroadcastDim::None>(
                ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
        }

        sin_cb_obj.pop_front(Wt);
        cos_cb_obj.pop_front(Wt);
    }

    // Done with the scalar, so remove from CB
    scalar_cb_obj.pop_front(onetile);
}
