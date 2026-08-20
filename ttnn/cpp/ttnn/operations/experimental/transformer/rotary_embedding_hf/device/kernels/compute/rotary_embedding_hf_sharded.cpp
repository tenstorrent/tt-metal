// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // Total rows (tiles) owned by this core
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(10);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t half_Wt = Wt / 2;
    (void)Ht;
    constexpr auto bulk_block_input = [](uint32_t cb_id) {
        return ckl::input(
            cb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto held_block_input = [](uint32_t cb_id) {
        return ckl::input(
            cb_id,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto bulk_output = [](uint32_t cb_id) {
        return ckl::output(cb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto rotated_input = bulk_block_input(rotated_in_interm_cb_id);
    constexpr auto in_input = ckl::input(
        in_cb_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::AtEnd,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Disabled);
    constexpr auto sin_input = held_block_input(sin_cb_id);
    constexpr auto cos_input = held_block_input(cos_cb_id);
    constexpr auto sin_interm_input = bulk_block_input(sin_interm_cb_id);
    constexpr auto cos_interm_input = bulk_block_input(cos_interm_cb_id);
    constexpr auto sin_output = bulk_output(sin_interm_cb_id);
    constexpr auto cos_output = bulk_output(cos_interm_cb_id);
    constexpr auto rotary_output = bulk_output(out_cb_id);

    CircularBuffer in_cb(in_cb_id);
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer scalar_cb(scalar_cb_id);
    CircularBuffer rotated_in_interm_cb(rotated_in_interm_cb_id);
    CircularBuffer cos_interm_cb(cos_interm_cb_id);
    CircularBuffer sin_interm_cb(sin_interm_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup(in_cb_id, sin_cb_id, sin_interm_cb_id);  // General Init for all binary ops

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar CB and push it.
    scalar_cb.wait_front(onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the CB.
        sin_cb.reserve_back(Wt);
        cos_cb.reserve_back(Wt);
        sin_cb.push_back(Wt);
        cos_cb.push_back(Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            rotated_in_interm_cb.reserve_back(Wt);
            sin_interm_cb.reserve_back(Wt);
            cos_interm_cb.reserve_back(Wt);
            out_cb.reserve_back(Wt);

            // Get the input
            in_cb.reserve_back(Wt);
            in_cb.push_back(Wt);
            in_cb.wait_front(Wt);

            // Process second half: multiply by -1 and store in rotated buffer
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(half_Wt).block_size(/*block_size=*/half_Wt),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        in_cb_id,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::input(scalar_cb_id, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{
                    half_Wt, 0u},
                // Copy first half to second half of rotated buffer
                ckl::CopyTile<
                    ckl::input(in_cb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                    ckl::Dst::D1>{},
                ckl::PackTile<
                    ckl::output(
                        rotated_in_interm_cb_id,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::Dst::D0>{0u},
                ckl::PackTile<
                    ckl::output(
                        rotated_in_interm_cb_id,
                        ckl::ReservePolicy::None,
                        ckl::PushPolicy::None,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::Dst::D1>{half_Wt});
            rotated_in_interm_cb.push_back(Wt);

            // sin_interim = rotated * sin (broadcast rows)
            mul_bcast_rows_init(rotated_in_interm_cb_id, sin_cb_id);
            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, rotated_input, ckl::input(sin_input, ckl::BroadcastDim::Row)>{},
                ckl::PackTile<sin_output>{});

            ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
                ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, in_input, ckl::input(cos_input, ckl::BroadcastDim::Row)>{},
                ckl::PackTile<cos_output>{});

            // out = cos_interim + sin_interim
            ckl::add<cos_interm_input, sin_interm_input, rotary_output>(
                ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt));
        }

        sin_cb.pop_front(Wt);
        cos_cb.pop_front(Wt);
    }

    // Done with the scalar, so remove from CB
    scalar_cb.pop_front(onetile);
}
