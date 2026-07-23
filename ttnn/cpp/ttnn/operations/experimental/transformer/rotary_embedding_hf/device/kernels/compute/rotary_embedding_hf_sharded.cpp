// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
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
        return ckl::output(cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled);
    };
    constexpr auto rotated_input = bulk_block_input(rotated_in_interm_cb);
    constexpr auto in_input = bulk_block_input(in_cb);
    constexpr auto sin_input = held_block_input(sin_cb);
    constexpr auto cos_input = held_block_input(cos_cb);
    constexpr auto sin_interm_input = bulk_block_input(sin_interm_cb);
    constexpr auto cos_interm_input = bulk_block_input(cos_interm_cb);
    constexpr auto sin_output = bulk_output(sin_interm_cb);
    constexpr auto cos_output = bulk_output(cos_interm_cb);
    constexpr auto rotary_output = bulk_output(out_cb);

    binary_op_init_common(in_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    // Wait for the reader kernel (reader_rotary_embedding_hf_sharded.cpp) to
    // write -1.0 into the scalar CB and push it.
    cb_wait_front(scalar_cb, onetile);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        // For decode mode, cos/sin are [1, batch, 1, head_dim] and this core's shard
        // may contain multiple batch rows. Push one row at a time and advance the CB.
        cb_reserve_back(sin_cb, Wt);
        cb_reserve_back(cos_cb, Wt);
        cb_push_back(sin_cb, Wt);
        cb_push_back(cos_cb, Wt);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            cb_reserve_back(rotated_in_interm_cb, Wt);
            cb_reserve_back(sin_interm_cb, Wt);
            cb_reserve_back(cos_interm_cb, Wt);
            cb_reserve_back(out_cb, Wt);

            // Get the input
            cb_reserve_back(in_cb, Wt);
            cb_push_back(in_cb, Wt);
            cb_wait_front(in_cb, Wt);

            // Process second half: multiply by -1 and store in rotated buffer
            mul_tiles_bcast_scalar_init_short(in_cb, scalar_cb);
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                mul_tiles_bcast_scalar(in_cb, scalar_cb, j + half_Wt, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j, rotated_in_interm_cb, j);
            }
            tile_regs_release();

            // Copy first half to second half of rotated buffer
            tile_regs_acquire();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                copy_tile_init_with_dt(in_cb);
                copy_tile(in_cb, j, j + half_Wt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < half_Wt; ++j) {
                pack_tile(j + half_Wt, rotated_in_interm_cb, j + half_Wt);
            }
            tile_regs_release();

            cb_push_back(rotated_in_interm_cb, Wt);

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

        cb_pop_front(sin_cb, Wt);
        cb_pop_front(cos_cb, Wt);
    }

    // Done with the scalar, so remove from CB
    cb_pop_front(scalar_cb, onetile);
}
