// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include <type_traits>
#include "api/debug/kernel_profiler.hpp"
#include "../persistent.h"

// The persistent op's sharded router, the reducer core (see dataflow/router_reduce.cpp): the sum of the chip's
// num_parts 1x32 partials (cb_parts, in part order) is its 32 experts' gate logits; score = sqrt(softplus(sum))
// and ranking = score + bias, as router_scores.cpp computes them, packed as bf16 [score, ranking] into cb_rout.
// DEST is fp32 here, so the partials add without a 16-bit rounding per step.
//
// Compile-time args: 0: cb_parts   1: num_parts   2: cb_rbias (the bias row, bias_tiles 1x32 tiles)
//   3: cb_rout   4: bias_tiles   5: my_chip (the tile of this chip's 32 experts' biases)   6: rows (the tiles' height)
//
// FE_PERSISTENT: once per iteration; popping cb_parts stays last (the reader's ready report waits on it).

namespace {
constexpr uint32_t kOneBits = 0x3f800000u;        // softplus beta and 1/beta: 1.0f
constexpr uint32_t kThresholdBits = 0x41a00000u;  // 20.0f, ttnn.softplus's default threshold
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_parts_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_parts = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rbias_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rout_id = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t my_chip = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    // SFPU iterations over a face's rows: 2 cover a 1x32 tile's; taller tiles take the whole face (the rows past
    // the tile's are never packed).
    constexpr int kTinyTileIterations = rows == 1u ? 2 : 8;

    compute_kernel_hw_startup(cb_parts_id, cb_rbias_id, cb_rout_id);

    // score = sqrt(softplus(logit)) in DEST 0, then ranking = score + bias in DEST 1, over `iterations` face rows.
    auto scores = [&](auto iterations) {
        constexpr int kIterations = decltype(iterations)::value;
        softplus_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_softplus,
            (APPROX, DST_ACCUM_MODE, kIterations),
            0,
            VectorMode::R,
            kOneBits,
            kOneBits,
            kThresholdBits));
        sqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sqrt,
            (APPROX, kIterations, DST_ACCUM_MODE, false),
            0,
            VectorMode::R));
        add_binary_tile_init();
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::ADD, kIterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            0,
            1,
            1,
            VectorMode::R)));
    };

    // The tile shape the packer is set up for (FE_NATIVE_ONE_ROW: a one-row iteration runs on the 1x32 aliases).
    bool shape_one_row = false;
    for (uint32_t iter = 0; fe_wait_go(iter); ++iter) {
        fe_rewind_cbs();
        const bool one_row = fe_one_row();
        const uint32_t cb_parts = fe_cb(cb_parts_id, one_row);
        const uint32_t cb_rbias = fe_cb(cb_rbias_id, one_row);
        const uint32_t cb_rout = fe_cb(cb_rout_id, one_row);
        if (one_row != shape_one_row) {
            pack_reconfig_data_format<true>(cb_rout);
            shape_one_row = one_row;
        }
        cb_reserve_back(cb_rout, 2);
        cb_wait_front(cb_rbias, bias_tiles);
        cb_wait_front(cb_parts, num_parts);
        DeviceZoneScopedN("FE_R_REDUCE");
        tile_regs_acquire();
#if defined(FE_NATIVE_ONE_ROW)
        // The parts (SrcA for the copy, SrcB for the adds) and the bias change tile shape with the request's rows.
        reconfig_full_operand(cb_parts, cb_parts);
#endif
        copy_init(cb_parts);
        copy_tile(cb_parts, 0, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts);
        for (uint32_t p = 1; p < num_parts; ++p) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts, p, 0);
        }
#if defined(FE_NATIVE_ONE_ROW)
        reconfig_full_operand_srca(cb_rbias);
#endif
        copy_init(cb_rbias);
        // Several rows: the reader built this chip's bias tile at 0 (every row the same biases); one row's bias row is
        // bias_tiles 1x32 tiles.
        copy_tile(cb_rbias, (rows == 1u || one_row) ? my_chip : 0u, 1);

        if (rows == 1u || one_row) {
            scores(std::integral_constant<int, 2>{});
        } else {
            scores(std::integral_constant<int, kTinyTileIterations>{});
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_rout);
        pack_tile(1, cb_rout);
        tile_regs_release();
        cb_push_back(cb_rout, 2);
        cb_pop_front(cb_rbias, bias_tiles);
        cb_pop_front(cb_parts, num_parts);
    }
}
