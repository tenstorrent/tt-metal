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
#include "api/debug/kernel_profiler.hpp"
#include "../persistent.h"

// The persistent op's sharded router, the reducer core (see dataflow/router_reduce.cpp): the sum of the chip's
// num_parts 1x32 partials (cb_parts, in part order) is its 32 experts' gate logits; score = sqrt(softplus(sum))
// and ranking = score + bias, as router_scores.cpp computes them, packed as bf16 [score, ranking] into cb_rout.
// DEST is fp32 here, so the partials add without a 16-bit rounding per step.
//
// Compile-time args: 0: cb_parts   1: num_parts   2: cb_rbias (the bias row, bias_tiles 1x32 tiles)
//   3: cb_rout   4: bias_tiles   5: my_chip (the tile of this chip's 32 experts' biases)
//
// FE_PERSISTENT: once per iteration; popping cb_parts stays last (the reader's ready report waits on it).

namespace {
constexpr uint32_t kOneBits = 0x3f800000u;        // softplus beta and 1/beta: 1.0f
constexpr uint32_t kThresholdBits = 0x41a00000u;  // 20.0f, ttnn.softplus's default threshold
constexpr int kTinyTileIterations = 2;            // a 1x32 tile's rows of a face
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_parts_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_parts = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rbias_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rout_id = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t my_chip = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(cb_parts_id, cb_rbias_id, cb_rout_id);

    for (uint32_t iter = 0; fe_wait_go(iter); ++iter) {
        fe_rewind_cbs();
        cb_reserve_back(cb_rout_id, 2);
        cb_wait_front(cb_rbias_id, bias_tiles);
        cb_wait_front(cb_parts_id, num_parts);
        DeviceZoneScopedN("FE_R_REDUCE");
        tile_regs_acquire();
        copy_init(cb_parts_id);
        copy_tile(cb_parts_id, 0, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts_id);
        for (uint32_t p = 1; p < num_parts; ++p) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_parts_id, p, 0);
        }
        copy_init(cb_rbias_id);
        copy_tile(cb_rbias_id, my_chip, 1);

        softplus_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_softplus,
            (APPROX, DST_ACCUM_MODE, kTinyTileIterations),
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
            (APPROX, kTinyTileIterations, DST_ACCUM_MODE, false),
            0,
            VectorMode::R));
        add_binary_tile_init();
        MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::ADD, kTinyTileIterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            0,
            1,
            1,
            VectorMode::R)));
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_rout_id);
        pack_tile(1, cb_rout_id);
        tile_regs_release();
        cb_push_back(cb_rout_id, 2);
        cb_pop_front(cb_rbias_id, bias_tiles);
        cb_pop_front(cb_parts_id, num_parts);
    }
}
