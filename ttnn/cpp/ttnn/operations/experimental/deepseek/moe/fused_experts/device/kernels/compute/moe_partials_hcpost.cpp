// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

// persistent_moe_request with the hyper-connection post step fused (see dataflow/moe_partials_hcpost.cpp), on one
// stream-collapse core: this core's columns of the new streams,
//     new[h] = post[h] * x + sum_g comb[g, h] * old[g],     x = the sum of the op's chips' partials,
// in fp32 DEST. Every operand is this core's columns of one row, viewed as `tiles` 4x32 tiles in the same order,
// and post and comb come as tiles filled with their value, so every step is elementwise (no broadcast). The FPU
// eltwise ops (mul/add_tiles, add_reuse_dest, scalar broadcast) give wrong results on 4x32 tiles with fp32 DEST, so
// every operand is copied into DEST and combined on the SFPU; each phase configures its unpack/copy once.
//   1. While the partials are on their way: mix[h] = sum_g comb[g, h] * old[g] into cb_mix (fp32).
//   2. Once they are in, the critical path: x = their sum, in slot (chip) order, into cb_x (fp32); then
//      new[h] = post[h] * x + mix[h] (every operand fp32) into cb_out, tile h * tiles + t.
// The SFPU steps cover only a 4x32 tile's rows: faces 0 and 1 (VectorMode::R), 4 rows each.
//
// Compile-time args: 0: num_slots   1: tiles (per row)   2: num_streams   3: cb_slots   4: cb_s (old streams)
//   5: cb_post (fp32)   6: cb_comb (comb[g, h] at tile g * num_streams + h)   7: cb_mix   8: cb_x   9: cb_out
// Runtime args: 0: skip (a stop request)

namespace {
constexpr int kTinyIterations = 2;  // a 4x32 tile's 4 rows of a face, 2 per SFPU iteration

// DEST tile odst = DEST tile a + DEST tile b, over a 4x32 tile.
FORCE_INLINE void add_dst(uint32_t a, uint32_t b, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, kTinyIterations, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        odst,
        VectorMode::R)));
}

// DEST tile odst = DEST tile a * DEST tile b, over a 4x32 tile.
FORCE_INLINE void mul_dst(uint32_t a, uint32_t b, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kTinyIterations, DST_ACCUM_MODE),
        a,
        b,
        odst,
        VectorMode::R)));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t streams = get_compile_time_arg_val(2);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(3);
    constexpr uint32_t cb_s = get_compile_time_arg_val(4);
    constexpr uint32_t cb_post = get_compile_time_arg_val(5);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(6);
    constexpr uint32_t cb_mix = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t row_tiles = streams * tiles;

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    compute_kernel_hw_startup(cb_s, cb_comb, cb_mix);

    // 1.
    cb_wait_front(cb_s, row_tiles);
    cb_wait_front(cb_comb, streams * streams);
    cb_reserve_back(cb_mix, row_tiles);
    {
        DeviceZoneScopedN("MOE_HC_MIX");
        reconfig_data_format_srca(cb_s);  // cb_s and cb_comb are both bf16 4x32
        pack_reconfig_data_format(cb_mix);
        copy_init(cb_s);
        mul_binary_tile_init();
        for (uint32_t h = 0; h < streams; ++h) {
            for (uint32_t t = 0; t < tiles; ++t) {
                tile_regs_acquire();
                copy_tile(cb_s, t, 0);
                copy_tile(cb_comb, h, 1);
                mul_dst(0, 1, 0);
                for (uint32_t g = 1; g < streams; ++g) {
                    copy_tile(cb_s, g * tiles + t, 1);
                    copy_tile(cb_comb, g * streams + h, 2);
                    mul_dst(1, 2, 1);
                    add_dst(0, 1, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_mix, h * tiles + t);
                tile_regs_release();
            }
        }
    }
    cb_push_back(cb_mix, row_tiles);

    // 2.
    cb_wait_front(cb_slots, num_slots * tiles);
    DeviceZoneScopedN("MOE_HC_POST");
    cb_reserve_back(cb_x, tiles);
    reconfig_data_format_srca(cb_slots);
    pack_reconfig_data_format(cb_x);
    copy_init(cb_slots);
    add_binary_tile_init();
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_slots, t, 0);
        for (uint32_t m = 1; m < num_slots; ++m) {
            copy_tile(cb_slots, m * tiles + t, 1);
            add_dst(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_x, t);
        tile_regs_release();
    }
    cb_push_back(cb_x, tiles);

    cb_wait_front(cb_x, tiles);
    cb_wait_front(cb_post, streams);
    cb_wait_front(cb_mix, row_tiles);
    cb_reserve_back(cb_out, row_tiles);
    reconfig_data_format_srca(cb_x);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_x);
    mul_binary_tile_init();
    for (uint32_t h = 0; h < streams; ++h) {
        for (uint32_t t = 0; t < tiles; ++t) {
            tile_regs_acquire();
            copy_tile(cb_x, t, 0);
            copy_tile(cb_post, h, 1);
            copy_tile(cb_mix, h * tiles + t, 2);
            mul_dst(0, 1, 0);
            add_dst(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_out, h * tiles + t);
            tile_regs_release();
        }
    }
    cb_push_back(cb_out, row_tiles);
    cb_pop_front(cb_slots, num_slots * tiles);
    cb_pop_front(cb_x, tiles);
    cb_pop_front(cb_mix, row_tiles);
    cb_pop_front(cb_post, streams);
    cb_pop_front(cb_comb, streams * streams);
    cb_pop_front(cb_s, row_tiles);
}
