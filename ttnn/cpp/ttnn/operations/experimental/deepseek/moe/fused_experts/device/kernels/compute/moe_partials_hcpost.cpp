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
// in fp32 DEST. Every operand is this core's columns of one row, viewed as `tiles` whole tiles in the same order,
// and post and comb come as tiles filled with their value, so every step is an elementwise copy and SFPU op.
//   1. While the partials are on their way: mix[h] = sum_g comb[g, h] * old[g] into cb_mix (fp32).
//   2. Once they are in: x = their sum, in slot (chip) order, into cb_x (fp32); then new[h] = post[h] * x + mix[h]
//      into cb_out, tile h * tiles + t.
//
// Compile-time args: 0: num_slots   1: tiles (per row)   2: num_streams   3: cb_slots   4: cb_s (old streams)
//   5: cb_post   6: cb_comb (comb[g, h] at tile g * num_streams + h)   7: cb_mix   8: cb_x   9: cb_out
// Runtime args: 0: skip (a stop request)

namespace {
FORCE_INLINE void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void add_into(uint32_t dst, uint32_t other) {
    add_binary_tile_init();
    add_binary_tile(dst, other, dst);
}

FORCE_INLINE void mul_into(uint32_t dst, uint32_t other) {
    mul_binary_tile_init();
    mul_binary_tile(dst, other, dst);
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
        pack_reconfig_data_format(cb_mix);
        for (uint32_t h = 0; h < streams; ++h) {
            for (uint32_t t = 0; t < tiles; ++t) {
                tile_regs_acquire();
                copy_from(cb_s, t, 0);
                copy_from(cb_comb, h, 1);
                mul_into(0, 1);
                for (uint32_t g = 1; g < streams; ++g) {
                    copy_from(cb_s, g * tiles + t, 1);
                    copy_from(cb_comb, g * streams + h, 2);
                    mul_into(1, 2);
                    add_into(0, 1);
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
    pack_reconfig_data_format(cb_x);
    for (uint32_t t = 0; t < tiles; ++t) {
        tile_regs_acquire();
        copy_from(cb_slots, t, 0);
        for (uint32_t m = 1; m < num_slots; ++m) {
            copy_from(cb_slots, m * tiles + t, 1);
            add_into(0, 1);
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
    pack_reconfig_data_format(cb_out);
    for (uint32_t h = 0; h < streams; ++h) {
        for (uint32_t t = 0; t < tiles; ++t) {
            tile_regs_acquire();
            copy_from(cb_x, t, 0);
            copy_from(cb_post, h, 1);
            mul_into(0, 1);
            copy_from(cb_mix, h * tiles + t, 1);
            add_into(0, 1);
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
