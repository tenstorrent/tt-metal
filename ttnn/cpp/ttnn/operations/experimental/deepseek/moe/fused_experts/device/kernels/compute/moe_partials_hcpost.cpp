// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

// persistent_moe_request with the hyper-connection post step fused (see dataflow/moe_partials_hcpost.cpp), on one
// stream-collapse core: this core's columns of the new streams,
//     new[h] = post[h] * x + sum_g comb[g, h] * old[g],     x = the sum of the op's chips' partials.
// Every row part (a slot, an old or a new stream row) is ONE 32x32 bf16 tile: its CB's pages are part_bytes apart and
// unpack/pack address tile i at i * page size, so a part's elements are the tile's first ones and the rest are
// don't-cares. Every step is elementwise or a scalar broadcast, so the don't-cares never reach a part's elements --
// tt-blaze's BcastMulAdd views its [1, 32] rows as [T, 32] tiles the same way -- and the FPU runs on whole tiles
// (its eltwise ops are wrong on the 4x32 tiles a part would otherwise take). post and comb are scalars at element 0
// of their pages. New rows are packed in order, each over the previous one's don't-care tail.
//   1. While the partials are on their way: mix[h] = sum_g comb[g, h] * old[g] into cb_mix (products through
//      cb_prod, then summed in DEST).
//   2. Once they are in, the critical path: x = their sum in slot (chip) order (dest-reuse adds, as
//      moe_partials_sum.cpp) into cb_x; then new[h] = post[h] * x + mix[h] into cb_out tile h.
//
// Compile-time args: 0: num_slots   1: num_streams   2: cb_slots   3: cb_s (old streams)   4: cb_post
//   5: cb_comb (comb[g, h] at page g * num_streams + h)   6: cb_mix   7: cb_x   8: cb_out   9: cb_prod
// Runtime args: 0: skip (a stop request)

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t streams = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_post = get_compile_time_arg_val(4);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mix = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_prod = get_compile_time_arg_val(9);
    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    // Every CB is bf16 32x32: no data format reconfiguration anywhere.
    compute_kernel_hw_startup(cb_s, cb_comb, cb_prod);

    // 1.
    cb_wait_front(cb_s, streams);
    cb_wait_front(cb_comb, streams * streams);
    cb_reserve_back(cb_mix, streams);
    {
        DeviceZoneScopedN("MOE_HC_MIX");
        for (uint32_t h = 0; h < streams; ++h) {
            cb_reserve_back(cb_prod, streams);
            tile_regs_acquire();
            mul_bcast_scalar_init(cb_s, cb_comb);
            for (uint32_t g = 0; g < streams; ++g) {
                mul_tiles_bcast_scalar(cb_s, cb_comb, g, g * streams + h, g);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t g = 0; g < streams; ++g) {
                pack_tile<true>(g, cb_prod, g);
            }
            tile_regs_release();
            cb_push_back(cb_prod, streams);

            cb_wait_front(cb_prod, streams);
            tile_regs_acquire();
            copy_init(cb_prod);
            copy_tile(cb_prod, 0, 0);
            add_reuse_dest_init<kReuse>(cb_prod);
            for (uint32_t g = 1; g < streams; ++g) {
                add_reuse_dest_tiles<kReuse>(cb_prod, g, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_mix, h);
            tile_regs_release();
            cb_pop_front(cb_prod, streams);
        }
    }
    cb_push_back(cb_mix, streams);

    // 2.
    cb_wait_front(cb_slots, num_slots);
    DeviceZoneScopedN("MOE_HC_POST");
    cb_reserve_back(cb_x, 1);
    tile_regs_acquire();
    copy_init(cb_slots);
    copy_tile(cb_slots, 0, 0);
    add_reuse_dest_init<kReuse>(cb_slots);
    for (uint32_t m = 1; m < num_slots; ++m) {
        add_reuse_dest_tiles<kReuse>(cb_slots, m, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_x);
    tile_regs_release();
    cb_push_back(cb_x, 1);

    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_post, streams);
    cb_wait_front(cb_mix, streams);
    cb_reserve_back(cb_out, streams);
    tile_regs_acquire();
    mul_bcast_scalar_init(cb_x, cb_post);
    for (uint32_t h = 0; h < streams; ++h) {
        mul_tiles_bcast_scalar(cb_x, cb_post, 0, h, h);
    }
    add_reuse_dest_init<kReuse>(cb_mix);
    for (uint32_t h = 0; h < streams; ++h) {
        add_reuse_dest_tiles<kReuse>(cb_mix, h, h);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h = 0; h < streams; ++h) {
        pack_tile<true>(h, cb_out, h);
    }
    tile_regs_release();
    cb_push_back(cb_out, streams);
    cb_pop_front(cb_slots, num_slots);
    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_mix, streams);
    cb_pop_front(cb_post, streams);
    cb_pop_front(cb_comb, streams * streams);
    cb_pop_front(cb_s, streams);
}
