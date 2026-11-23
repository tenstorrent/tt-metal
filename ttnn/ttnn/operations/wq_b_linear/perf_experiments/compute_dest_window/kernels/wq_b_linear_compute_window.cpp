// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker compute, DEST-window variant (perf experiment compute_dest_window).
//
// RAW LLK: bypasses compute_kernel_lib::matmul_block. The helper ties the DEST accumulation window to its
// in0_block_k (= one x round = one W block): it waits/pops and spills to L1 once per K-step. Here the x round
// and W block stay block_k_tiles wide (DM unchanged) but DEST accumulates `win` whole K-steps (win*block_k_tiles
// k-tiles) per output subblock before one packer-L1-acc pack. Mechanism exploited: fewer acquire/commit/pack
// round trips (one per window instead of one per K-step), and no interm CB / last-block software reload --
// every window L1-accumulates straight into the (once-reserved) cb_out_block, the first window with acc off.
// Capability gap in the helper: it cannot wait/pop/spill in units decoupled from in0_block_k.
//
// Window schedule (compile-time): num_windows, then the K-step count of each window (sum = num_k_blocks).
// Host guarantees no window wraps cb_w_block (depth % window boundaries) -- the LLK indexes tiles linearly
// from the CB read pointer. x is fully resident (never wraps).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w_block = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_block = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(5);
    constexpr uint32_t sb_h = get_compile_time_arg_val(6);
    constexpr uint32_t sb_w = get_compile_time_arg_val(7);
    constexpr uint32_t bk = get_compile_time_arg_val(8);
    constexpr uint32_t num_windows = get_compile_time_arg_val(11);
    // Window sizes in K-steps: always 12 CT slots (12..23), unused ones 0.
    constexpr uint32_t win_sizes[12] = {
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(19),
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(23)};
    static_assert(num_windows >= 1 && num_windows <= 12);

    constexpr uint32_t bn = in1_num_subblocks * sb_w;  // W block row width (tiles)
    constexpr uint32_t x_round_tiles = in0_num_subblocks * sb_h * bk;
    constexpr uint32_t w_block_tiles = bk * bn;
    constexpr uint32_t sb_tiles = sb_h * sb_w;
    constexpr uint32_t out_block_tiles = sb_tiles * in0_num_subblocks * in1_num_subblocks;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x_resident, x_round_tiles);
        cb_wait_front(cb_w_block, w_block_tiles);
    }
    MaybeDeviceZoneScope("compute_matmul");
    ckernel::matmul_block_init(cb_x_resident, cb_w_block, 0, sb_w, sb_h, bk);
    cb_reserve_back(cb_out_block, out_block_tiles);

    for (uint32_t win = 0; win < num_windows; ++win) {
        const uint32_t n = win_sizes[win];
        const bool last = win == num_windows - 1;
        cb_wait_front(cb_x_resident, n * x_round_tiles);
        cb_wait_front(cb_w_block, n * w_block_tiles);
        uint32_t pushed = 0;
        for (uint32_t sm = 0; sm < in0_num_subblocks; ++sm) {
            for (uint32_t sn = 0; sn < in1_num_subblocks; ++sn) {
                tile_regs_acquire();
                for (uint32_t s = 0; s < n; ++s) {
                    uint32_t in0_index = s * x_round_tiles + sm * sb_h * bk;
                    uint32_t in1_index = s * w_block_tiles + sn * sb_w;
                    for (uint32_t kk = 0; kk < bk; ++kk) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(
                            cb_x_resident, cb_w_block, in0_index, in1_index, 0, false, sb_w, sb_h, bk);
#endif
                        in0_index++;
                        in1_index += bn;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
#ifdef RECONFIG_EACH_SUBBLOCK
                // Attribution probe: the helper's per-subblock packer reconfig (format + L1-acc) every pack.
                PACK((pack_reconfig_data_format(cb_out_block)));
                PACK((llk_pack_reconfig_l1_acc(win == 0 ? 0 : 1)));
#else
                if (sm == 0 && sn == 0) {
                    PACK((llk_pack_reconfig_l1_acc(win == 0 ? 0 : 1)));
                }
#endif
                const uint32_t sb = sm * in1_num_subblocks + sn;
                const uint32_t base = (sb - pushed) * sb_tiles;
                for (uint32_t t = 0; t < sb_tiles; ++t) {
                    pack_tile<true>(t, cb_out_block, base + t);
                }
                tile_regs_release();
                if (last) {
                    cb_push_back(cb_out_block, sb_tiles);
                    pushed++;
                }
            }
        }
#ifdef DRAIN_PROBE
        // Attribution probe: mirror the helper's per-K-block interm spill push + drain wait/pop (no data).
        if (!last) {
            constexpr uint32_t cb_interm_acc = get_compile_time_arg_val(2);
            for (uint32_t s = 0; s < in0_num_subblocks * in1_num_subblocks; ++s) {
                cb_reserve_back(cb_interm_acc, sb_tiles);
                cb_push_back(cb_interm_acc, sb_tiles);
            }
            for (uint32_t s = 0; s < in0_num_subblocks * in1_num_subblocks; ++s) {
                cb_wait_front(cb_interm_acc, sb_tiles);
                cb_pop_front(cb_interm_acc, sb_tiles);
            }
        }
#endif
        cb_pop_front(cb_x_resident, n * x_round_tiles);
        cb_pop_front(cb_w_block, n * w_block_tiles);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}
