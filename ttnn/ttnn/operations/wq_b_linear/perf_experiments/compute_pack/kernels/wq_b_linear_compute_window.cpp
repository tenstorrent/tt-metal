// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear worker compute, DEST-window variant (perf experiment compute_pack, Perf 2).
//
// RAW LLK: bypasses compute_kernel_lib::matmul_block. Capability gap: the helper ties its DEST accumulation
// window (one acquire -> pack per subblock) to in0_block_k, which is also its wait/pop unit, i.e. one x round /
// one W block. Here x rounds and W blocks stay block_k_tiles wide (DM unchanged, round 0 lands early), but DEST
// accumulates `n` whole K-steps (n*block_k_tiles k-tiles, host keeps it <= 16 for the PCC gate) per output
// subblock before ONE packer-L1-acc pack into cb_out_block (reserved once, pushed once at the end; TileRowMajor
// layout, same as the helper). Mechanism: n x fewer acquire/commit/pack round trips; the packer's L1
// read-modify-write per K-step (~pack-only 10.7 us at n=1) drops by n, so math (the LLK floor) binds alone.
// Packer L1-acc flips once (window 1); format never changes (pack target = hw_startup's out CB).
//
// CT args: 0..8 as the op's compute (cb_x, cb_w, cb_out, in0_nsb, in1_nsb, sb_h, sb_w, bk, num_k_blocks),
// 9 = num_windows, 10..33 = K-steps per window (24 slots, unused 0). Host guarantees no window wraps cb_w_block
// (the LLK indexes tiles linearly from the CB read pointer); x is fully resident (never wraps).

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t in0_nsb = get_compile_time_arg_val(3);
    constexpr uint32_t in1_nsb = get_compile_time_arg_val(4);
    constexpr uint32_t sb_h = get_compile_time_arg_val(5);
    constexpr uint32_t sb_w = get_compile_time_arg_val(6);
    constexpr uint32_t bk = get_compile_time_arg_val(7);
    constexpr uint32_t num_windows = get_compile_time_arg_val(9);
    constexpr uint32_t win_sizes[24] = {
        get_compile_time_arg_val(10), get_compile_time_arg_val(11), get_compile_time_arg_val(12),
        get_compile_time_arg_val(13), get_compile_time_arg_val(14), get_compile_time_arg_val(15),
        get_compile_time_arg_val(16), get_compile_time_arg_val(17), get_compile_time_arg_val(18),
        get_compile_time_arg_val(19), get_compile_time_arg_val(20), get_compile_time_arg_val(21),
        get_compile_time_arg_val(22), get_compile_time_arg_val(23), get_compile_time_arg_val(24),
        get_compile_time_arg_val(25), get_compile_time_arg_val(26), get_compile_time_arg_val(27),
        get_compile_time_arg_val(28), get_compile_time_arg_val(29), get_compile_time_arg_val(30),
        get_compile_time_arg_val(31), get_compile_time_arg_val(32), get_compile_time_arg_val(33)};
    static_assert(num_windows >= 1 && num_windows <= 24);

    constexpr uint32_t bm = in0_nsb * sb_h;
    constexpr uint32_t bn = in1_nsb * sb_w;
    constexpr uint32_t x_round_tiles = bm * bk;
    constexpr uint32_t w_block_tiles = bk * bn;
    constexpr uint32_t out_block_tiles = bm * bn;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x, x_round_tiles);
        cb_wait_front(cb_w, w_block_tiles);
    }
    MaybeDeviceZoneScope("compute_matmul");
    ckernel::matmul_block_init(cb_x, cb_w, 0, sb_w, sb_h, bk);
    cb_reserve_back(cb_out, out_block_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));

    for (uint32_t win = 0; win < num_windows; ++win) {
        const uint32_t n = win_sizes[win];
        if (win == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        cb_wait_front(cb_x, n * x_round_tiles);
        cb_wait_front(cb_w, n * w_block_tiles);
#ifdef LAST_ROWS
        // Tail variant: the last window (1 K-step) runs as bm row subblocks of 1 x bn, so the pack left
        // after the final math is one row (bn tiles) instead of a whole sb_h x sb_w subblock.
        if (win == num_windows - 1 && n == 1) {
            ckernel::matmul_block_init(cb_x, cb_w, 0, bn, 1, bk);
            for (uint32_t m = 0; m < bm; ++m) {
                tile_regs_acquire();
                for (uint32_t kk = 0; kk < bk; ++kk) {
#ifndef SKIP_COMPUTE
                    ckernel::matmul_block(cb_x, cb_w, m * bk + kk, kk * bn, 0, false, bn, 1, bk);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t c = 0; c < bn; ++c) {
                    pack_tile<true>(c, cb_out, m * bn + c);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_x, x_round_tiles);
            cb_pop_front(cb_w, w_block_tiles);
            continue;
        }
#endif
        for (uint32_t sm = 0; sm < in0_nsb; ++sm) {
            for (uint32_t sn = 0; sn < in1_nsb; ++sn) {
                tile_regs_acquire();
                for (uint32_t s = 0; s < n; ++s) {
                    uint32_t in0_index = s * x_round_tiles + sm * sb_h * bk;
                    uint32_t in1_index = s * w_block_tiles + sn * sb_w;
                    for (uint32_t kk = 0; kk < bk; ++kk) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(cb_x, cb_w, in0_index, in1_index, 0, false, sb_w, sb_h, bk);
#endif
                        in0_index++;
                        in1_index += bn;
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                // TileRowMajor: tile (m, n) at m * bn + n, offset from the (once-reserved) write pointer.
                for (uint32_t r = 0; r < sb_h; ++r) {
                    for (uint32_t c = 0; c < sb_w; ++c) {
                        pack_tile<true>(r * sb_w + c, cb_out, (sm * sb_h + r) * bn + sn * sb_w + c);
                    }
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_x, n * x_round_tiles);
        cb_pop_front(cb_w, n * w_block_tiles);
    }
    cb_push_back(cb_out, out_block_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));
}
