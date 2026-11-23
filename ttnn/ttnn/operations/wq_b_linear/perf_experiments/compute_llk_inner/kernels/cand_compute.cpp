// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// compute_llk_inner candidate: raw-LLK matmul K-loop for wq_b_linear (same CB protocol as the op's compute).
//
// Bypasses compute_kernel_lib::matmul_block (matmul_block_helpers.inl). Why (capability, not ergonomics):
//  1. The helper's LastBlockTarget::Out + packer_l1_acc path spills K-steps 0..n-2 into cb_interm_acc and,
//     on the last K-step, RELOADS every subblock from interm into DEST (reconfig_data_format_srca x2 +
//     copy_init + copy_block + a full matmul_block_init per subblock). Here every K-step (the last one too)
//     packs with L1 accumulation straight into cb_out_block, reserved once: no interm, no reload, no
//     per-subblock re-init. LastBlockTarget::Interm cannot express this either (it packs the last K-step
//     into interm, not into the output CB the writer drains per subblock).
//  2. The helper re-issues pack_reconfig_data_format + llk_pack_reconfig_l1_acc on every subblock; here the
//     packer format never changes (out == the pack target from hw_startup) and L1-acc flips once (K-step 1).
//  3. Subblock shapes the helper cannot express: non-uniform M split (SPLIT41 = one 4x2 + one 1x2 subblock).
//
// Knobs (defines): SB_H/SB_W uniform subblock (default: the descriptor's, 5x1 on the focus case); SPLIT41 = 4x2 + 1x2
// subblocks. The output CB keeps the writer's 5x1 SubblockMajor layout: tile (m, n) sits at n*5 + m. With 5x1 subblocks
// each column is pushed as soon as its last K-step pack lands (same as the op); other shapes push both at the end.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef NOMOP
#include "api/compute/experimental/matmul_custom.h"
#define MM_INIT mm_no_mop_init_short
#define MM_BLOCK matmul_block_no_mop
#else
#define MM_INIT matmul_block_init
#define MM_BLOCK matmul_block
#endif
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef SB_H
#define SB_H get_compile_time_arg_val(6)  // descriptor's subblock (5x1 on the focus case)
#endif
#ifndef SB_W
#define SB_W get_compile_time_arg_val(7)
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t block_m = get_compile_time_arg_val(4) * get_compile_time_arg_val(6);  // 5
    constexpr uint32_t block_n = get_compile_time_arg_val(5) * get_compile_time_arg_val(7);  // 2
    constexpr uint32_t kt = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t x_round = block_m * kt;
    constexpr uint32_t w_block = kt * block_n;
    constexpr uint32_t out_tiles = block_m * block_n;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);

    {
        MaybeDeviceZoneScope("compute_wait_x0");
        cb_wait_front(cb_x, x_round);
        cb_wait_front(cb_w, w_block);
    }
    MaybeDeviceZoneScope("compute_matmul");

#ifdef SPLIT41
    // subblock list: (m0, h, n0, w)
    constexpr uint32_t nsb = 2;
    constexpr uint32_t sb_m0[nsb] = {0, 4};
    constexpr uint32_t sb_h[nsb] = {4, 1};
    constexpr uint32_t sb_n0[nsb] = {0, 0};
    constexpr uint32_t sb_w[nsb] = {2, 2};
    constexpr bool push_per_column = false;
#else
    constexpr uint32_t nsb_m = block_m / SB_H, nsb_n = block_n / SB_W;
    constexpr uint32_t nsb = nsb_m * nsb_n;
    uint32_t sb_m0[nsb], sb_h[nsb], sb_n0[nsb], sb_w[nsb];
    for (uint32_t i = 0; i < nsb_m; ++i) {
        for (uint32_t j = 0; j < nsb_n; ++j) {
            // column-major subblock order so a 5x1 split finishes column 0 first
            const uint32_t s = j * nsb_m + i;
            sb_m0[s] = i * SB_H;
            sb_h[s] = SB_H;
            sb_n0[s] = j * SB_W;
            sb_w[s] = SB_W;
        }
    }
    constexpr bool push_per_column = (SB_H == block_m && SB_W == 1);
#endif

#ifdef SPLIT41
    MM_INIT(cb_x, cb_w, 0, sb_w[0], sb_h[0], kt);
#else
    MM_INIT(cb_x, cb_w, 0, SB_W, SB_H, kt);
#endif
    cb_reserve_back(cb_out, out_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));

    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
        const bool last = kb == num_k_blocks - 1;
        if (kb > 0) {
            cb_wait_front(cb_x, x_round);
            cb_wait_front(cb_w, w_block);
        }
        if (kb == 1) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
        uint32_t pushed = 0;
        for (uint32_t s = 0; s < nsb; ++s) {
#ifdef SPLIT41
            if (s == 1) {
                MM_INIT(cb_x, cb_w, 0, sb_w[1], sb_h[1], kt);
            }
#endif
            tile_regs_acquire();
            for (uint32_t kk = 0; kk < kt; ++kk) {
#ifdef DIAG_NO_THROTTLE_CHECK
                // diagnostic only: skips the per-call FW throttle-flag read (not a graduatable change)
                UNPACK((llk_unpack_AB_matmul(
                    cb_x, cb_w, sb_m0[s] * kt + kk, kk * block_n + sb_n0[s], sb_w[s], sb_h[s], kt)));
                MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(0, sb_w[s], sb_h[s])));
#else
                MM_BLOCK(cb_x, cb_w, sb_m0[s] * kt + kk, kk * block_n + sb_n0[s], 0, 0, sb_w[s], sb_h[s], kt);
#endif
            }
            tile_regs_commit();
            tile_regs_wait();
#ifdef DIAG_PER_SB_RECONFIG
            // diagnostic: the helper's per-subblock pack reconfig pair
            PACK((pack_reconfig_data_format(cb_out)));
            PACK((llk_pack_reconfig_l1_acc(kb == 0 ? 0 : 1)));
#endif
            for (uint32_t r = 0; r < sb_h[s]; ++r) {
                for (uint32_t c = 0; c < sb_w[s]; ++c) {
                    const uint32_t pos = (sb_n0[s] + c) * block_m + sb_m0[s] + r;
#ifndef DIAG_SKIP_PACK
                    pack_tile<true>(r * sb_w[s] + c, cb_out, pos - pushed);
#else
                    (void)pos;
#endif
                }
            }
            tile_regs_release();
            if constexpr (push_per_column) {
                if (last) {
                    cb_push_back(cb_out, block_m);
                    pushed += block_m;
                }
            }
        }
#ifdef SPLIT41
        // restore the first subblock's MOP for the next K-step
        if (!last) {
            MM_INIT(cb_x, cb_w, 0, sb_w[0], sb_h[0], kt);
        }
#endif
        cb_pop_front(cb_x, x_round);
        cb_pop_front(cb_w, w_block);
    }
    if constexpr (!push_per_column) {
        cb_push_back(cb_out, block_m);
        cb_push_back(cb_out, block_m);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}
