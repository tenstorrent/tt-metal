// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: down projection on its own cores (TRISC). Per virtual expert v (a 128-row sub-block of
// expert e = v / S): y[rows, PCD columns] = h_all(v) @ Wd[:, columns], accumulated in DST over all KT_D K-tiles, one
// row tile at a time (PCD <= 8 accumulators). h_all holds both M-groups' h K-tile-major: [G][KT_D][MTG rows], so a
// gate/up core's slice (its NP K-tiles, all MTG rows) is one contiguous run.
// The expert's down weights stay resident in the in1 ring (RING blocks of [KBLK_D x PCD], filled by se6_dw.cpp) for
// its S sub-blocks and are popped after the last one.
//
// CT: 0 MTG (row tiles per M-group), 1 G, 2 KT_D, 3 KBLK_D, 4 PCD, 5 NUM_EXPERTS, 6 S, 7 SLOT_TILES, 8 RING
#include "se_meta.hpp"
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#ifdef SE_Y_RM
#include "api/compute/pack_untilize.h"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#endif

constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
constexpr uint32_t out_cb = tt::CBIndex::c_16;

constexpr uint32_t mtg = get_compile_time_arg_val(0);
constexpr uint32_t groups = get_compile_time_arg_val(1);
constexpr uint32_t kt_d = get_compile_time_arg_val(2);
constexpr uint32_t kblk_d = get_compile_time_arg_val(3);
constexpr uint32_t pcd = get_compile_time_arg_val(4);
constexpr uint32_t num_experts = get_compile_time_arg_val(5);
constexpr uint32_t sub = get_compile_time_arg_val(6);
constexpr uint32_t slot = get_compile_time_arg_val(7);
constexpr uint32_t ring = get_compile_time_arg_val(8);
constexpr uint32_t nblk = kt_d / kblk_d;
constexpr uint32_t hk = 8;
constexpr uint32_t group_tiles = kt_d * mtg;
constexpr uint32_t h_all_tiles = groups * group_tiles;
constexpr uint32_t mt = groups * mtg;
// PCD > DST output columns (small-I subgrids have few down cores): column passes of <= 8 bf16 / 4 fp32 (SE_DN_FP32:
// fp32 DEST accumulation over K, the host sets fp32_dest_acc_en) DST tiles, each re-running the row's K loop over the
// same h and resident weights
#if defined(SE_DN_FP32) && defined(SE_DN_FULL_SYNC)
constexpr uint32_t dst_cols = 8;  // full-sync DST: 16 bf16 / 8 fp32 tiles (math and pack no longer overlap)
#elif defined(SE_DN_FP32)
constexpr uint32_t dst_cols = 4;
#else
constexpr uint32_t dst_cols = 8;
#endif
#ifdef SE_Y_RM
// pack_untilize places column block c at c * block width, so every pass has the same width: the largest divisor of
// PCD that fits DST
constexpr uint32_t largest_div(uint32_t n, uint32_t cap) {
    for (uint32_t d = cap < n ? cap : n; d > 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}
constexpr uint32_t cpw = largest_div(pcd, dst_cols);
#else
constexpr uint32_t cpw = pcd < dst_cols ? pcd : dst_cols;
#endif
static_assert(pcd <= 16 && kt_d % kblk_d == 0 && kt_d % hk == 0 && ring >= nblk);
#ifdef SE_Y_RM
// Row-major y: each row tile is pack-untilized into PCD pages (32 rows x PCD * 32 bf16) and pushed on its own
static_assert(pcd % cpw == 0);
#endif

uint32_t popped = 0;

#ifdef SE_SMALL_T
// Small-M role split (with SE_DYN): when every active expert is small the reader tails drop their down columns and
// cores with an extra slice (RT 0) also compute PCX more columns, a second pass per row (weights CB 4, out CB 17).
// CT: 9 PCX, 10 KBLK_X, 11 SLOT_X, 12 RING_X.
constexpr uint32_t x_cb = tt::CBIndex::c_4;
constexpr uint32_t xo_cb = tt::CBIndex::c_17;
constexpr uint32_t pcx = get_compile_time_arg_val(9);
constexpr uint32_t kblk_x = get_compile_time_arg_val(10);
constexpr uint32_t slot_x = get_compile_time_arg_val(11);
constexpr uint32_t ring_x = get_compile_time_arg_val(12);
constexpr uint32_t nblk_x = kt_d / kblk_x;
uint32_t popped_x = 0;

FORCE_INLINE uint32_t wblock_x(uint32_t a) {
    cb_wait_front(x_cb, (a - popped_x + 1) * slot_x);
    return static_cast<uint32_t>(static_cast<int32_t>(a % ring_x) - static_cast<int32_t>(popped_x % ring_x)) * slot_x;
}
#endif

#ifdef SE_DN_REG
// Pinned schedule (se_dyn.hpp): load l's block j is stream block l * NBLK + j in ring slot region(l) * NBLK + j,
// popped by count after its last use (as se3_compute.cpp's gate/up ring).
FORCE_INLINE uint32_t wblock_dyn(uint32_t p, uint32_t phys) {
    if (p + 1 > popped) {
        cb_wait_front(in1_cb, (p + 1 - popped) * slot);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(phys) - static_cast<int32_t>(popped % ring)) * slot;
}
#endif

FORCE_INLINE uint32_t wblock(uint32_t a) {
    cb_wait_front(in1_cb, (a - popped + 1) * slot);
    return static_cast<uint32_t>(static_cast<int32_t>(a % ring) - static_cast<int32_t>(popped % ring)) * slot;
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(h_all_cb, in1_cb, out_cb);
    matmul_block_init(h_all_cb, in1_cb, false, cpw, 1, hk);
#ifdef SE_Y_RM
    pack_untilize_dest_init<cpw, pcd>(out_cb);
#endif
#ifdef SE_DYN
    // Dynamic counts: CB 6 holds [n_act, num_v, subs of each active expert] (from the down core's data movement); the
    // in1 ring holds only the active experts, so an active expert's index is its place in the stream.
    cb_wait_front(tt::CBIndex::c_6, 1);
    const uint32_t n_act = read_tile_value(tt::CBIndex::c_6, 0, 0);
    const uint32_t num_v = read_tile_value(tt::CBIndex::c_6, 0, 1);
    uint32_t e = 0, s = 0, subs_e = n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS) : 0;
#ifdef SE_SMALL_T
    const bool xs = read_tile_value(tt::CBIndex::c_6, 0, SE_META_SMALL) != 0 && get_arg_val<uint32_t>(0) != 0;
#endif
#ifdef SE_DN_REG
    static_assert(ring == SE_GU_NREG * nblk);
    uint32_t gw = n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_GU) : 0;
#endif
    uint32_t lmt = n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_LMT) : mt;
    for (uint32_t v = 0; v < num_v; ++v) {
        const uint32_t rows = s + 1 == subs_e ? lmt : mt;  // row tiles holding tokens (the rest: never written out)
#ifdef SE_DN_REG
        const bool last_sub = s + 1 == subs_e && ((gw >> 16) & 1);  // the load's last use: its weights go
        const uint32_t ld = gw & 0xFF, ph0 = ((gw >> 8) & 0xFF) * nblk;
#else
        const bool last_sub = s + 1 == subs_e;
#endif
#else
    for (uint32_t v = 0; v < num_experts * sub; ++v) {
        const uint32_t e = v / sub;
        const bool last_sub = v % sub == sub - 1;
        constexpr uint32_t rows = mt;
#endif
        {
#ifdef SE_ZONES
            DeviceZoneScopedN("SE_H_WAIT");
#endif
            cb_wait_front(h_all_cb, h_all_tiles);
        }
#ifdef SE_ZONES
        DeviceZoneScopedN("SE_DOWN");
#endif
#ifndef SE_Y_RM
        cb_reserve_back(out_cb, mt * pcd);
#endif
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t h0 = (r / mtg) * group_tiles + r % mtg;
#ifdef SE_Y_RM
            cb_reserve_back(out_cb, pcd);
#endif
            for (uint32_t c0 = 0; c0 < pcd; c0 += cpw) {
                const uint32_t cw = pcd - c0 < cpw ? pcd - c0 : cpw;
                const bool final_pass = c0 + cw == pcd;
                if constexpr (pcd > cpw) {
                    matmul_block_init(h_all_cb, in1_cb, false, cw, 1, hk);
                }
                tile_regs_acquire();
                uint32_t w = 0;
                for (uint32_t kk = 0; kk < kt_d; ++kk) {
                    if (kk % kblk_d == 0) {
#ifdef SE_DN_REG
                        w = wblock_dyn(ld * nblk + kk / kblk_d, ph0 + kk / kblk_d);
#else
                        w = wblock(e * nblk + kk / kblk_d);
#endif
                    }
                    matmul_block(h_all_cb, in1_cb, h0 + kk * mtg, w + (kk % kblk_d) * pcd + c0, 0, false, cw, 1, hk);
#ifdef SE_EARLY_POP
                    // The expert's last row (final column pass): each weight block goes as soon as it is used, so
                    // the next expert's blocks stream into the ring while this row still runs.
                    if (last_sub && final_pass && r == rows - 1 && kk % kblk_d == kblk_d - 1) {
                        cb_pop_front(in1_cb, slot);
                        ++popped;
                    }
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
#ifdef SE_Y_RM
                pack_untilize_dest<cpw, pcd>(out_cb, 1, c0 / cpw);
#else
                for (uint32_t i = 0; i < cw; ++i) {
                    pack_tile<true>(i, out_cb, r * pcd + c0 + i);  // row-major [MT x PCD]
                }
#endif
                tile_regs_release();
            }
#ifdef SE_Y_RM
            cb_push_back(out_cb, pcd);
#endif
        }
#ifdef SE_SMALL_T
        if (xs) {
            matmul_block_init(h_all_cb, x_cb, false, pcx, 1, hk);
            cb_reserve_back(xo_cb, mt * pcx);
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t h0 = (r / mtg) * group_tiles + r % mtg;
                tile_regs_acquire();
                uint32_t w = 0;
                for (uint32_t kk = 0; kk < kt_d; ++kk) {
                    if (kk % kblk_x == 0) {
                        w = wblock_x(e * nblk_x + kk / kblk_x);
                    }
                    matmul_block(h_all_cb, x_cb, h0 + kk * mtg, w + (kk % kblk_x) * pcx, 0, false, pcx, 1, hk);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < pcx; ++i) {
                    pack_tile<true>(i, xo_cb, r * pcx + i);
                }
                tile_regs_release();
            }
            cb_push_back(xo_cb, mt * pcx);
            if (last_sub) {
                cb_pop_front(x_cb, nblk_x * slot_x);
                popped_x += nblk_x;
            }
            matmul_block_init(h_all_cb, in1_cb, false, cpw, 1, hk);
        }
#endif
        cb_pop_front(h_all_cb, h_all_tiles);
#ifndef SE_Y_RM
        cb_push_back(out_cb, mt * pcd);
#endif
#ifndef SE_EARLY_POP
        if (last_sub) {  // the expert's last sub-block: its weights can go
            cb_pop_front(in1_cb, nblk * slot);
            popped += nblk;
        }
#endif
#ifdef SE_DYN
        if (++s == subs_e) {
            s = 0;
            ++e;
            subs_e = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS + e) : 0;
            lmt = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_LMT + e) : mt;
#ifdef SE_DN_REG
            gw = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_GU + e) : 0;
#endif
        }
#endif
    }
#ifdef SE_DYN
    cb_pop_front(tt::CBIndex::c_6, 1);
#endif
}
