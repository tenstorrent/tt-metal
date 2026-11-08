// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Down compute (TRISC): the down projection on the down cores. Per virtual expert v (a 128-row sub-block of
// expert e = v / S): y[rows, PCD columns] = h_all(v) @ Wd[:, columns], accumulated in DST over all KT_D K-tiles, one
// row tile at a time (PCD <= 8 accumulators). h_all holds h K-tile-major: [KT_D][MT rows], so a
// gate/up core's slice (its NP K-tiles, all MTG rows) is one contiguous run.
// The expert's down weights stay resident in the in1 ring (RING blocks of [KBLK_D x PCD], filled by
// down_weight_reader.cpp) for its S sub-blocks and are popped after the last one.
//
// CT: 0 MT, 1 KT_D, 2 KBLK_D, 3 PCD, 4 NUM_EXPERTS, 5 S, 6 SLOT_TILES, 7 RING
#include "schedule_meta.hpp"
#include "flat_prof.hpp"
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#ifdef SE_Y_RM
#include "api/compute/pack_untilize.h"
#endif

#ifdef SE_BENCH  // compute-only microbenchmark (as gate_up_compute.cpp): no CB waits / reserves, SE_BENCH sub-blocks
#define cb_wait_front(...) ((void)0)
#define cb_reserve_back(...) ((void)0)
#endif
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t h_all_cb = tt::CBIndex::c_2;
constexpr uint32_t out_cb = tt::CBIndex::c_16;

constexpr uint32_t mt = get_compile_time_arg_val(0);
constexpr uint32_t kt_d = get_compile_time_arg_val(1);
constexpr uint32_t kblk_d = get_compile_time_arg_val(2);
constexpr uint32_t pcd = get_compile_time_arg_val(3);
constexpr uint32_t num_experts = get_compile_time_arg_val(4);
constexpr uint32_t sub = get_compile_time_arg_val(5);
constexpr uint32_t slot = get_compile_time_arg_val(6);
constexpr uint32_t ring = get_compile_time_arg_val(7);
constexpr uint32_t nblk = kt_d / kblk_d;
constexpr uint32_t hk = 8;
constexpr uint32_t h_all_tiles = kt_d * mt;
// PCD > DST output columns (small-I subgrids have few down cores): column passes of <= 8 bf16 / 4 fp32 (SE_DN_FP32:
// fp32 DEST accumulation over K, the host sets fp32_dest_acc_en) DST tiles, each re-running the row's K loop over the
// same h and resident weights
// (SE_DN_FP32 comes with a full-sync DST: 16 bf16 / 8 fp32 tiles, math and pack no longer overlap)
constexpr uint32_t dst_cols = 8;
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
uint32_t d_rd = 0;  // popped % RING, kept by the pops (no division in the K loop)
// (profiler: cycles waiting for h / weights (UNPACK), out space and math (PACK), DST (MATH), the matmul (MATH))
FP_VAR(fp_wh);
FP_VAR(fp_ww);
FP_VAR(fp_ores);
FP_VAR(fp_regw);
FP_VAR(fp_acq);
FP_VAR(fp_mm);

#ifdef SE_DN_REG
// Pinned schedule (expert_schedule.hpp): load l's block j is stream block l * NBLK + j in ring slot region(l) * NBLK +
// j, popped by count after its last use (as gate_up_compute.cpp's gate/up ring).
FORCE_INLINE uint32_t wblock_dyn(uint32_t p, uint32_t phys) {
    if (p + 1 > popped) {
        FP_T0(t);
        cb_wait_front(in1_cb, (p + 1 - popped) * slot);
        FP_ADD(fp_ww, t);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(phys) - static_cast<int32_t>(d_rd)) * slot;
}
#endif

// a: the stream block, ar: its ring slot (a % RING)
FORCE_INLINE uint32_t wblock(uint32_t a, uint32_t ar) {
    FP_T0(t);
    cb_wait_front(in1_cb, (a - popped + 1) * slot);
    FP_ADD(fp_ww, t);
    return static_cast<uint32_t>(static_cast<int32_t>(ar) - static_cast<int32_t>(d_rd)) * slot;
}

// One row tile's column pass [c0, c0 + cw): DST += h_all row r @ the resident weights over all KT_D K tiles, a weight
// block (KBLK_D K tiles) at a time. b0: the load's first stream block, ph0: its first ring slot (SE_DN_REG; else b0 %
// RING). POP: the expert's last row's final pass - each weight block goes as soon as it is used, so the next expert's
// blocks stream into the ring while this row still runs. Only POP = false is inlined: the TRISCs' code fetch limits
// the K loop once it grows (as gate_up_compute.cpp's row passes), so the popping row is out of line.
template <bool POP>
FORCE_INLINE void dn_row(uint32_t r, uint32_t c0, uint32_t cw, uint32_t b0, uint32_t ph0) {
    for (uint32_t j = 0; j < nblk; ++j) {
#ifdef SE_DN_REG
        const uint32_t w = wblock_dyn(b0 + j, ph0 + j);
#else
        const uint32_t w = wblock(b0 + j, ph0 + j < ring ? ph0 + j : ph0 + j - ring);
#endif
        for (uint32_t i = 0; i < kblk_d; ++i) {
            matmul_block(h_all_cb, in1_cb, r + (j * kblk_d + i) * mt, w + i * pcd + c0, 0, false, cw, 1, hk);
        }
        if constexpr (POP) {
            cb_pop_front(in1_cb, slot);
            ++popped;
            d_rd = d_rd + 1 == ring ? 0 : d_rd + 1;
        }
    }
}
template <bool POP>
__attribute__((noinline)) void dn_row_cold(uint32_t r, uint32_t c0, uint32_t cw, uint32_t b0, uint32_t ph0) {
    dn_row<POP>(r, c0, cw, b0, ph0);
}

void kernel_main() {
    FP_ZONE("FRE-DN-C");
    compute_kernel_hw_startup<SrcOrder::Reverse>(h_all_cb, in1_cb, out_cb);
#ifdef SE_PACK_SRND  // stochastic rounding in the packer (the default rounds bf16 -> bfp8 ties away from zero)
    UNPACK((_llk_unpack_configure_stoch_rnd_<StochRndType::Pack>()));
#endif
    matmul_block_init(h_all_cb, in1_cb, false, cpw, 1, hk);
#ifdef SE_Y_RM
    pack_untilize_dest_init<cpw, pcd>(out_cb);
#endif
#ifdef SE_BENCH
    {
        FP_ZONE("DN-BENCH");
        const uint32_t brows = get_arg_val<uint32_t>(1), bb0 = get_arg_val<uint32_t>(2);
        for (uint32_t v = 0; v < SE_BENCH; ++v) {
            for (uint32_t r = 0; r < brows; ++r) {
                tile_regs_acquire();
#ifdef SE_BENCH_COLD  // (the op's code shape: the popping row's out-of-line path present, a runtime flag)
                if (get_arg_val<uint32_t>(0) && r == brows - 1) {
                    dn_row_cold<true>(r, 0, pcd <= cpw ? pcd : cpw, bb0, bb0);
                } else
#endif
                    dn_row<false>(r, 0, pcd <= cpw ? pcd : cpw, bb0, bb0);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < pcd; ++i) {
                    pack_tile<true>(i, out_cb, r * pcd + i);
                }
                tile_regs_release();
            }
            cb_push_back(out_cb, mt * pcd);
        }
    }
    return;
#endif
    // Dynamic counts: CB 6 holds [n_act, num_v, subs of each active expert] (from the down core's data movement); the
    // in1 ring holds only the active experts, so an active expert's index is its place in the stream.
    cb_wait_front(tt::CBIndex::c_6, 1);
    const uint32_t n_act = read_tile_value(tt::CBIndex::c_6, 0, 0);
    const uint32_t num_v = read_tile_value(tt::CBIndex::c_6, 0, 1);
    uint32_t e = 0, s = 0, subs_e = n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS) : 0;
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
        {
            FP_T0(t);
            cb_wait_front(h_all_cb, h_all_tiles);
            FP_ADD(fp_wh, t);
        }
#ifdef SE_PROF_SB  // (profiler: one zone a sub-block's down projection, after its h has landed)
        DeviceZoneScopedN("DN-SB");
#endif
#ifndef SE_Y_RM
        {
            FP_T0(t);
            cb_reserve_back(out_cb, mt * pcd);
            FP_ADD(fp_ores, t);
        }
#endif
        for (uint32_t r = 0; r < rows; ++r) {
#ifdef SE_Y_RM
            {
                cb_reserve_back(out_cb, pcd);
            }
#endif
            for (uint32_t c0 = 0; c0 < pcd; c0 += cpw) {
                // (one column pass when PCD fits DST: its width a constant)
                const uint32_t cw = pcd <= cpw ? pcd : (pcd - c0 < cpw ? pcd - c0 : cpw);
                const bool final_pass = c0 + cw == pcd;
                if constexpr (pcd > cpw) {
                    matmul_block_init(h_all_cb, in1_cb, false, cw, 1, hk);
                }
                {
                    FP_T0(t);
                    tile_regs_acquire();
                    FP_ADD(fp_acq, t);
                }
                FP_T0(tmm);
#ifdef SE_DN_REG
                const uint32_t b0 = ld * nblk, r0 = ph0;
#else
                const uint32_t b0 = e * nblk, r0 = b0 % ring;
#endif
                if (last_sub && final_pass && r == rows - 1) {
                    dn_row_cold<true>(r, c0, cw, b0, r0);
                } else {
                    dn_row<false>(r, c0, cw, b0, r0);
                }
                FP_ADD(fp_mm, tmm);
                tile_regs_commit();
                {
                    FP_T0(t);
                    tile_regs_wait();
                    FP_ADD(fp_regw, t);
                }
#if defined(SE_Y_RM)
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
        cb_pop_front(h_all_cb, h_all_tiles);
#ifndef SE_Y_RM
        cb_push_back(out_cb, mt * pcd);
#endif
        if (++s == subs_e) {
            s = 0;
            ++e;
            subs_e = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_SUBS + e) : 0;
            lmt = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_LMT + e) : mt;
#ifdef SE_DN_REG
            gw = e < n_act ? read_tile_value(tt::CBIndex::c_6, 0, SE_META_GU + e) : 0;
#endif
        }
    }
    cb_pop_front(tt::CBIndex::c_6, 1);
    FP_EMIT("DN-WH", fp_wh);
    FP_EMIT("DN-WW", fp_ww);
    FP_EMIT("DN-ORES", fp_ores);
    FP_EMIT("DN-REGW", fp_regw);
    FP_EMIT("DN-ACQ", fp_acq);
    FP_EMIT("DN-MM", fp_mm);
}
