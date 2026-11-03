// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// matmul_kblock_cost compute-only bench: the conv2d_nhwc BLOCK K-exchange matmul (bm x bk x bn per
// K-block, packer-L1 fp32 accumulation into a caller-owned partials block) with the exchange removed
// (the producer only cycles the CBs; in0/in1 slots keep their data).
//
// HELPER (baseline): compute_kernel_lib::matmul_block exactly as conv2d_nhwc_compute.cpp calls it
//   (ShortAfterPreKBlock, TileRowMajor, caller-owned partials), PreKBlock = no-op.
// RAW (candidate menu, raw LLK — bypasses compute_kernel_lib::matmul_block because the helper
//   (a) requires out_subblock_h | bm (bm = 13 is prime -> only 1x2 subblocks), (b) cannot accumulate
//   two K-blocks in DEST (its in0 index assumes one contiguous [row][bk] block per K step), and
//   (c) re-issues pack_reconfig_data_format + llk_pack_reconfig_l1_acc on every subblock):
//   SBH      subblock rows (1 or 2; a 13-row block with SBH=2 ends in a 1-row tail)
//   KACC     K-blocks accumulated in DEST before one packer-L1 pass (1 or 2)
//   INIT_KB  re-run reconfig + matmul_block_init each K-block group (as ShortAfterPreKBlock does)
//   PCFG_SB  pack reconfig + l1_acc per subblock (helper behaviour) instead of once per group
//   TAIL_OVER  tail rows computed as a full SBH subblock (extra rows unpacked/computed, not packed)
//   NO_PACK / NO_MATH  diagnostics (skip packs / skip matmul LLK calls)

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
constexpr uint32_t cb_in1 = get_compile_time_arg_val(1);
constexpr uint32_t cb_out = get_compile_time_arg_val(2);
constexpr uint32_t BM = get_compile_time_arg_val(3);
constexpr uint32_t BK = get_compile_time_arg_val(4);
constexpr uint32_t BN = get_compile_time_arg_val(5);
constexpr uint32_t NKB = get_compile_time_arg_val(6);
constexpr uint32_t IN0_T = BM * BK;
constexpr uint32_t IN1_T = BK * BN;
constexpr uint32_t OUT_T = BM * BN;

#ifndef SBH
#define SBH 1
#endif
#ifndef KACC
#define KACC 1
#endif
#ifndef SBW
#define SBW BN
#endif

struct NoopPre {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

using namespace compute_kernel_lib;

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    CircularBuffer in0(cb_in0), in1(cb_in1), out(cb_out);
    out.reserve_back(OUT_T);
#ifdef HELPER
    const auto shape = MatmulBlockShape::of(BM / 1, 1, 1, BN, BK, NKB);
    matmul_block<
        false,
        true,
        LastBlockTarget::Interm,
        OutputCBLayout::TileRowMajor,
        matmul_config::InitMode::ShortAfterPreKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        NoPostCompute,
        NoopPre,
        NoPostKBlock,
        0,
        NoKBlockInnerDimFn,
        NoIn0Source,
        NoIn1BaseOffset,
        true>(in0, in1, out, out, shape, {}, NoopPre{}, BN, BN, {}, {}, {}, {});
#else
    static_assert(NKB % KACC == 0);
    // SBW == BN: wide subblocks (reuse in0 across the bn in1 tiles), SBH rows each.
    // SBW == 1 : tall subblocks (reuse one in1 tile across SBH in0 rows), per output column.
    // Rows past the last whole SBH group form a 1 x BN tail (its own init) unless TAIL_OVER.
    constexpr uint32_t full_rows = (BM / SBH) * SBH;
    constexpr uint32_t tail = BM - full_rows;
    constexpr uint32_t ncols = BN / SBW;
    matmul_block_init(cb_in0, cb_in1, 0, SBW, SBH, BK);
    for (uint32_t kb = 0; kb < NKB; kb += KACC) {
        in0.wait_front(KACC * IN0_T);
        in1.wait_front(KACC * IN1_T);
#ifdef INIT_KB
        reconfig_data_format(cb_in1, cb_in0);
        PACK((pack_reconfig_data_format(cb_out)));
        matmul_block_init(cb_in0, cb_in1, 0, SBW, SBH, BK);
#elif !defined(TAIL_OVER)
        if constexpr (tail != 0) {
            if (kb != 0) {
                matmul_block_init(cb_in0, cb_in1, 0, SBW, SBH, BK);
            }
        }
#endif
#ifndef PCFG_SB
        PACK((llk_pack_reconfig_l1_acc(kb == 0 ? 0 : 1)));
#endif
#ifdef TAIL_OVER
        constexpr uint32_t row_end = BM;
#else
        constexpr uint32_t row_end = full_rows;
#endif
        for (uint32_t r0 = 0; r0 < row_end; r0 += SBH) {
            const uint32_t prows = r0 + SBH <= BM ? SBH : BM - r0;
            for (uint32_t c0 = 0; c0 < BN; c0 += SBW) {
                tile_regs_acquire();
#ifndef NO_MATH
#ifdef NOUNROLL
#pragma GCC unroll 1
#endif
                for (uint32_t g = 0; g < KACC; ++g) {
#ifdef NOUNROLL
#pragma GCC unroll 1
#endif
                    for (uint32_t k = 0; k < BK; ++k) {
                        matmul_block(
                            cb_in0, cb_in1, g * IN0_T + r0 * BK + k, g * IN1_T + k * BN + c0, 0, 0, SBW, SBH, BK);
                    }
                }
#endif
                tile_regs_commit();
                tile_regs_wait();
#ifdef PCFG_SB
                PACK((pack_reconfig_data_format(cb_out)));
                PACK((llk_pack_reconfig_l1_acc(kb == 0 ? 0 : 1)));
#endif
#ifndef NO_PACK
                for (uint32_t r = 0; r < prows; ++r) {
                    for (uint32_t c = 0; c < SBW; ++c) {
                        pack_tile<true>(r * SBW + c, cb_out, (r0 + r) * BN + c0 + c);
                    }
                }
#endif
                tile_regs_release();
            }
        }
#ifndef TAIL_OVER
        if constexpr (tail != 0) {
            matmul_block_init(cb_in0, cb_in1, 0, BN, tail, BK);
            tile_regs_acquire();
#ifndef NO_MATH
            for (uint32_t g = 0; g < KACC; ++g) {
                for (uint32_t k = 0; k < BK; ++k) {
                    matmul_block(
                        cb_in0, cb_in1, g * IN0_T + full_rows * BK + k, g * IN1_T + k * BN, 0, 0, BN, tail, BK);
                }
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
#ifndef NO_PACK
            for (uint32_t t = 0; t < tail * BN; ++t) {
                pack_tile<true>(t, cb_out, full_rows * BN + t);
            }
#endif
            tile_regs_release();
        }
#endif
        in0.pop_front(KACC * IN0_T);
        in1.pop_front(KACC * IN1_T);
    }
#endif
    out.push_back(OUT_T);
    PACK((llk_pack_reconfig_l1_acc(0)));
}
