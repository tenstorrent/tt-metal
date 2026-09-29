// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// K-split SDPA merge (sdpa_k_split_merge), compute (fp32 DEST, full sync: 8 tiles). Per work item, partitions p < S
// with row max m_p (column 0 of the max tile), running sum l_p (row sum of the 32 per-column partials) and
// unnormalized output O_p:
//   M = max_p m_p, a_p = exp(scale (m_p - M)), coef_p = a_p / sum_q a_q l_q, out = sum_p coef_p O_p.
// S <= 3, two DEST sessions per item, no fp32 round trips through L1:
//   1. m_p (copy) and L_p = l_p x ones (matmul: every column the row sum) into DEST, the coefficients by SFPU in
//      place (m: D[0, S), L then a L: D[S, 2S), M then 1 / den: D[2S]) -> packed to c_25 (fp32)
//   2. per group of G output column tiles: coef_p (column broadcast) * O_p for every partition into DEST, summed by
//      SFPU adds, packed as bf16 output
// 3 < S <= 6 (2 S + 1 tiles do not fit): the row sums L_p get their own session first (packed fp32 to c_24, unpacked
// straight to DEST), and session 1 holds a_p in D[0, S), M then den in D[S], L_p one at a time in D[S + 1].
// Items alternate between two data movement lanes (ksplit_merge_dm.cpp): item k's CBs are lane k % 2's
// (max c_(4l), sum c_(1+4l), O c_(2+4l), out c_(16+l)); c_3 all-ones tile, c_24 row sums, c_25 coefficients.
// CT: 0 S, 1 DVt, 2 scale (fp32 bits)   RT: 0 items on this core
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t CB_ONE = tt::CBIndex::c_3, CB_SUM = tt::CBIndex::c_24, CB_COEF = tt::CBIndex::c_25;

// The coefficient math is per row: only column 0 of each stats tile matters, so the SFPU works on the left column
// of faces only (VectorMode::C: columns 0-15, half the work of a full tile). Same calls as the compute API wrappers.
ALWI void sub_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void add_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void mul_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void mul_scalar_c(uint32_t i, uint32_t bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        i,
        VectorMode::C,
        bits));
}
constexpr uint32_t DST = 8;  // fp32 DEST tiles, full sync
constexpr uint32_t scale_bits_g = get_compile_time_arg_val(2);

// D[0, S): m_p -> a_p; D[x]: M (max over p); scale and exp in place
ALWI void max_sub_exp(uint32_t S, uint32_t x) {
    binary_max_tile_init();
    binary_max_tile(0, S > 1 ? 1 : 0, x, VectorMode::C);
    for (uint32_t p = 2; p < S; ++p) {
        binary_max_tile(x, p, x, VectorMode::C);
    }
    sub_binary_tile_init();
    for (uint32_t p = 0; p < S; ++p) {
        sub_c(p, x, p);
    }
    binop_with_scalar_tile_init();
    for (uint32_t p = 0; p < S; ++p) {
        mul_scalar_c(p, scale_bits_g);
    }
    exp_tile_init<false>();
    for (uint32_t p = 0; p < S; ++p) {
        exp_tile<false>(p, VectorMode::C);  // a_p
    }
}

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), DVt = get_compile_time_arg_val(1);
    static_assert(S >= 1 && S + 2 <= DST, "session 1 needs S + 2 DEST tiles (S <= 6)");
    constexpr bool FAST = 2 * S + 1 <= DST;                // m_p and L_p together in DEST
    constexpr uint32_t G = DST / S < DVt ? DST / S : DVt;  // output column tiles per session-2 group
    const uint32_t mine = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(tt::CBIndex::c_0, CB_ONE, CB_COEF);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t k = 0; k < mine; ++k) {
        const uint32_t lane = k & 1;
        const uint32_t CB_M = 4 * lane, CB_L = 1 + 4 * lane, CB_O = 2 + 4 * lane, CB_OUT = 16 + lane;

        // 1. coefficients
        cb_wait_front(CB_M, S);
        cb_wait_front(CB_L, S);
        if constexpr (!FAST) {
            // 0. row sums L_p = l_p x ones -> c_24 (fp32)
            cb_reserve_back(CB_SUM, S);
            tile_regs_acquire();
            reconfig_data_format(CB_L, CB_ONE);
            matmul_init(CB_L, CB_ONE);
            for (uint32_t p = 0; p < S; ++p) {
                matmul_tiles(CB_L, CB_ONE, p, 0, p);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_SUM);
            for (uint32_t p = 0; p < S; ++p) {
                pack_tile(p, CB_SUM);
            }
            tile_regs_release();
            cb_push_back(CB_SUM, S);
            cb_wait_front(CB_SUM, S);
        }
        cb_reserve_back(CB_COEF, S);
        tile_regs_acquire();
        reconfig_data_format(CB_M, CB_ONE);
        copy_init(CB_M);
        for (uint32_t p = 0; p < S; ++p) {
            copy_tile(CB_M, p, p);
        }
        constexpr uint32_t X = FAST ? 2 * S : S;  // M, then the denominator
        if constexpr (FAST) {
            matmul_init(CB_L, CB_ONE);
            for (uint32_t p = 0; p < S; ++p) {
                matmul_tiles(CB_L, CB_ONE, p, 0, S + p);  // every column = the row sum of the 32 partials
            }
            max_sub_exp(S, X);
            mul_binary_tile_init();
            for (uint32_t p = 0; p < S; ++p) {
                mul_c(p, S + p, S + p);  // a_p L_p
            }
            add_binary_tile_init();
            for (uint32_t p = 1; p < S; ++p) {
                add_c(S, S + p, S);  // den
            }
        } else {
            max_sub_exp(S, X);
            constexpr uint32_t Y = S + 1;
            for (uint32_t p = 0; p < S; ++p) {
                reconfig_data_format_srca(CB_SUM);
                copy_init(CB_SUM);
                copy_tile(CB_SUM, p, Y);  // L_p (unpacked to DEST in fp32)
                mul_binary_tile_init();
                mul_c(p, Y, p == 0 ? X : Y);  // a_p L_p (p = 0: replaces M)
                if (p > 0) {
                    add_binary_tile_init();
                    add_c(X, Y, X);  // den
                }
            }
        }
        constexpr uint32_t DEN = FAST ? S : X;
        recip_tile_init();
        recip_tile(DEN, VectorMode::C);
        mul_binary_tile_init();
        for (uint32_t p = 0; p < S; ++p) {
            mul_c(p, DEN, p);  // coef_p
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_COEF);
        for (uint32_t p = 0; p < S; ++p) {
            pack_tile(p, CB_COEF);
        }
        tile_regs_release();
        cb_push_back(CB_COEF, S);
        cb_pop_front(CB_M, S);
        cb_pop_front(CB_L, S);
        if constexpr (!FAST) {
            cb_pop_front(CB_SUM, S);
        }

        // 2. out = sum_p coef_p * O_p, up to G column tiles at a time: partition p's products in D[p G, p G + g)
        cb_wait_front(CB_COEF, S);
        cb_wait_front(CB_O, S * DVt);
        cb_reserve_back(CB_OUT, DVt);
        for (uint32_t j0 = 0; j0 < DVt; j0 += G) {
            const uint32_t g = DVt - j0 < G ? DVt - j0 : G;
            tile_regs_acquire();
            reconfig_data_format(CB_O, CB_COEF);
            mul_bcast_cols_init(CB_O, CB_COEF);
            for (uint32_t p = 0; p < S; ++p) {
                for (uint32_t j = 0; j < g; ++j) {
                    mul_tiles_bcast_cols(CB_O, CB_COEF, p * DVt + j0 + j, p, p * G + j);
                }
            }
            if constexpr (S > 1) {
                add_binary_tile_init();
                for (uint32_t p = 1; p < S; ++p) {
                    for (uint32_t j = 0; j < g; ++j) {
                        add_binary_tile(j, p * G + j, j);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            for (uint32_t j = 0; j < g; ++j) {
                pack_tile(j, CB_OUT);
            }
            tile_regs_release();
        }
        cb_push_back(CB_OUT, DVt);
        cb_pop_front(CB_COEF, S);
        cb_pop_front(CB_O, S * DVt);
    }
}
