// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator compute, the weighted accumulate: each row times its gate weight (HiFi4, scalar broadcast
// from the weight CB's element 0) into DEST (fp32), packed onto the token's fp32 sum with packer L1 accumulation, then
// the bf16 sums out. SNAP (combine_snap q): each product first rounded on the SFPU to a multiple of q, so every fp32
// add is exact while the partial sums stay below 2^24 q - the sum is then the same in any landing order. SNAP_HI
// (MoEBlockConfig::combine_snap_split, C_hi = 1.5 x 2^23 q_hi, q_hi = 2^18 q): each product v split into hi = v rounded
// to a multiple of q_hi and lo = v - hi (exact) rounded to a multiple of q (one SFPU pass, combine_split_face), summed
// into two fp32 sums (CB_ACC: the
// tokens' lo sums, then their hi sums), out = hi + lo rounded once - exact, so order-free, while the hi sums stay below
// 2^24 q_hi (DeepSeek-V4.1 FIX.23: a single sum of 21376 lost its order freedom at 32). The reader ends the rows with
// a 0xFFFFFFFF token (the device plan's counts: the compute does not know them).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

// SNAP_HI's split in one SFPU pass (a face a call, VectorMode::RC): v (this tile) -> lo = ((v - hi) + C) - C here,
// hi = (v + C_HI) - C_HI in the tile HI_TILES on. The SFPU adder rounds to nearest even; the same operations in the
// same order as add_unary / sub_unary / sub_binary would do them.
template <uint32_t C_HI, uint32_t C, int HI_TILES>
inline void combine_split_face() {
    constexpr int HI = HI_TILES * 32;  // (an fp32 DEST tile is 32 SFPU rows)
    const sfpi::vFloat ch = __builtin_bit_cast(float, C_HI);
    const sfpi::vFloat cl = __builtin_bit_cast(float, C);
#pragma GCC unroll 8
    for (int r = 0; r < 8; r++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat hi = v + ch;
        hi = hi - ch;
        sfpi::vFloat lo = v - hi;
        lo = lo + cl;
        lo = lo - cl;
        sfpi::dst_reg[0] = lo;
        sfpi::dst_reg[HI] = hi;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

constexpr uint32_t DH = 4;  // fp32 tiles per DEST half

void kernel_main() {
    constexpr uint32_t cb_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t MAXT = get_compile_time_arg_val(5);
    constexpr uint32_t cb_w = get_compile_time_arg_val(6);
    constexpr uint32_t SNAP = get_compile_time_arg_val(7);     // (fp32 bits of C = 1.5 x 2^23 q; 0: off)
    constexpr uint32_t SNAP_HI = get_compile_time_arg_val(8);  // (fp32 bits of C_hi; 0: one sum a token)
    // (the split needs a DEST slot for hi beside each product: half the tiles a DEST half)
    constexpr uint32_t DT = SNAP_HI ? DH / 2 : DH;
    const uint32_t n_tok = get_arg_val<uint32_t>(0);
    const uint32_t hi_base = n_tok * NT;  // (CB_ACC: the tokens' hi sums after their lo sums)
    compute_kernel_hw_startup(cb_row, cb_w, cb_acc);
    mul_bcast_scalar_init(cb_row, cb_w);
    uint8_t seen[MAXT];
    for (uint32_t i = 0; i < MAXT; ++i) {
        seen[i] = 0;
    }
    cb_reserve_back(cb_acc, (SNAP_HI ? 2 : 1) * n_tok * NT);
    while (true) {
        cb_wait_front(cb_meta, 1);
        const uint32_t t = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
        if (t == 0xFFFFFFFF) {
            break;  // (the reader's end marker)
        }
        cb_wait_front(cb_row, NT);
        cb_wait_front(cb_w, 1);
        pack_reconfig_l1_acc(seen[t] ? 1 : 0);
        for (uint32_t j0 = 0; j0 < NT; j0 += DT) {  // a DEST half holds DH fp32 tiles
            const uint32_t nj = NT - j0 < DT ? NT - j0 : DT;
            tile_regs_acquire();
            if constexpr (SNAP) {
                mul_bcast_scalar_init(cb_row, cb_w);
            }
            for (uint32_t j = 0; j < nj; ++j) {
                mul_tiles_bcast_scalar(cb_row, cb_w, j0 + j, 0, j);
            }
            if constexpr (SNAP_HI) {  // hi (DEST DT + j) = (v + C_hi) - C_hi; lo (DEST j) = ((v - hi) + C) - C
                MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
                for (uint32_t j = 0; j < nj; ++j) {
                    MATH((_llk_math_eltwise_unary_sfpu_params_(
                        ckernel::sfpu::combine_split_face<SNAP_HI, SNAP, DT>, j, VectorMode::RC)));
                }
            } else if constexpr (SNAP) {  // every term to a multiple of q: (v + C) - C, C = 1.5 x 2^23 q
                binop_with_scalar_tile_init();
                for (uint32_t j = 0; j < nj; ++j) {
                    add_unary_tile(j, SNAP);
                    sub_unary_tile(j, SNAP);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nj; ++j) {
                pack_tile<true>(j, cb_acc, t * NT + j0 + j);
            }
            if constexpr (SNAP_HI) {
                for (uint32_t j = 0; j < nj; ++j) {
                    pack_tile<true>(DT + j, cb_acc, hi_base + t * NT + j0 + j);
                }
            }
            tile_regs_release();
        }
        seen[t] = 1;
        cb_pop_front(cb_row, NT);
        cb_pop_front(cb_w, 1);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, (SNAP_HI ? 2 : 1) * n_tok * NT);
    cb_wait_front(cb_acc, (SNAP_HI ? 2 : 1) * n_tok * NT);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    for (uint32_t t = 0; t < n_tok; ++t) {
        cb_reserve_back(cb_out, NT);
        if (!seen[t]) {  // (no rows - a padding token: zeros, not the CB's stale L1)
            fill_tile_init();
            for (uint32_t j0 = 0; j0 < NT; j0 += DH) {
                const uint32_t nj = NT - j0 < DH ? NT - j0 : DH;
                tile_regs_acquire();
                for (uint32_t j = 0; j < nj; ++j) {
                    fill_tile(j, 0.0f);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < nj; ++j) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, NT);
            copy_init(cb_acc);
            continue;
        }
        for (uint32_t j0 = 0; j0 < NT; j0 += DT) {
            const uint32_t nj = NT - j0 < DT ? NT - j0 : DT;
            tile_regs_acquire();
            if constexpr (SNAP_HI) {
                copy_init(cb_acc);
            }
            for (uint32_t j = 0; j < nj; ++j) {
                copy_tile(cb_acc, t * NT + j0 + j, j);
            }
            if constexpr (SNAP_HI) {  // out = lo + hi, one rounding (both sums exact)
                for (uint32_t j = 0; j < nj; ++j) {
                    copy_tile(cb_acc, hi_base + t * NT + j0 + j, nj + j);
                }
                add_binary_tile_init();
                for (uint32_t j = 0; j < nj; ++j) {
                    add_binary_tile(j, nj + j, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nj; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, NT);
    }
}
