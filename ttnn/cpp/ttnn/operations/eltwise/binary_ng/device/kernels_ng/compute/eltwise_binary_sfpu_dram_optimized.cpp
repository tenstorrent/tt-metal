// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/binary_comp.h"

#include "lltt.h"
#include "sfpi.h"

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

#include "tools/profiler/kernel_profiler.hpp"

// experimental
namespace {

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU elementwise multiplication.
//
// The standard `mul_binary_tile` expands (per tile) to 4 faces x 8 sfpi
// iterations of compiled SFPU instructions. That sequence is identical from
// one tile to the next for a given precision mode; the only thing that needs
// to change per tile is the DST base. We therefore record one sfpi-iteration
// body once at init time into the MATH-thread replay buffer and replay it at
// runtime, matching the output of the FPU-based `BINARY_OP` (eltwise multiply)
// used in `eltwise_binary_dram_optimized.cpp`:
//   * multiply in FP32 inside the SFPU
//   * when dest is BF16, software RNE FP32 -> BF16 and clamp 0*x = x*0 = 0
//     so the result matches the FPU's BF16 multiply bit-exactly
//   * when dest is FP32, skip rounding/clamping
//
// Layout baked into the recorded body (offsets are in dest rows, relative to
// the current dst_reg base set by `_llk_math_eltwise_binary_sfpu_start_`):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1], next tile slot)
//   result overwrites offset 0
// The body ends with INCRWC(SFP_DESTREG_STRIDE) mirroring `sfpi::dst_reg++`.
//
// The per-tile wrapper preserves the 4-face / 2 x SETRWC face-advance
// structure of `_llk_math_eltwise_binary_sfpu_params_` but seeds the DST
// base with `idst0` so the fixed offsets land on DST[idst0] and
// DST[idst0 + 1]. Callers must therefore use the same pairing the kernel
// already relies on: `mul_binary_tile_replay(i*2, i*2 + 1, i*2)`.
// ---------------------------------------------------------------------------

// Number of instructions in one recorded sfpi-iteration body.
//
// Two implementations are supported, selected at compile time:
//
//   DISABLE_SFPLOADMACRO defined  -> discrete SFPU instructions (SFPLOAD x2,
//                                    SFPMUL, etc.). Used as a portability /
//                                    debugging fallback or on builds where
//                                    the SFPLOADMACRO unit is unavailable.
//     is_fp32_dest_acc_en = true  : SFPLOAD x2 + SFPMUL + SFPNOP + SFPSTORE
//                                   + INCRWC = 6
//     is_fp32_dest_acc_en = false : + SFP_STOCH_RND
//                                   + 2 * (SFPSETCC + SFPMOV + SFPENCC) = 13
//
//   DISABLE_SFPLOADMACRO undefined -> SFPLOADMACRO fuses LD + MAD (+ optional
//                                     STOCH_RND) (+ optional STORE) into a
//                                     single instruction driven by Macro
//                                     Sequence Register 0. RHS is pre-loaded
//                                     into LREG[0] via a regular SFPLOAD;
//                                     LHS is loaded by SFPLOADMACRO into
//                                     LREG[1], where the macro's MAD
//                                     overwrites it with LREG[0]*LREG[1].
//     is_fp32_dest_acc_en = true  : SFPLOAD + SFPLOADMACRO + INCRWC = 3
//                                   (macro pipeline does LD + MAD + STORE)
//     is_fp32_dest_acc_en = false : 2*SFPLOAD + SFPLOADMACRO
//                                   + 2 * (SFPSETCC + SFPMOV + SFPENCC)
//                                   + SFPSTORE + INCRWC = 11
//                                   (macro pipeline does LD + MAD + STOCH_RND;
//                                    zero-clamp and STORE are issued manually)
#ifdef DISABLE_SFPLOADMACRO
constexpr std::uint32_t SFPU_BINARY_MUL_REPLAY_LEN = DST_ACCUM_MODE ? 6 : 13;
#else
constexpr std::uint32_t SFPU_BINARY_MUL_REPLAY_LEN = DST_ACCUM_MODE ? 3 : 11;
#endif

// A 32x32 tile occupies 64 rows in dest.
constexpr std::uint32_t SFPU_BINARY_MUL_DST_TILE_ROWS = 64;

/*
List of ops I need to add support for sfpu:

*/
// TODO: template specisaliztion based on op type, alias for template? add = generisc_sfpu_elt<OP=add>

// These helpers reference LLK symbols (e.g. `_llk_math_eltwise_binary_sfpu_start_`)
// which are only visible on the MATH thread. Gate the definitions accordingly so
// UNPACK/PACK TUs do not try to parse them.
#ifdef TRISC_MATH

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void _program_sfpu_binary_mul_replay_() {
#ifndef DISABLE_SFPLOADMACRO
    // ----- One-time SFPLOADMACRO setup (NOT recorded into the replay buffer) -----
    //
    // Each SFPLOADMACRO call performs:
    //   1. LD  : load from dst[base + dest_reg_addr] into a loaded LREG
    //            (loaded LREG = lreg_ind[1:0], so LREG[0..3] only)
    //   2. Pipeline : run the slots scheduled by Macro Sequence Register
    //                 lreg_ind[3:2] (SIMPLE / MAD / ROUND / STORE).
    //
    // Slot bit layout (per typecast/exp init patterns):
    //   bit 7 : UsesLoadValAsSrcB (bit 6 src for SIMPLE) -- override the
    //           instruction's srcB with the loaded LREG
    //   bit 6 : UsesStaging        -- redirect the instruction's dest to
    //           LREG[16] (staging) instead of the loaded LREG. We do not need
    //           staging here: MAD writes the product back into the loaded
    //           LREG, which the next stage then consumes.
    //   bits 5:3 : delay (cycles after LD before this slot fires)
    //   bits 2:0 : macro instruction mux index
    //              (3 = fixed STORE, 4..7 = programmable templates 0..3)
    //
    // Programmable Macro Instruction Template 1 (mux 5): SFPMAD
    //   lreg_dest = 13 selects backdoor-load into template 1.
    //   When triggered by SFPLOADMACRO with mad_bits bit 7 = 1 and bit 6 = 0:
    //     loaded_LREG = LREG[0] * loaded_LREG + LCONST_0
    //                 = RHS * LHS = LHS * RHS
    TTI_SFPMAD(p_sfpu::LREG0, 0, p_sfpu::LCONST_0, 13, 0);

    if constexpr (!is_fp32_dest_acc_en) {
        // Programmable Macro Instruction Template 2 (mux 6): SFP_STOCH_RND
        //   lreg_dest = 14 selects backdoor-load into template 2.
        //   When triggered with round_bits bit 7 = 0 and bit 6 = 0, this
        //   stochastic-rounds the loaded LREG (the MAD result, FP32) into
        //   FP16B in place. This mirrors the original SFP_STOCH_RND on LREG[2]
        //   in the discrete-instruction implementation.
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }

    // Macro Sequence Register 0:
    //   simple slot: disabled
    //   MAD slot   : template 1 (SFPMAD) at delay 0; bit 7 = 1 selects loaded
    //                LREG as srcB; bit 6 = 0 writes the result back to the
    //                loaded LREG (no staging).
    //   round slot : FP32 mode -> disabled.
    //                BF16 mode -> template 2 (SFP_STOCH_RND) at delay 2,
    //                writes FP16B back to the loaded LREG.
    //   store slot : FP32 mode -> fixed STORE (mux 3) at delay 2. Reads the
    //                loaded LREG (= MAD result) and writes it to the LD's
    //                dest_reg_addr. This replaces the discrete SFPSTORE.
    //                BF16 mode -> disabled. The post-macro zero-clamp logic
    //                must run before the store, so we issue SFPSTORE manually.
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0x80 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = is_fp32_dest_acc_en ? 0u : (0x00u | 0x00u | (2u << 3) | (4u + 2u));
        constexpr std::uint32_t store_bits = is_fp32_dest_acc_en ? (0x00u | 0x00u | (2u << 3) | 3u) : 0u;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc config: UnitDelayKind[0] = 1 (WaitForElapsedInstructions, prevents
    // pipeline advancement on dest bank conflicts). StoreMod0 = DEFAULT lets
    // the store inherit the dst's natural format (matches the original
    // SFPSTORE's InstrModLoadStore::DEFAULT).
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::DEFAULT, 8, 1);
#endif  // !DISABLE_SFPLOADMACRO

    // ----- Recorded per-iteration replay body -----
    //
    // Layout (offsets in dest rows, relative to the current dst_reg base):
    //   operand A (LHS)  at offset 0
    //   operand B (RHS)  at offset 64 (= 1 * SFPU_BINARY_MUL_DST_TILE_ROWS)
    //   result overwrites offset 0
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_REPLAY_LEN);

#ifdef DISABLE_SFPLOADMACRO
    // -----------------------------------------------------------------------
    // Discrete-instruction fallback (DISABLE_SFPLOADMACRO defined).
    // -----------------------------------------------------------------------
    // Loads both operands explicitly, multiplies them with SFPMUL, and stores
    // the result. This is the original implementation, preserved as a
    // portability path for builds where SFPLOADMACRO is unavailable or
    // disabled for debugging.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPNOP;

    if constexpr (!is_fp32_dest_acc_en) {
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);

        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPENCC(0, 0, 0, 0);
    }

    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
#else
    // -----------------------------------------------------------------------
    // SFPLOADMACRO-fused implementation.
    // -----------------------------------------------------------------------
    // Pre-load operand B (RHS) at offset 64 into LREG[0]. The MAD template
    // installed above uses LREG[0] as srcA, so this value participates in the
    // multiply executed inside the SFPLOADMACRO pipeline.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    if constexpr (!is_fp32_dest_acc_en) {
        // BF16 also needs the original LHS value to drive the post-mul zero
        // clamp (the macro's MAD overwrites LREG[1] with the product, so we
        // can no longer test LHS against zero from there).
        TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }

    // Issue SFPLOADMACRO: macro_idx = 0, loaded LREG = LREG[1], offset = 0.
    //   FP32 : LD LHS -> LREG[1]; MAD LREG[0]*LREG[1] -> LREG[1]; STORE LREG[1] to offset 0.
    //   BF16 : LD LHS -> LREG[1]; MAD LREG[0]*LREG[1] -> LREG[1]; STOCH_RND fp32->fp16b on LREG[1].
    TTI_SFPLOADMACRO((0 << 2) | 1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);

    if constexpr (!is_fp32_dest_acc_en) {
        // Bit-exact match for the FPU's BF16 multiply when an operand is 0:
        // stochastic rounding can otherwise leave a non-zero LSB on 0*x or x*0.
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_DST_TILE_ROWS);
    }
#endif  // DISABLE_SFPLOADMACRO

    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _llk_math_eltwise_binary_sfpu_binop_mul_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, SFPU_BINARY_MUL_REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

#endif  // TRISC_MATH

// Program the MATH-thread replay buffer with one sfpi-iteration body of the
// SFPU multiply. Must be called once, after `BINARY_SFPU_INIT`, so the SFPU
// is already configured. Uses `lltt::NoExec` so recording does not emit any
// actual SFPU work - only the recorded instructions are installed.
ALWI void mul_binary_tile_init_replay() { MATH((_program_sfpu_binary_mul_replay_<APPROX, DST_ACCUM_MODE>())); }

// Drop-in replacement for `mul_binary_tile(idst0, idst1, odst)` that performs
// the multiply by replaying the pre-recorded body. Requires `idst1 == idst0 + 1`
// and `odst == idst0`, which matches the kernel's `(i*2, i*2 + 1, i*2)` layout.
ALWI void mul_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_binop_mul_replay_(idst0)));
}

}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_per_batch = get_arg_val<uint32_t>(1);

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    uint32_t remaining = num_tiles;
    while (remaining > 0) {
        uint32_t n_tiles;
        if (remaining >= num_tiles_per_batch) {
            n_tiles = num_tiles_per_batch;
        } else {
            n_tiles = remaining;
        }

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n_tiles);
        cb_wait_front(cb_post_lhs, n_tiles);

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n_tiles);
        cb_wait_front(cb_post_rhs, n_tiles);

        cb_reserve_back(cb_out, n_tiles);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();
        {
            copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
            DeviceZoneScopedN("copy_tile_ lhs");
            for (uint32_t i = 0; i < n_tiles; ++i) {
                copy_tile(cb_post_lhs, i, i * 2);
            }
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        {
            {
                DeviceZoneScopedN("copy_tile rhs");
                for (uint32_t i = 0; i < n_tiles; ++i) {
                    copy_tile(cb_post_rhs, i, i * 2 + 1);
                }
            }
            {
                DeviceZoneScopedN("compute");

                for (uint32_t i = 0; i < n_tiles; ++i) {
#if HAS_ACTIVATIONS(POST)
                    BINARY_SFPU_INIT
#endif
                    {
                        DeviceZoneScopedN("mul_replay");
                        BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
                    }

                    PROCESS_POST_ACTIVATIONS(i * 2);
                }
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        {
            DeviceZoneScopedN("pack_tile");
            for (uint32_t i = 0; i < n_tiles; ++i) {
                pack_tile(i * 2, cb_out);
            }
            tile_regs_release();
        }

        cb_push_back(cb_out, n_tiles);
        cb_pop_front(cb_post_lhs, n_tiles);
        cb_pop_front(cb_post_rhs, n_tiles);
        remaining -= n_tiles;
    }
}
