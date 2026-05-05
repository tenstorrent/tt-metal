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

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU bitwise operations: AND / OR / XOR.
//
// Mirrors the multiplication path. One sfpi-iteration body is recorded into
// the MATH-thread replay buffer at init time; per tile, the recorded body
// is replayed 8 times per face for each of the 4 faces. The recorded body
// is identical to one iteration of `_calculate_sfpu_binary_bitwise_`:
//
//   SFPLOAD  LREG[0] <- DST[0]                  // operand A
//   SFPLOAD  LREG[1] <- DST[1 * tile_rows]      // operand B
//   SFPAND/OR/XOR  LREG[0] = LREG[0] op LREG[1] // bitwise op
//   SFPSTORE LREG[0] -> DST[0]                  // result overwrites A
//   INCRWC                                      // advance dst_reg by SFP_DESTREG_STRIDE
//
// = 5 instructions, identical for all three ops; only the AND/OR/XOR opcode
// in slot 3 differs.
//
// Layout (offsets in dest rows, relative to the current dst_reg base set by
// `_llk_math_eltwise_binary_sfpu_start_`):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1], next tile slot)
//   result overwrites offset 0
// matching the kernel pairing `bitwise_<op>_binary_tile(i*2, i*2+1, i*2)`.
//
// `data_format` selects the SFPLOAD/SFPSTORE instruction mode the same way
// `llk_math_eltwise_binary_sfpu_bitwise<>` does:
//   UInt16 -> InstrModLoadStore::LO16
//   Int32 / UInt32 -> InstrModLoadStore::INT32
// Anything else fails the static_assert below. The op + format are baked
// into the replay buffer at init time, so the per-tile loop is op-agnostic.
// ---------------------------------------------------------------------------

constexpr std::uint32_t SFPU_BINARY_BITWISE_REPLAY_LEN = 5;
constexpr std::uint32_t SFPU_BINARY_BITWISE_DST_TILE_ROWS = 64;

#ifdef TRISC_MATH

template <ckernel::sfpu::BinaryBitwiseOp BITWISE_OP, InstrModLoadStore INSTRUCTION_MODE>
inline void _program_sfpu_binary_bitwise_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_BITWISE_REPLAY_LEN);

    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_BITWISE_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 1 * SFPU_BINARY_BITWISE_DST_TILE_ROWS);

    // SFP{AND,OR,XOR} encoding (TT_OP_SFP{AND,OR,XOR}(imm12, lreg_c, lreg_dest, instr_mod1)):
    //   LREG[lreg_dest] = LREG[lreg_dest] op LREG[lreg_c]
    // -> LREG[0] = LREG[0] op LREG[1].
    if constexpr (BITWISE_OP == ckernel::sfpu::BinaryBitwiseOp::AND) {
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    } else if constexpr (BITWISE_OP == ckernel::sfpu::BinaryBitwiseOp::OR) {
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    } else if constexpr (BITWISE_OP == ckernel::sfpu::BinaryBitwiseOp::XOR) {
        TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    }

    TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_BITWISE_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _llk_math_eltwise_binary_sfpu_bitwise_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, SFPU_BINARY_BITWISE_REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

#endif  // TRISC_MATH

// DataFormat -> InstrModLoadStore mirrors `llk_math_eltwise_binary_sfpu_bitwise`:
// UInt16 -> LO16, Int32 / UInt32 -> INT32. Anything else triggers a compile error.
template <DataFormat data_format>
struct sfpu_bitwise_load_store_mode {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for bitwise replay. Supported: Int32, UInt32, UInt16");
    static constexpr InstrModLoadStore value =
        (data_format == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
};

// One-shot replay-buffer programming helpers, one per op. Must be called once,
// after `binary_bitwise_tile_init`, on the MATH thread. Records into replay
// slot 0 with `lltt::NoExec` so no actual SFPU work is emitted.
template <DataFormat data_format>
ALWI void bitwise_and_binary_init_replay() {
    MATH((_program_sfpu_binary_bitwise_replay_<
          ckernel::sfpu::BinaryBitwiseOp::AND,
          sfpu_bitwise_load_store_mode<data_format>::value>()));
}

template <DataFormat data_format>
ALWI void bitwise_or_binary_init_replay() {
    MATH((_program_sfpu_binary_bitwise_replay_<
          ckernel::sfpu::BinaryBitwiseOp::OR,
          sfpu_bitwise_load_store_mode<data_format>::value>()));
}

template <DataFormat data_format>
ALWI void bitwise_xor_binary_init_replay() {
    MATH((_program_sfpu_binary_bitwise_replay_<
          ckernel::sfpu::BinaryBitwiseOp::XOR,
          sfpu_bitwise_load_store_mode<data_format>::value>()));
}

// Drop-in replacements for `bitwise_{and,or,xor}_binary_tile<DataFormat>`.
// The op + format are baked into the replay buffer by the matching
// `bitwise_<op>_binary_init_replay<data_format>()`, so the per-tile body is
// op-agnostic and shares `_llk_math_eltwise_binary_sfpu_bitwise_replay_`.
// `data_format` is kept as a template parameter only to mirror the upstream
// API signature. Requires `idst1 == idst0 + 1` and `odst == idst0`.
ALWI void bitwise_and_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_bitwise_replay_(idst0)));
}

ALWI void bitwise_or_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_bitwise_replay_(idst0)));
}

ALWI void bitwise_xor_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_bitwise_replay_(idst0)));
}

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU integer shifts: LEFT_SHIFT / RIGHT_SHIFT.
//
// Mirrors the multiplication / bitwise paths but, unlike bitwise where every
// op compiles to the same 5-instruction body, left and right shift have
// different sfpi sequences so they live in separate replay programmers and
// per-tile loops with their own length constants.
//
// Both ops use the same DataFormat -> InstrModLoadStore mapping as bitwise
// (UInt16 -> LO16, Int32 / UInt32 -> INT32) and assume the SFPU's default
// `SIGN_MAGNITUDE_FORMAT = false` LLK setting (which is also the factory
// default at all `binary_left_shift_tile` / `binary_right_shift_tile` call
// sites). The recorded bodies match `_calculate_binary_left_shift_` and
// `_calculate_binary_right_shift_` instruction-for-instruction so results
// are bit-identical to the upstream non-replay path.
//
// Layout (offsets in dest rows, relative to current dst_reg base):
//   shift_value  at offset 0   (DST[base])
//   shift_amount at offset 64  (DST[base + 1])
//   result overwrites offset 0
// matching the kernel pairing `binary_<l/r>_shift_tile(i*2, i*2+1, i*2)`.
// ---------------------------------------------------------------------------

// LEFT_SHIFT (10 instructions):
//   SFPLOAD x2 + SFPSETCC + SFPIADD + SFPCOMPC + SFPMOV + SFPENCC
//   + SFPSHFT + SFPSTORE + INCRWC
constexpr std::uint32_t SFPU_BINARY_LEFT_SHIFT_REPLAY_LEN = 10;

// RIGHT_SHIFT (18 instructions):
//   SFPLOAD x2 + SFPMOV (save shift_value) + SFPSETCC + SFPIADD
//   + SFPMOV + SFPENCC                                         (out-of-range -> 0)
//   + SFPIADD (negate shift_amount) + SFPSHFT                  (logical right shift)
//   + SFPSETCC x2 + SFPIADD + SFPNOT + SFPSHFT + SFPOR + SFPENCC (sign-extend if neg)
//   + SFPSTORE + INCRWC
constexpr std::uint32_t SFPU_BINARY_RIGHT_SHIFT_REPLAY_LEN = 18;

constexpr std::uint32_t SFPU_BINARY_SHIFT_DST_TILE_ROWS = 64;

#ifdef TRISC_MATH

template <InstrModLoadStore INSTRUCTION_MODE>
inline void _program_sfpu_binary_left_shift_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LEFT_SHIFT_REPLAY_LEN);

    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 1 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);

    // If shift_amount < 0 OR shift_amount >= 32 the shift result is forced to 0.
    // The SETCC begins a "shift_amount >= 0" guard (mod1 = 4 = LREG_GE0); inside
    // the guard we compute (shift_amount - 32) into LREG2 and use SFPCOMPC to
    // flip the CC predicate to "shift_amount >= 32", then zero LREG0.
    TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);  // 0xFE0 = -32 (sign-extended 12-bit imm)
    TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

    // SFPSHFT with positive amount in LREG1 shifts LREG0 left by LREG1.
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

template <InstrModLoadStore INSTRUCTION_MODE>
inline void _program_sfpu_binary_right_shift_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_RIGHT_SHIFT_REPLAY_LEN);

    TTI_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_3, 1 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);

    // Save the original shift_value in LREG4 so the sign-extend block below
    // can test "value < 0" after SFPSHFT has overwritten LREG0.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);

    // Out-of-range guard: if shift_amount < 0 or >= 32, force result to 0.
    TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LCONST_0);  // 0xFE0 = -32
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

    // Negate shift_amount (LREG1 = -LREG1) so SFPSHFT does a logical right shift.
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 6);
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

    // Arithmetic-shift sign-extend: only run when the saved shift_value (LREG4)
    // is negative AND shift_amount > 0 (otherwise nothing to fill).
    //   mask_bits   = ~0 shifted left by (32 - shift_amount), giving the high
    //                 `shift_amount` bits set.
    //   result      = result | mask_bits
    TTI_SFPSETCC(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);
    TTI_SFPIADD(0x020, p_sfpu::LREG1, p_sfpu::LREG2, 5);  // LREG2 = 32 - shift_amount
    TTI_SFPNOT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

    TTI_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_3, 0 * SFPU_BINARY_SHIFT_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _llk_math_eltwise_binary_sfpu_left_shift_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, SFPU_BINARY_LEFT_SHIFT_REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

inline void _llk_math_eltwise_binary_sfpu_right_shift_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, SFPU_BINARY_RIGHT_SHIFT_REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

#endif  // TRISC_MATH

// One-shot replay-buffer programming helpers for shifts. Must be called once,
// after `binary_shift_tile_init`, on the MATH thread. Each helper records
// into replay slot 0 with `lltt::NoExec` so no actual SFPU work is emitted.
// Reuses `sfpu_bitwise_load_store_mode` since the DataFormat -> InstrModLoadStore
// mapping is identical for shifts and bitwise ops.
template <DataFormat data_format>
ALWI void binary_left_shift_init_replay() {
    MATH((_program_sfpu_binary_left_shift_replay_<sfpu_bitwise_load_store_mode<data_format>::value>()));
}

template <DataFormat data_format>
ALWI void binary_right_shift_init_replay() {
    MATH((_program_sfpu_binary_right_shift_replay_<sfpu_bitwise_load_store_mode<data_format>::value>()));
}

// Drop-in replacements for `binary_left_shift_tile<DataFormat>` and
// `binary_right_shift_tile<DataFormat>`. The format is baked into the replay
// buffer by the matching `binary_<l/r>_shift_init_replay<data_format>()`.
// Requires `idst1 == idst0 + 1` and `odst == idst0`.
ALWI void binary_left_shift_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_left_shift_replay_(idst0)));
}

ALWI void binary_right_shift_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_right_shift_replay_(idst0)));
}

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU comparisons:
//   * INT32  : LT / GT / GE
//   * UINT16 : LT / GT
//   * FP32   : LT / GT / LE / GE / EQ  (branchless, pure-TTI re-implementation)
//
// NE is still routed through the non-replay path: it has no upstream demand
// for replay yet, and the existing sfpi `binary_comp_fp32_equal_mask<ne>`
// path is correct. INT32 LE and UINT16 GE/LE are similarly omitted.
//
// The INT32 and UINT16 bodies match `calculate_binary_comp_int32` and
// `calculate_binary_comp_uint16` instruction-for-instruction (with hardcoded
// `dst_index_in0 = 0`, `dst_index_in1 = 1`, `dst_index_out = 0`), so output
// is bit-identical to the upstream non-replay path. The FP32 ordered bodies
// (LT/GT/LE/GE) are branchless, pure-TTI re-implementations of the upstream
// sfpi `binary_comp_fp32_ordered_mask` predicate; the FP32 EQ body is a
// branchless re-implementation of `binary_comp_fp32_equal_mask<eq>`. All
// FP32 bodies produce an FP32 0.0f/1.0f result that matches the upstream
// FP32 comparison semantics for finite values, signed zeros, signed
// infinities and NaNs.
//
// Layout (offsets in dest rows, relative to current dst_reg base):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1])
//   result overwrites offset 0
// matching the kernel pairing `<op>_<dtype>_tile(i*2, i*2+1, i*2)`.
// ---------------------------------------------------------------------------

// LT / GT (INT32):
//   SFPLOAD x2 + SFPMOV x2 (copy A,B)
//   + SFPSHFT x2 (extract sign bits of A,B)
//   + SFPXOR (different-sign mask in LREG3)
//   + SFPSETCC + SFPIADD + SFPSHFT     (same-sign branch: sign of A-B / B-A)
//   + SFPCOMPC + SFPLOADI               (opposite-sign branch: result = 0)
//   + SFPSETCC + SFPLOADI + SFPENCC     (then result = 1 iff A < B / A > B)
//   + SFPSTORE + INCRWC
//   = 17 instructions
constexpr std::uint32_t SFPU_BINARY_LT_GT_INT32_REPLAY_LEN = 17;

// GE (INT32): LT body (15 instructions before SFPSTORE) + SFPLOADI + SFPXOR
//             (invert) + SFPSTORE + INCRWC = 19 instructions.
constexpr std::uint32_t SFPU_BINARY_GE_INT32_REPLAY_LEN = 19;

// LT / GT (UINT16):
//   SFPLOAD x2 + SFPIADD (A-B / B-A as int) + SFPSHFT (sign bit -> 0/1)
//   + SFPSTORE + INCRWC = 6 instructions.
constexpr std::uint32_t SFPU_BINARY_LT_GT_UINT16_REPLAY_LEN = 6;

// FP32 LT/GT (strict): 28 instructions.
//   SFPLOAD x2 + SFPABS x2                           (loads + |in0|, |in1|)
//   + SFPMAD + SFPNOP                                (signed diff)
//   + SFPABS + SFPSHFT                               (|diff|, strict sign-bit)
//   + SFPLOADI                                       (INF constant)
//   + SFPMOV + SFPIADD + SFPSHFT
//   + SFPIADD + SFPSHFT + SFPOR                      (nan_either)
//   + SFPMOV + SFPIADD + SFPSHFT                     (is_nan(|diff|))
//   + SFPIADD + SFPSHFT                              (is_zero_nonneg(|diff|))
//   + SFPOR + SFPOR                                  (force_zero)
//   + SFPNOT + SFPAND                                (result = strict & ~force_zero)
//   + SFPIADD + SFPAND                               (0/1 -> 0.0f/1.0f)
//   + SFPSTORE + INCRWC
constexpr std::uint32_t SFPU_BINARY_LT_GT_FP32_REPLAY_LEN = 28;

// FP32 LE/GE (non-strict): 35 instructions. Same prologue as LT/GT through
// `nan_either`, then a different epilogue:
//   + SFPIADD + SFPSHFT                              (zero_diff)
//   + SFPMOV + SFPXOR + SFPABS + SFPIADD + SFPSHFT   (bits_eq(in0,in1))
//   + SFPIADD + SFPABS + SFPIADD + SFPSHFT           (is_inf(|in0|))
//   + SFPAND                                         (same_signed_inf)
//   + SFPOR + SFPOR                                  (pre_result)
//   + SFPNOT + SFPAND                                (result = pre_result & ~nan_either)
//   + SFPIADD + SFPAND                               (0/1 -> 0.0f/1.0f)
//   + SFPSTORE + INCRWC
constexpr std::uint32_t SFPU_BINARY_LE_GE_FP32_REPLAY_LEN = 35;

// FP32 EQ: 26 instructions. Branchless re-implementation of
// `binary_comp_fp32_equal_mask`:
//   bits_eq    = (in0 ^ in1) == 0      (same bit pattern; covers ±finite, ±inf)
//   both_zero  = (|in0| | |in1|) == 0  (covers +0 vs -0 - different bits)
//   nan_either = is_nan(|in0|) | is_nan(|in1|)
//   result     = (bits_eq | both_zero) & ~nan_either
// Instruction breakdown:
//   SFPLOAD x2 + SFPABS x2 (FLOAT)                   (loads, |in0|, |in1|)
//   + SFPLOADI                                       (INF constant)
//   + SFPMOV + SFPXOR + SFPABS (INT) + SFPIADD + SFPSHFT  (bits_eq)
//   + SFPMOV + SFPOR + SFPIADD + SFPSHFT             (both_zero)
//   + SFPOR                                          (pre_eq = bits_eq | both_zero)
//   + SFPIADD + SFPSHFT + SFPIADD + SFPSHFT + SFPOR  (nan_either)
//   + SFPNOT + SFPAND                                (result = pre_eq & ~nan_either)
//   + SFPIADD + SFPAND                               (0/1 -> 0.0f/1.0f)
//   + SFPSTORE + INCRWC
constexpr std::uint32_t SFPU_BINARY_EQ_FP32_REPLAY_LEN = 26;

constexpr std::uint32_t SFPU_BINARY_COMP_DST_TILE_ROWS = 64;

#ifdef TRISC_MATH

// Shared per-tile loop. The existing mul / bitwise / shift paths each have
// their own near-identical copy of this loop; this helper collapses that
// duplication for the comparison ops by templating on the replay length.
// Behaviour is identical to those copies: idst0 seeds the dst_reg base, the
// recorded body is replayed 8 times per face for each of the 4 faces, and
// the face boundary advances dst_reg by 16 rows via two SETRWC pulses.
template <std::uint32_t REPLAY_LEN>
inline void _llk_math_eltwise_binary_sfpu_run_replay_(std::uint32_t idst0) {
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(idst0);

#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#pragma GCC unroll 0
        for (int d = 0; d < 8; d++) {
            lltt::replay(0, REPLAY_LEN);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    _llk_math_eltwise_binary_sfpu_done_();
}

// INT32 comparison body shared by LT, GT and GE up to the final write-back.
// After this prologue:
//   * For LT: LREG1 holds the result (0 or 1).
//   * For GT: LREG0 holds the result (0 or 1).
//   * For GE: LREG1 holds the LT result (caller XORs with 1 to invert).
//
// The same-sign branch uses an int32 subtract and the sign of the difference;
// the opposite-sign branch is decided directly from `sign(A) != sign(B)`,
// which is sufficient because two ints with different signs cannot subtract
// without overflow. `LREG2` carries `sign(A)` into the opposite-sign branch
// (used to decide which side of zero A is on). `LREG3` carries the
// different-sign predicate that gates the SETCC.
//
// IMPORTANT: A is ALWAYS loaded into LREG0 and B into LREG1 - same as the
// upstream `calculate_binary_comp_int32`. LT vs GT differ only in the
// SFPIADD direction (and thus which LREG holds the result and which mode
// the opposite-sign SETCC uses); they do NOT swap operand registers. Doing
// so would also flip `LREG2` to hold sign(B), breaking the opposite-sign
// branch which must test sign(A).
//
// `OutLREG` is the final result register: LREG1 for LT/GE, LREG0 for GT
// (matching where the same-sign sign-of-diff is left).
template <bool IsGt, std::uint8_t OutLREG>
inline void _record_sfpu_binary_lt_gt_int32_body_() {
    // Load A into LREG0 and B into LREG1 (same as LLK; never swapped).
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 1 * SFPU_BINARY_COMP_DST_TILE_ROWS);

    // Copy A,B into LREG2,LREG3 then logical-shift by -31 to extract sign bits.
    // Mod1 = 1 means immediate-shift mode with imm12 = -31 (sign-extended) =>
    // logical right shift by 31; result is 1 iff input had MSB = 1.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);               // LREG2 <- A
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);               // LREG3 <- B
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);  // LREG2 = sign(A)
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // LREG3 = sign(B)

    // LREG3 = sign(A) ^ sign(B): 0 if same sign, 1 if different signs.
    TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

    // Same-sign branch: int32 subtract and take the sign of the result.
    // imod = 6 makes SFPIADD compute `dst = src_c - dst` with operand-B 2's
    // complement on the way in.
    //   LT: SFPIADD(0, LREG0, LREG1, 6) -> LREG1 = LREG0 - LREG1 = A - B,
    //       sign of which is 1 iff A < B; result lives in LREG1.
    //   GT: SFPIADD(0, LREG1, LREG0, 6) -> LREG0 = LREG1 - LREG0 = B - A,
    //       sign of which is 1 iff A > B; result lives in LREG0.
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    if constexpr (IsGt) {
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
    } else {
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
    }

    // Opposite-sign branch (CC predicate flipped by SFPCOMPC). With different
    // signs the answer is determined entirely by sign(A) (still in LREG2):
    //   LT: A < B  iff  A is negative      -> 1 if sign(A) != 0, else 0
    //   GT: A > B  iff  A is non-negative  -> 1 if sign(A) == 0, else 0
    // Initialize the result LREG to 0, then SETCC on LREG2 to overwrite with 1
    // in the matching subset of lanes.
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPLOADI(OutLREG, sfpi::SFPLOADI_MOD0_USHORT, 0x00);
    constexpr std::uint8_t setcc_mode = IsGt ? sfpi::SFPSETCC_MOD1_LREG_EQ0 : sfpi::SFPSETCC_MOD1_LREG_NE0;
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, setcc_mode);
    TTI_SFPLOADI(OutLREG, sfpi::SFPLOADI_MOD0_USHORT, 0x01);
    TTI_SFPENCC(0, 0, 0, 0);
}

inline void _program_sfpu_binary_lt_int32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_INT32_REPLAY_LEN);

    _record_sfpu_binary_lt_gt_int32_body_</*IsGt=*/false, /*OutLREG=*/p_sfpu::LREG1>();

    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _program_sfpu_binary_gt_int32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_INT32_REPLAY_LEN);

    _record_sfpu_binary_lt_gt_int32_body_</*IsGt=*/true, /*OutLREG=*/p_sfpu::LREG0>();

    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _program_sfpu_binary_ge_int32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_GE_INT32_REPLAY_LEN);

    // Reuse the LT body and then invert the result.
    _record_sfpu_binary_lt_gt_int32_body_</*IsGt=*/false, /*OutLREG=*/p_sfpu::LREG1>();

    // GE = NOT LT: load 1 into a scratch register, then XOR with the LT
    // result (LREG1) to flip bit 0. LREG7 is used to match the upstream
    // sequence exactly (same opcode encoding -> same instruction word).
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0x01);
    TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

// UINT16 LT / GT body. UINT16 inputs are zero-extended at SFPLOAD time, so
// sign of `A - B` (interpreted as int32) directly tells us A < B without
// the same-sign / opposite-sign disambiguation INT32 needs.
//
// `SwapAB` selects GT: loads operands in (B, A) order so the same SFPIADD +
// SFPSHFT pair computes the correct sign.
template <bool SwapAB>
inline void _record_sfpu_binary_lt_gt_uint16_body_() {
    // SFPLOAD with mode LO16: read 16-bit value from dst, zero-extend to 32 bits.
    constexpr std::uint32_t off_a =
        SwapAB ? (1 * SFPU_BINARY_COMP_DST_TILE_ROWS) : (0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    constexpr std::uint32_t off_b =
        SwapAB ? (0 * SFPU_BINARY_COMP_DST_TILE_ROWS) : (1 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, off_a);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, off_b);

    // LREG1 = LREG0 - LREG1 (imod=6 selects `dst = src_c - dst`); MSB of the
    // 32-bit result is 1 iff "loaded A" < "loaded B" as unsigned 16-bit.
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);

    // Logical right shift by 31 -> 1 in lane iff the difference was negative.
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);

    TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _program_sfpu_binary_lt_uint16_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_UINT16_REPLAY_LEN);
    _record_sfpu_binary_lt_gt_uint16_body_</*SwapAB=*/false>();
}

inline void _program_sfpu_binary_gt_uint16_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_UINT16_REPLAY_LEN);
    _record_sfpu_binary_lt_gt_uint16_body_</*SwapAB=*/true>();
}

// ---------------------------------------------------------------------------
// Branchless FP32 ordered comparison (LT / GT / LE / GE), pure TTI macros.
//
// Per-lane 0/1 predicates are built from sign bits of FP/int subtracts and
// combined with bitwise ops; the final 0/1 mask is then converted to
// 0.0f/1.0f via:
//   (-mask) & LCONST_1   // 0 -> 0, 1 -> 0xFFFFFFFF -> 0x3F800000
// LREG layout during the body:
//   LREG0: in0 (FP32)         (kept for in0^in1 in LE/GE)
//   LREG1: in1 (FP32)         (kept for in0^in1 in LE/GE)
//   LREG2: diff -> nan_either / final result
//   LREG3: |in0| -> is_inf_in0 (LE/GE) / is_zero_nonneg(|diff|) (LT/GT)
//   LREG4: |in1| -> is_nan(|in1|) -> zero_diff (LE/GE) / is_nan(|diff|) (LT/GT)
//   LREG5: |diff| -> bits_eq scratch (LE/GE)
//   LREG6: strict mask -> pre_result (LE/GE)
//   LREG7: 0x7F800000 (kept alive)
template <SfpuType OP>
inline void _record_sfpu_binary_comp_fp32_body_() {
    static_assert(
        OP == SfpuType::lt || OP == SfpuType::gt || OP == SfpuType::le || OP == SfpuType::ge,
        "FP32 replay body supports only ordered comparisons (lt/gt/le/ge).");
    constexpr bool is_strict = (OP == SfpuType::lt || OP == SfpuType::gt);
    constexpr bool is_lt_le = (OP == SfpuType::lt || OP == SfpuType::le);

    // ----- 1. Loads + abs values --------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 1 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPABS_MOD1_FLOAT);  // |in0|
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG4, sfpi::SFPABS_MOD1_FLOAT);  // |in1|

    // ----- 2. diff = in0 - in1 (LT/LE) or in1 - in0 (GT/GE) -----------------
    // SFPMAD(a, b, c, dst, mod) -> dst = a*b + c.
    if constexpr (is_lt_le) {
        TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, 0);  // -in1 + in0
    } else {
        TTI_SFPMAD(p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0);  // -in0 + in1
    }
    TTI_SFPNOP;  // MAD latency

    // ----- 3. |diff| and strict sign-bit mask -------------------------------
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG5, sfpi::SFPABS_MOD1_FLOAT);  // |diff|
    // Logical right shift by 31 (mod=1 = SHIFT_IMM | LOGICAL): 0/1 sign bit.
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG6, 1);  // strict (0/1)

    // ----- 4. INF constant (reused 2 or 3 times) ----------------------------
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80);  // 0x7F800000

    // ----- 5. nan_either = is_nan(|in0|) | is_nan(|in1|), into LREG2 --------
    // is_nan(|x|) = sign_bit(INF - |x|) (1 iff |x| > INF).
    // SFPIADD mod=6 (ARG_2SCOMP|CC_NONE): dst = src_a - dst.
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG2, 6);              // INF - |in0|
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);  // is_nan(|in0|)
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, 6);              // INF - |in1| (overwrite |in1|)
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);  // is_nan(|in1|)
    TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);                // LREG2 = nan_either

    if constexpr (is_strict) {
        // ----- 6a. LT/GT: result = strict & ~(nan_either | is_nan(|diff|) | (|diff|==0)) --
        TTI_SFPMOV(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, 6);              // INF - |diff|
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);  // is_nan(|diff|)

        // is_zero_nonneg(|diff|) into LREG3 via (|diff| - 1) sign bit.
        // SFPIADD mod=5 (ARG_IMM|CC_NONE): dst = src_a + imm. imm12=0xFFF = -1.
        TTI_SFPIADD(0xFFF, p_sfpu::LREG5, p_sfpu::LREG3, 5);          // |diff| - 1
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // is_zero_nonneg

        TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);  // |= is_nan(|diff|)
        TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // |= is_zero_nonneg

        TTI_SFPNOT(0, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // ~force_zero
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);  // & strict
    } else {
        // ----- 6b. LE/GE: result = (strict | zero_diff | same_signed_inf) & ~nan_either ---
        TTI_SFPIADD(0xFFF, p_sfpu::LREG5, p_sfpu::LREG4, 5);          // |diff| - 1
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);  // zero_diff

        // bits_eq = is_zero_any(in0 ^ in1) into LREG5.
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
        TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);                      // in0 ^ in1
        TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPABS_MOD1_INT);  // |...| (2's-comp)
        TTI_SFPIADD(0xFFF, p_sfpu::LREG5, p_sfpu::LREG5, 5);                 // |...| - 1
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG5, p_sfpu::LREG5, 1);         // bits_eq

        // is_inf_in0 = is_zero_any(|in0| - INF) into LREG3 (overwrite |in0|).
        TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 6);                     // INF - |in0|
        TTI_SFPABS(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPABS_MOD1_INT);  // | INF - |in0| |
        TTI_SFPIADD(0xFFF, p_sfpu::LREG3, p_sfpu::LREG3, 5);                 // - 1
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);         // is_inf_in0

        // same_signed_inf = bits_eq & is_inf_in0 (in LREG5).
        TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG5, 0);

        // pre_result = strict | zero_diff | same_signed_inf in LREG6.
        TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG6, 0);  // |= zero_diff
        TTI_SFPOR(0, p_sfpu::LREG5, p_sfpu::LREG6, 0);  // |= same_signed_inf

        // result = pre_result & ~nan_either, into LREG2 (currently nan_either).
        TTI_SFPNOT(0, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // ~nan_either
        TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);  // & pre_result
    }

    // ----- 7. 0/1 -> 0.0f/1.0f ---------------------------------------------
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 6);  // -result (0 or -1)
    TTI_SFPAND(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);   // & 0x3F800000

    // ----- 8. Store + advance dst ------------------------------------------
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _program_sfpu_binary_lt_fp32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_FP32_REPLAY_LEN);
    _record_sfpu_binary_comp_fp32_body_<SfpuType::lt>();
}

inline void _program_sfpu_binary_gt_fp32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LT_GT_FP32_REPLAY_LEN);
    _record_sfpu_binary_comp_fp32_body_<SfpuType::gt>();
}

inline void _program_sfpu_binary_le_fp32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LE_GE_FP32_REPLAY_LEN);
    _record_sfpu_binary_comp_fp32_body_<SfpuType::le>();
}

inline void _program_sfpu_binary_ge_fp32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_LE_GE_FP32_REPLAY_LEN);
    _record_sfpu_binary_comp_fp32_body_<SfpuType::ge>();
}

// ---------------------------------------------------------------------------
// Branchless FP32 EQ comparison, pure TTI macros.
//
// Re-implementation of `binary_comp_fp32_equal_mask<eq>` without sfpi v_if:
//   bits_eq    = (in0 ^ in1) == 0      // same bit pattern (covers finite ==
//                                      // finite, +inf == +inf, -inf == -inf)
//   both_zero  = (|in0| | |in1|) == 0  // ±0 vs ±0 (different sign bits)
//   nan_either = is_nan(|in0|) | is_nan(|in1|)
//   result     = (bits_eq | both_zero) & ~nan_either
//
// Final 0/1 -> 0.0f/1.0f via:
//   (-mask) & LCONST_1   // 0 -> 0, 1 -> 0xFFFFFFFF -> 0x3F800000
//
// LREG layout during the body:
//   LREG0: in0 (kept for in0 ^ in1)
//   LREG1: in1 (kept for in0 ^ in1)
//   LREG2: bits_eq -> pre_eq -> final result
//   LREG3: |in0| -> nan_either
//   LREG4: |in1| -> is_nan(|in1|)
//   LREG6: |in0| | |in1| -> both_zero
//   LREG7: 0x7F800000 (kept alive)
inline void _record_sfpu_binary_eq_fp32_body_() {
    // ----- 1. Loads + abs values --------------------------------------------
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 1 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPABS_MOD1_FLOAT);  // |in0|
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG4, sfpi::SFPABS_MOD1_FLOAT);  // |in1|

    // ----- 2. INF constant (used by NaN check) -----------------------------
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, 0x7F80);  // 0x7F800000

    // ----- 3. bits_eq = is_zero_any(in0 ^ in1) into LREG2 ------------------
    // is_zero_any(x) = sign_bit(|x| - 1): 1 iff |x| == 0.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);                      // in0 ^ in1
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPABS_MOD1_INT);  // |...| (2's-comp)
    TTI_SFPIADD(0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, 5);                 // - 1
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);         // bits_eq

    // ----- 4. both_zero = is_zero_any(|in0| | |in1|) into LREG6 ------------
    // FLOAT abs already cleared the sign bits, so |in0|, |in1| are
    // non-negative magnitudes; their bitwise-OR is zero iff both are ±0.
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG6, 0);
    TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG6, 0);                // LREG6 = |in0| | |in1|
    TTI_SFPIADD(0xFFF, p_sfpu::LREG6, p_sfpu::LREG6, 5);          // - 1
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG6, p_sfpu::LREG6, 1);  // both_zero

    // ----- 5. pre_eq = bits_eq | both_zero (in LREG2) ----------------------
    TTI_SFPOR(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);

    // ----- 6. nan_either = is_nan(|in0|) | is_nan(|in1|), into LREG3 -------
    // is_nan(|x|) = sign_bit(INF - |x|) (1 iff |x| > INF).
    // SFPIADD mod=6 (ARG_2SCOMP|CC_NONE): dst = src_a - dst.
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG3, 6);              // INF - |in0|
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // is_nan(|in0|)
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, 6);              // INF - |in1|
    TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);  // is_nan(|in1|)
    TTI_SFPOR(0, p_sfpu::LREG4, p_sfpu::LREG3, 0);                // nan_either

    // ----- 7. result = pre_eq & ~nan_either (in LREG2) ---------------------
    TTI_SFPNOT(0, p_sfpu::LREG3, p_sfpu::LREG3, 0);  // ~nan_either
    TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);  // & pre_eq

    // ----- 8. 0/1 -> 0.0f/1.0f ---------------------------------------------
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 6);  // -result (0 or -1)
    TTI_SFPAND(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);   // & 0x3F800000

    // ----- 9. Store + advance dst ------------------------------------------
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0 * SFPU_BINARY_COMP_DST_TILE_ROWS);
    TTI_INCRWC(0, sfpi::SFP_DESTREG_STRIDE, 0, 0);
}

inline void _program_sfpu_binary_eq_fp32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_EQ_FP32_REPLAY_LEN);
    _record_sfpu_binary_eq_fp32_body_();
}

#endif  // TRISC_MATH

// One-shot replay-buffer programming helpers, one per (op, dtype) pair.
// Must be called once, after the matching `<op>_<dtype>_tile_init`, on the
// MATH thread. Each helper records into replay slot 0 with `lltt::NoExec`.
ALWI void lt_int32_init_replay() { MATH((_program_sfpu_binary_lt_int32_replay_())); }
ALWI void gt_int32_init_replay() { MATH((_program_sfpu_binary_gt_int32_replay_())); }
ALWI void ge_int32_init_replay() { MATH((_program_sfpu_binary_ge_int32_replay_())); }
ALWI void lt_uint16_init_replay() { MATH((_program_sfpu_binary_lt_uint16_replay_())); }
ALWI void gt_uint16_init_replay() { MATH((_program_sfpu_binary_gt_uint16_replay_())); }
ALWI void lt_fp32_init_replay() { MATH((_program_sfpu_binary_lt_fp32_replay_())); }
ALWI void gt_fp32_init_replay() { MATH((_program_sfpu_binary_gt_fp32_replay_())); }
ALWI void le_fp32_init_replay() { MATH((_program_sfpu_binary_le_fp32_replay_())); }
ALWI void ge_fp32_init_replay() { MATH((_program_sfpu_binary_ge_fp32_replay_())); }
ALWI void eq_fp32_init_replay() { MATH((_program_sfpu_binary_eq_fp32_replay_())); }

// Drop-in replacements for the corresponding non-replay tile functions.
// Per (op, dtype) the replay length is fixed, so each tile_replay wrapper
// instantiates `_llk_math_eltwise_binary_sfpu_run_replay_` with the matching
// `SFPU_BINARY_<...>_REPLAY_LEN`. Requires `idst1 == idst0 + 1` and
// `odst == idst0`, matching the kernel's `(i*2, i*2+1, i*2)` pairing.
ALWI void lt_int32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_INT32_REPLAY_LEN>(idst0)));
}

ALWI void gt_int32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_INT32_REPLAY_LEN>(idst0)));
}

ALWI void ge_int32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_GE_INT32_REPLAY_LEN>(idst0)));
}

ALWI void lt_uint16_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_UINT16_REPLAY_LEN>(idst0)));
}

ALWI void gt_uint16_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_UINT16_REPLAY_LEN>(idst0)));
}

ALWI void lt_fp32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_FP32_REPLAY_LEN>(idst0)));
}

ALWI void gt_fp32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LT_GT_FP32_REPLAY_LEN>(idst0)));
}

ALWI void le_fp32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LE_GE_FP32_REPLAY_LEN>(idst0)));
}

ALWI void ge_fp32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_LE_GE_FP32_REPLAY_LEN>(idst0)));
}

ALWI void eq_fp32_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    (void)idst1;
    (void)odst;
    MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_EQ_FP32_REPLAY_LEN>(idst0)));
}

// ---------------------------------------------------------------------------
// Replay-buffer based SFPU integer multiplication: INT32 / UINT32 / UINT16.
//
// Mirrors the FP multiply replay path (`mul_binary_tile_replay` above), but
// for the integer dispatch normally served by `mul_int_tile<DataFormat>`.
// The recorded body is the per-iteration sequence of the upstream:
//   * INT32 / UINT32: `ckernel::sfpu::mul_int32`
//                     (sfpu/ckernel_sfpu_mul_int32.h, 30 instructions/iter)
//   * UINT16        : DISABLE_SFPLOADMACRO branch of
//                     `ckernel::sfpu::_mul_int_`
//                     (sfpu/ckernel_sfpu_mul_int.h, 20 instructions/iter)
//
// Both bodies depend on programmable LREG12-14 constants set by the upstream
// `mul_int_tile_init<DataFormat>` (which in turn calls `mul_int32_init` /
// `_init_mul_int_`):
//   INT32 / UINT32: vConstIntPrgm0 = 0x7ff, vConstIntPrgm1 = -11,
//                   vConstFloatPrgm2 = 8388608.0
//   UINT16        : vConstIntPrgm0 = 0xff, vConstIntPrgm1 = -8
//
// For UINT16, the non-DISABLE_SFPLOADMACRO build of `_init_mul_int_` instead
// programs `vConstFloatPrgm1 = 8388608.0` (LREG13) for the macro pipeline,
// which would break the discrete SFPSHFT2 used by this body. The
// `_program_sfpu_binary_mul_uint16_replay_` helper therefore re-asserts
// `vConstIntPrgm1 = -8` before recording, so the same discrete body is
// correct in both build modes.
//
// Both bodies advance dst_reg by 2 rows via SFPSTORE with `ADDR_MOD_2`,
// which `mul_int_tile_init` configures with `.dest.incr = 2`. No explicit
// INCRWC is needed inside the recorded body. Eight replays per face leave
// dst_reg advanced by 16 rows (= one face); the surrounding loop then
// advances another 16 rows via two SETRWC pulses, matching the upstream
// non-replay layout.
//
// Layout (offsets in dest rows, relative to current dst_reg base):
//   operand A at offset 0   (DST[base])
//   operand B at offset 64  (DST[base + 1])
//   result overwrites offset 0
// matching the kernel pairing `mul_int_tile(i*2, i*2 + 1, i*2)`.
// ---------------------------------------------------------------------------

constexpr std::uint32_t SFPU_BINARY_MUL_INT_DST_TILE_ROWS = 64;

// INT32 / UINT32 (30 instructions/iter):
//   SFPLOAD (in0)
//   + SFPSHFT2 x2 (a1 raw, a2 raw)
//   + SFPAND + SFPCAST + SFPCAST + SFPAND + SFPCAST  (a0/a1/a2 -> fp32 chunks)
//   + SFPLOAD (in1)
//   + SFPSHFT2 x2 (b1 raw, b2 raw)
//   + SFPCAST                                       (b2 -> fp32)
//   + SFPMAD                                        (top  = a0*b2 + 2**23)
//   + SFPAND + SFPCAST                              (b1 -> fp32)
//   + SFPMAD                                        (top += a1*b1)
//   + SFPAND + SFPCAST                              (b0 -> fp32)
//   + SFPMAD                                        (top += a2*b0)
//   + SFPMAD                                        (mid = a0*b1 + 2**23)
//   + SFPMAD                                        (low = a0*b0 + 2**23)
//   + SFPMAD                                        (mid += a1*b0)
//   + SFPEXMAN x3                                   (extract integer mantissas)
//   + SFPSHFT x2                                    (top <<= 22, mid <<= 11)
//   + SFPIADD x2                                    (low += mid + top)
//   + SFPSTORE                                      (auto-advance via ADDR_MOD_2)
constexpr std::uint32_t SFPU_BINARY_MUL_INT32_REPLAY_LEN = 30;

// UINT16 discrete body (20 instructions/iter):
//   SFPLOAD (in0)
//   + SFPSHFT2 + SFPCAST + SFPAND + SFPCAST         (a1 / a0 -> fp32)
//   + SFPLOAD (in1)
//   + SFPSHFT2 + SFPCAST + SFPAND + SFPCAST         (b1 / b0 -> fp32)
//   + SFPMAD x3                                     (hi0=a0*b1, lo=a0*b0, hi1=a1*b0)
//   + SFP_STOCH_RND x3                              (round to u16)
//   + SFPIADD                                       (hi = hi0 + hi1)
//   + SFPSHFT                                       (hi <<= 8)
//   + SFPIADD                                       (lo += hi)
//   + SFPSTORE                                      (auto-advance via ADDR_MOD_2)
constexpr std::uint32_t SFPU_BINARY_MUL_UINT16_REPLAY_LEN = 20;

#ifdef TRISC_MATH

// One-iteration body of `ckernel::sfpu::mul_int32`. Inputs are split into
// 11-bit chunks (a0/a1/a2, b0/b1/b2), promoted to fp32 (lossless), multiplied
// via three FMA streams (top/mid/low), then reassembled into a 32-bit int via
// SFPEXMAN + SFPSHFT + SFPIADD. Bit-exact with the upstream non-replay path.
inline void _program_sfpu_binary_mul_int32_replay_() {
    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_INT32_REPLAY_LEN);

    // a -> a0 (LREG0), a1 (LREG2), a2 (LREG4) as fp32. LREG12 = 0x7ff,
    // LREG13 = -11 (logical right shift amount).
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // a >> 11
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // a >> 22
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG2, 0);  // a1 = (a >> 11) & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);     // a1 -> fp32
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);     // a2 -> fp32 (top 10 bits, no mask)
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // a0 = a & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);     // a0 -> fp32

    // b -> b0 (LREG1), b1 (LREG3), b2 (LREG5) as fp32.
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG5, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);  // b2 -> fp32

    // top = a0*b2 + 2**23 (LREG14 = 8388608.0). FMA writes back into LREG5.
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG14, p_sfpu::LREG5, 0);

    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG3, 0);  // b1 = (b >> 11) & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);     // b1 -> fp32
    // top += a1*b1
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // b0 = b & 0x7ff
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);     // b0 -> fp32
    // top += a2*b0
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG5, 0);

    // mid = a0*b1 + 2**23 (into LREG6)
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG6, 0);
    // low = a0*b0 + 2**23 (overwrites LREG0)
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG0, 0);
    // mid += a1*b0
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    // Extract the integer mantissas (PAD9 zero-extends bit 23 = 0).
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);

    // SFPSHFT mod=1 = immediate-shift, logical fill (matches the rest of this
    // file's `TTI_SFPSHFT(..., 1)` usage; the named sfpi constants for SHIFT
    // mode use a different encoding convention than the SFPSHFT op).
    TTI_SFPSHFT(22, 0, p_sfpu::LREG5, 1);                                      // top <<= 22
    TTI_SFPSHFT(11, 0, p_sfpu::LREG6, 1);                                      // mid <<= 11
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);  // low += mid
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);  // low += top

    // ADDR_MOD_2 (= entry 6 in the FPU table, configured by mul_int_tile_init
    // with .dest.incr = 2) auto-advances dst_reg by 2 rows after the store.
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
}

// One-iteration body of `ckernel::sfpu::_mul_int_` (DISABLE_SFPLOADMACRO
// branch). 16-bit inputs split into two 8-bit chunks each; partial products
// are computed in fp32 (each fits in 16 bits before rounding) and stitched
// back via SFP_STOCH_RND (FP32 -> u16) + SFPIADD + SFPSHFT.
inline void _program_sfpu_binary_mul_uint16_replay_() {
    // Force the discrete-body LREG13 constant. The upstream non-DISABLE init
    // sets vConstFloatPrgm1 = 8388608.0 (LREG13) for the macro pipeline,
    // which would break the SFPSHFT2 below; this re-asserts the value used
    // by the DISABLE_SFPLOADMACRO branch. Safe in both build modes.
    sfpi::vConstIntPrgm1 = -8;

    lltt::record<lltt::NoExec>(0, SFPU_BINARY_MUL_UINT16_REPLAY_LEN);

    // a -> a0 (LREG0), a1 (LREG2) as fp32. LREG12 = 0xff, LREG13 = -8.
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // a >> 8
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);                                               // a1 -> fp32
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                                            // a0 = a & 0xff
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);                                               // a0 -> fp32

    // b -> b0 (LREG1), b1 (LREG3) as fp32.
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::LO16, ADDR_MOD_3, 1 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG3, sfpi::SFPSHFT2_MOD1_SHFT_LREG);  // b >> 8
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);                                               // b1 -> fp32
    TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);                                            // b0 = b & 0xff
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);                                               // b0 -> fp32

    // hi0 = a0*b1; lo = a0*b0; hi1 = a1*b0 (all written back to operand reg).
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);

    // FP32 -> UINT16 (each partial < 2**16).
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);

    // hi = hi0 + hi1; hi <<= 8; lo += hi.
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_CC_NONE);
    // SFPSHFT mod=1 = immediate-shift, logical fill (see int32 body comment).
    TTI_SFPSHFT(8, 0, p_sfpu::LREG2, 1);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);

    // ADDR_MOD_2 auto-advance (see int32 body comment above). LO16 stores the
    // low 16 bits of LREG0 (which now holds the combined u16 product).
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_2, 0 * SFPU_BINARY_MUL_INT_DST_TILE_ROWS);
}

#endif  // TRISC_MATH

// One-shot replay-buffer programming helpers. Must be called once per kernel,
// after `mul_int_tile_init<data_format>()`, on the MATH thread. The init
// configures sfpi prgm constants (LREG12-14) and addr_mods (ADDR_MOD_6/7);
// the helper records the body into replay slot 0 with `lltt::NoExec`.
//
// `data_format` selects the body: UInt16 -> 8-bit-chunk discrete body,
// Int32 / UInt32 -> 11-bit-chunk fp32 body. UINT32 multiplication aliases
// INT32 because two's-complement multiplication of equal-width integers
// produces the same low 32 bits regardless of signedness.
template <DataFormat data_format>
ALWI void mul_int_binary_init_replay() {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    if constexpr (data_format == DataFormat::UInt16) {
        MATH((_program_sfpu_binary_mul_uint16_replay_()));
    } else {
        MATH((_program_sfpu_binary_mul_int32_replay_()));
    }
}

// Drop-in replacement for `mul_int_tile<data_format>(idst0, idst1, odst)`.
// The format-specific body is baked into the replay buffer by the matching
// `mul_int_binary_init_replay<data_format>()`. Requires `idst1 == idst0 + 1`
// and `odst == idst0`, matching the kernel's `(i*2, i*2 + 1, i*2)` pairing.
template <DataFormat data_format>
ALWI void mul_int_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    static_assert(
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32 || data_format == DataFormat::UInt16,
        "Unsupported data format for mul_int replay. Supported: Int32, UInt32, UInt16");
    (void)idst1;
    (void)odst;
    if constexpr (data_format == DataFormat::UInt16) {
        MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_MUL_UINT16_REPLAY_LEN>(idst0)));
    } else {
        MATH((_llk_math_eltwise_binary_sfpu_run_replay_<SFPU_BINARY_MUL_INT32_REPLAY_LEN>(idst0)));
    }
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
