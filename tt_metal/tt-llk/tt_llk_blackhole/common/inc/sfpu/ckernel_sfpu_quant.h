// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_load_config.h"
#include "llk_defs.h"
#include "lltt.h"

namespace ckernel::sfpu
{

// Replay-buffer slots for the per-iteration register-only compute bodies of
// the quant / requant / dequant kernels. The body is recorded once by each
// op's _init_{quant,requant,dequant}_int32_ and then replayed by every
// invocation of the matching _{quant,requant,dequant}_int32_ kernel. The
// two SIGN_MAGNITUDE_FORMAT variants of a given kernel safely share its
// slot - the init writes whichever variant it was templated for and the
// kernel uses the matching REPLAY_LEN. Distinct slots between kernels are
// required so a single compute kernel can mix all three ops without each
// init clobbering the others' recordings.
//
// Body content (see the inits for the exact emission order):
//   QUANT   ( 2s-comp, 5 ) : SFPMAD, SFPNOP, STOCH_RND, SFPCAST, SFPSETSGN
//   QUANT   (sign-magn, 3) : SFPMAD, SFPNOP, STOCH_RND
//   REQUANT ( 2s-comp, 8 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32), SFPMAD,
//                             SFPNOP, STOCH_RND, SFPCAST+SFPSETSGN(out)
//   REQUANT (sign-magn, 4) : SFPCAST(int->fp32), SFPMAD, SFPNOP, STOCH_RND
//   DEQUANT ( 2s-comp, 7 ) : SFPCAST+SFPSETSGN(in), SFPCAST(int->fp32), SFPADD,
//                             SFPNOP, SFPMUL, SFPNOP
//   DEQUANT (sign-magn, 5) : SFPCAST(int->fp32), SFPADD, SFPNOP, SFPMUL, SFPNOP
constexpr std::uint32_t QUANT_REPLAY_SLOT          = 0;
constexpr std::uint32_t QUANT_REPLAY_LEN_2S_COMP   = 5;
constexpr std::uint32_t QUANT_REPLAY_LEN_SIGN_MAGN = 3;

constexpr std::uint32_t REQUANT_REPLAY_SLOT          = QUANT_REPLAY_SLOT + QUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t REQUANT_REPLAY_LEN_2S_COMP   = 8;
constexpr std::uint32_t REQUANT_REPLAY_LEN_SIGN_MAGN = 4;

constexpr std::uint32_t DEQUANT_REPLAY_SLOT          = REQUANT_REPLAY_SLOT + REQUANT_REPLAY_LEN_2S_COMP;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_2S_COMP   = 7;
constexpr std::uint32_t DEQUANT_REPLAY_LEN_SIGN_MAGN = 5;

// Configure ADDR_MOD_6 with dest auto-increment by 2 so the per-iteration
// SFPSTORE walks dst_reg through the face's 4-row x 8-col blocks. Replaces
// sfpi::dst_reg++ in the kernel bodies and lets each loop be purely TTI-issued.
// Called once by each _init_{quant,requant,dequant}_int32_ since quant_int32
// isn't in the LLK init's "configure ADDR_MOD_6 with dest+=2" allow-list.
inline void _quant_kernels_configure_dest_incr_addrmod_()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _init_quant_int32_(const std::uint32_t zero_point)
{
    // One-time setup for _quant_int32_:
    //   1. load the fp32 zero-point constant into LREG2;
    //   2. program ADDR_MOD_6 with dest+=2 for the per-iteration SFPSTORE;
    //   3. record the register-only compute body into the SFPU replay buffer
    //      under QUANT_REPLAY_SLOT (NoExec - we don't want to issue SFPMAD/
    //      STOCH_RND against undefined LREG0/LREG1 contents at record time).
    // Subsequent _quant_int32_ calls replay the recorded body, shrinking the
    // unrolled binary from ~ITERATIONS*REPLAY_LEN body instructions down to
    // one replay invocation per iteration.
    _sfpu_load_imm32_(2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(QUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point)
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // SFPMAD has a 2-cycle write latency on LREG0 and SFP_STOCH_RND below
        // reads LREG0, so exactly one SFPU pipeline bubble is required. Use
        // TTI_SFPNOP (SFPU NOP, opcode 0x8f) rather than the generic Tensix
        // TTI_NOP (0x02) so the bubble lands in the SFPU pipe and so the
        // recorded body contains only SFPU-pipe opcodes (a hard requirement
        // for correct replay-buffer playback).
        TTI_SFPNOP;
        // fp32 -> int sign-magnitude. LREG9 holds 0.0 used as the zero descale.
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 3);
        if constexpr (!SIGN_MAGNITUDE_FORMAT)
        {
            // sign-magn -> 2's complement, then SETSGN to work around a
            // Blackhole RTL bug in the cast.
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _init_requant_int32_(const std::uint32_t zero_point)
{
    // One-time setup for _requant_int32_; see _init_quant_int32_ for the
    // record/replay rationale. Loads the zero point into LREG2, programs
    // ADDR_MOD_6 with dest+=2, then records the register-only compute into
    // REQUANT_REPLAY_SLOT (NoExec).
    _sfpu_load_imm32_(2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(REQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        if constexpr (!SIGN_MAGNITUDE_FORMAT)
        {
            // Input arrives in 2's-complement bits in LREG0; convert to
            // sign-magnitude (which the int->fp32 SFPCAST below expects)
            // via the same cast+SETSGN pair the output side uses to work
            // around the Blackhole RTL bug in the cast.
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(0, 0, 0);
        // D(LREG0) = LREG0 * LREG1 + LREG2 (zero point)
        TTI_SFPMAD(0, 1, 2, 0, 0);
        // SFPMAD has a 2-cycle write latency on LREG0 and STOCH_RND below
        // reads it next, so one SFPU pipeline bubble is required. SFPNOP
        // (not the Tensix TTI_NOP) so the bubble lands in the SFPU pipe and
        // the recorded body contains only SFPU-pipe opcodes.
        TTI_SFPNOP;
        // fp32 -> int sign-magnitude.
        TTI_SFP_STOCH_RND(0, 0, 9, 0, 0, 3);
        if constexpr (!SIGN_MAGNITUDE_FORMAT)
        {
            // sign-magnitude -> 2's complement (with the same SETSGN bug-fix).
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
    }
}

template <bool APPROXIMATION_MODE /*unused*/, bool SIGN_MAGNITUDE_FORMAT = false>
inline void _init_dequant_int32_(const std::uint32_t zero_point)
{
    // One-time setup for _dequant_int32_; see _init_quant_int32_ for the
    // record/replay rationale. The caller passes -zero_point (so the
    // recorded body computes (A + LREG2) * B = (A - zero_point) * B).
    _sfpu_load_imm32_(2, zero_point);
    _quant_kernels_configure_dest_incr_addrmod_();

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    lltt::record<lltt::NoExec>(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
    {
        if constexpr (!SIGN_MAGNITUDE_FORMAT)
        {
            // Input arrives in 2's-complement bits in LREG0; convert to
            // sign-magnitude so the int->fp32 SFPCAST below sees what it
            // expects. SETSGN is the workaround for the Blackhole cast bug.
            TTI_SFPCAST(0, 4, InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP);
            TTI_SFPSETSGN(0, 4, 0, 0);
        }
        // int32 sign-magnitude -> fp32.
        TTI_SFPCAST(0, 0, 0);
        // SFPADD = VA*VB + VC ; with LREG10 = 1.0 this collapses to A + LREG2.
        TTI_SFPADD(0, 10, 2, 0, 0);
        // SFPADD has a 2-cycle write latency on LREG0; SFPMUL reads it next.
        TTI_SFPNOP;
        // SFPMUL with LREG9 = 0.0 ignored as +C : LREG0 = (A + LREG2) * LREG1.
        TTI_SFPMUL(0, 1, 9, 0, 0);
        // SFPMUL has a 2-cycle write latency on LREG0; the SFPSTORE that
        // follows the replay reads it. Keep this NOP inside the recorded
        // body rather than relying on the implicit gap between the replay
        // completing and TT_SFPSTORE issuing.
        TTI_SFPNOP;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _quant_int32_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Operand A is input (fp32).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_quant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement).
    //
    // Tile layout in Dest: each tile occupies 64 dest-address units (4 faces
    // x 16 addr/face). Each SFPLOAD/SFPSTORE moves 4 dest rows x 8 SFPU lanes,
    // so advancing dst_reg by +2 between iterations walks one face's eight
    // 4-row x 8-col blocks (= one full call site, ITERATIONS == 8).
    //
    // The replay-buffer body at QUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_quant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? QUANT_REPLAY_LEN_SIGN_MAGN : QUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: inline TT_SFPLOADs (variable addresses can't live inside
    // the replay buffer because TT_* macros write to instrn_buffer[0]), replay
    // the recorded compute, then SFPSTORE under ADDR_MOD_6 which also auto-
    // advances dst_reg by 2 for the next iteration's loads.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(0, 3, ADDR_MOD_7, in0_off);                                 // operand A (fp32)
        TT_SFPLOAD(1, 3, ADDR_MOD_7, in1_off);                                 // operand B (fp32 scaler)
        lltt::replay(QUANT_REPLAY_SLOT, REPLAY_LEN);                           // MAD + SFPNOP + STOCH_RND + (CAST + SETSGN)
        TT_SFPSTORE(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off); // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _requant_int32_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Operand A is input to requant (int32, sign-magnitude or 2's complement bits).
    // Operand B is scaling factor (fp32).
    // LREG2 holds the zero-point constant (fp32) loaded by _init_requant_int32_.
    // Output is int32 scaled to int8 range (sign-magnitude or 2's-complement).
    //
    // The replay-buffer body at REQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_requant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? REQUANT_REPLAY_LEN_SIGN_MAGN : REQUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute
    // (the input cast doesn't touch LREG1 so reordering is safe), replay the
    // recorded body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off); // operand A (int32)
        TT_SFPLOAD(1, 3, ADDR_MOD_7, in1_off);                                // operand B (fp32 scaler)
        lltt::replay(REQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_6, out_off); // store + dst_reg += 2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool SIGN_MAGNITUDE_FORMAT>
inline void _dequant_int32_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    // Operand A[LREG0] is input to dequant (int32, sign-magnitude or 2's complement bits).
    // Operand B[LREG1] is scaling factor (fp32).
    // LREG2 holds the (negated) zero-point constant loaded by _init_dequant_int32_;
    // i.e. the formula computed is (A + LREG2) * B, which is (A - zero_point) * B
    // when the caller passes -zero_point through the init.
    //
    // The replay-buffer body at DEQUANT_REPLAY_SLOT and ADDR_MOD_6's dest+=2
    // slot are programmed by _init_dequant_int32_<APPROXIMATION_MODE,
    // SIGN_MAGNITUDE_FORMAT>, which must run before the first call here.
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr std::uint32_t REPLAY_LEN = SIGN_MAGNITUDE_FORMAT ? DEQUANT_REPLAY_LEN_SIGN_MAGN : DEQUANT_REPLAY_LEN_2S_COMP;

    const std::uint32_t in0_off = dst_index_in0 * dst_tile_size;
    const std::uint32_t in1_off = dst_index_in1 * dst_tile_size;
    const std::uint32_t out_off = dst_index_out * dst_tile_size;

    // Per iteration: hoist both TT_SFPLOADs ahead of the recorded compute,
    // replay the body, then SFPSTORE under ADDR_MOD_6 which auto-advances
    // dst_reg by 2 for the next iteration.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_7, in0_off); // operand A (int32)
        TT_SFPLOAD(1, 3, ADDR_MOD_7, in1_off);                                // operand B (fp32 scaler)
        lltt::replay(DEQUANT_REPLAY_SLOT, REPLAY_LEN);
        TT_SFPSTORE(0, 3, ADDR_MOD_6, out_off); // store fp32 + dst_reg += 2
    }
}

} // namespace ckernel::sfpu
