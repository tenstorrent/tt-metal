// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_code_sequence.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"

using namespace ckernel;
using namespace ckernel::math;

// Dest rows one skip instruction steps: INCRWC's row increment is four bits wide, so this is half of a
// 16-row face slot. The multi-tile sequence issues two per skipped face; the ct_dim == 1 sequence walks
// dest in steps of this size (see the split layout below).
constexpr std::uint32_t _llk_math_face_compressed_mm_dest_step_rows_ = 8;

// ct_dim == 1 split accumulation: consecutive MVMULs land in different 8-row blocks so the FPU does not
// stall on the read-after-write. A meta covers two K rows of the two-face tile. Each face accumulates the
// even K row in its slot, rows 0 and 16, and the odd one _split_partial_rows_ above it, rows 8 and 24;
// finalize adds the odd partial into the slot, so the accumulation never leaves dest and keeps its full
// width. The dest counter walks 0, 8, 8, 16 over the meta's four faces (noB +8, revB +0, noB +8), which
// keeps every skip a single INCRWC, and the face-1 MVMULs (revB and the endings) write
// _split_face1_dst_ rows past the counter, so face 1's rows sit one face above face 0's.
constexpr std::uint32_t _llk_math_face_compressed_mm_split_partial_rows_ = _llk_math_face_compressed_mm_dest_step_rows_;
constexpr std::uint32_t _llk_math_face_compressed_mm_split_face1_dst_    = FACE_R_DIM - _llk_math_face_compressed_mm_dest_step_rows_;
// SrcB row where finalize parks face 1's odd partial; face 0's goes to row 0 and ADDR_MOD_4 steps between them.
constexpr std::uint32_t _llk_math_face_compressed_mm_finalize_face1_srcb_row_ = 8;

// The math thread owns the replay buffer from replay_buf_offset up; the SFPU owns everything below it.
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_base_ = ckernel::math::replay_buf_offset;
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_slot_ = ckernel::REPLAY_BUF_SIZE - _llk_math_face_compressed_mm_replay_base_;

// The math code sequences: one character is one instruction, recorded into the replay buffer at init (see
// instr_for_code in _llk_math_face_compressed_mm_mop_config_ for the encodings). Upper case multiplies a
// face in with MVMUL and the letter names the addrmod that runs with it, so it says what moves after the
// face is handled; lower case skips a face with no data, stepping the same counters with SETRWC or INCRWC:
//
//   h       skip: dest advances 8 rows, both sources hold -- half of noB's step, or all of it at ct_dim == 1
//   N   noB    dest advances one face (half a face at ct_dim == 1), both sources hold
//   i / I   incB   SrcB advances one activation row block, dest restarts
//   c / C   clrB   end of a K block: the SrcB and dest counters reset
//   r / R   revB   SrcB advances one activation row block, dest holds (ct_dim == 1, see the split layout)
//
// Skips are plain counter instructions, never ZEROACC: no row select is known to make Blackhole's 16-row
// ZEROACC a no-op, and the 0xff the Wormhole tree once carried as one was measured to zero the last face of
// the current dest half once the dest counter is past 256 rows, which at ct_dim == 16 is a live output face.
// Each string is an overlap-packing of exactly the fragments its tables ask for, sized to the instructions
// this thread owns:
//
//   multi_seq (ct_dim >= 2) needs hh i c N I C singly, plus hhi Ni hhI NI for the two-face midIncB only an
//   odd width emits. They pack into twelve characters.
//
//   one_seq (ct_dim == 1) needs the I and C endings, hr for the MOP operands, and the six three-face
//   midRevB fragments -- {h,N}{r,R}{h,N} less hrh and hrN, which the MOP issues directly. They pack as
//   Nrh at 2, hRh at 4, hRN at 6, NrN at 8, NRN at 10, NRh at 12, and hr at 14.
static constexpr auto _llk_math_face_compressed_mm_multi_seq_ =
    ckernel::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("hhiNihhINIcC");
static constexpr auto _llk_math_face_compressed_mm_one_seq_ =
    ckernel::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("ICNrhRhRNrNRNRhr");

static constexpr auto _llk_math_face_compressed_mm_even_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m) -> std::uint32_t
    {
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // all-skip case, hh x3, encoded in the mop
    });

static constexpr auto _llk_math_face_compressed_mm_even_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch (hdr)
        {
            case 0b00:
                return face4 ? find("N") : find("hh"); // noB
            case 0b01:
                return face4 ? find("I") : find("i"); // endIncB
            case 0b10:
                return face4 ? find("C") : find("c"); // endClrB
            default:
                return TT_OP_NOP; // invalid hdr
        }
    });

static constexpr auto _llk_math_face_compressed_mm_odd_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr      = m & 0b11;
        const std::uint32_t faces12  = (m >> 2) & 0b011;
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (hdr)
        {
            case 0b00:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // noB
            case 0b01:
                // midIncB only two faces in this case: face1 noB, face2 incB
                switch (faces12)
                {
                    case 0b00:
                        return find("hhi");
                    case 0b01:
                        return find("Ni");
                    case 0b10:
                        return find("hhI");
                    default:
                        return find("NI");
                }
            case 0b10:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // endIncB
            case 0b11:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // endClrB
            default:
                return 0;
        }
    });

static constexpr auto _llk_math_face_compressed_mm_odd_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr     = m & 0b11;
        const std::uint32_t faces34 = (m >> 4) & 0b11;
        const std::uint32_t face4   = (m >> 5) & 0b01;
        switch (hdr)
        {
            case 0b00:
                return face4 ? find("N") : find("hh"); // noB
            case 0b01:
                return TT_OP_MOP(p_mop::MASK_LOOP, 1, faces34); // midIncB the mop walks two faces
            case 0b10:
                return face4 ? find("I") : find("i"); // endIncB
            case 0b11:
                return face4 ? find("C") : find("c"); // endClrB
            default:
                return 0;
        }
    });

static constexpr auto _llk_math_face_compressed_mm_one_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        // Bit i of faces123 upper-cases face i of the midRevB pattern h r h.
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (faces123)
        {
            case 0b000:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 0); // "hrh" encoded in the mop
            case 0b001:
                return find("Nrh");
            case 0b010:
                return find("hRh");
            case 0b011:
                return find("NRh");
            case 0b100:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 1); // "hrN" encoded in the mop
            case 0b101:
                return find("NrN");
            case 0b110:
                return find("hRN");
            default:
                return find("NRN");
        }
    });

// The endIncB skip steps SrcB by the activation rows per face, 1 or 8, so this table comes in one copy per
// value; _llk_math_face_compressed_mm_tables_ picks the copy at runtime. The other skips in this regime
// live in the replay buffer, whose slot the one_seq fills to the last instruction.
template <std::uint32_t srcb_incr>
static constexpr auto _llk_math_face_compressed_mm_one_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch ((hdr << 1) | face4)
        {
            case 0b000:
                // endIncB zero face4 (i): SrcB += srcb_incr through its CR, dest and its CR back to 0
                return TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0 /*rwc_d*/, srcb_incr /*rwc_b*/, 0 /*rwc_a*/, p_setrwc::SET_BD);
            case 0b001:
                return find("I"); // endIncB data face4 (I)
            case 0b010:
                return TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); // endClrB zero face4 (c)
            case 0b011:
                return find("C"); // endClrB data face4 (C)
            default:
                return TT_OP_NOP; // invalid hdr
        }
    });

/**
 * @brief Program the address modifiers the code sequence moves the sources and dest with.
 *
 * Every MVMUL in the sequence names one of these, and it decides what moves afterwards: ADDR_MOD_0 holds
 * both sources and advances dest one face (half a face at ct_dim == 1), ADDR_MOD_1 and ADDR_MOD_3 step SrcB
 * to the next activation row block, ADDR_MOD_2 clears the SrcB and dest counters, ADDR_MOD_4 steps SrcB onto
 * the partial finalize merges and ADDR_MOD_5 holds everything for finalize's moves. Only the ct_dim == 1
 * sequence and its finalize name ADDR_MOD_3, 4 and 5, so only that width programs them. The SrcB steps go
 * through its CR, so the CR always equals the counter and the skip instructions, which have no addrmod of
 * their own, can step SrcB relative to it (SETRWC and INCRWC with SrcBCr).
 *
 * @tparam ct_dim: Output width in tiles. Only whether it is 1 matters: that regime walks dest in 8-row
 *                 steps and holds it on revB (see the split layout at the top of the file), every other
 *                 width walks it in 16-row faces.
 * @param face_r_dim: Activation rows per face, 1 or 8. This is the SrcB increment, so ADDR_MOD_1 and
 *                    ADDR_MOD_3 are programmed from one of two compile-time arms.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 */
template <std::uint32_t ct_dim>
inline void _llk_math_face_compressed_mm_addrmod_config_(const std::uint32_t face_r_dim)
{
    constexpr std::int16_t face_rows = ct_dim == 1 ? _llk_math_face_compressed_mm_dest_step_rows_ : FACE_R_DIM;

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = face_rows, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // noB: hold both sources, advance dest by one face (half a face, one split step, at ct_dim == 1)

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_2); // clrB: reset the SrcB and dest counters

    // One arm per M value: set() lowers to TTI_SETC16, so every addrmod field must be a compile-time constant.
    if (face_r_dim == 1)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 1, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1); // incB: advance SrcB one activation row block, reset dest to the beginning

        if constexpr (ct_dim == 1)
        {
            addr_mod_t {
                .srca = {.incr = 0, .clr = 0, .cr = 0},
                .srcb = {.incr = 1, .clr = 0, .cr = 1},
                .dest = {.incr = 0, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_3); // revB: advance SrcB one activation row block, hold dest (the face-1 MVMUL offsets its own rows)
        }
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1); // incB: advance SrcB one activation row block, reset dest to the beginning

        if constexpr (ct_dim == 1)
        {
            addr_mod_t {
                .srca = {.incr = 0, .clr = 0, .cr = 0},
                .srcb = {.incr = 8, .clr = 0, .cr = 1},
                .dest = {.incr = 0, .clr = 0, .cr = 0},
            }
                .set(ADDR_MOD_3); // revB: advance SrcB one activation row block, hold dest (the face-1 MVMUL offsets its own rows)
        }
    }

    if constexpr (ct_dim == 1)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = _llk_math_face_compressed_mm_finalize_face1_srcb_row_, .clr = 0, .cr = 1},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_4); // finalize: advance SrcB from face 0's parked partial to face 1's, hold dest

        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_5); // finalize: hold everything, the moves address their rows directly
    }
}

/**
 * @brief Record the math code sequence into the replay buffer and program the MOP that replays it.
 *
 * One instruction per character: lower case skips a face, stepping the counters with INCRWC or SETRWC,
 * and upper case multiplies it in with MVMUL. ct_dim == 1 records the split-accumulation sequence, every
 * other width the multi-tile one.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8. Only 1 takes MVMUL's single-row SrcB broadcast,
 *                    which drops the SrcB alignment requirement from 8 rows to 1. It is also the SrcB
 *                    increment the incB and revB skips step by.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 * @ref _llk_math_face_compressed_mm_addrmod_config_ programs the addrmods the MVMULs name.
 */
template <std::uint32_t ct_dim>
inline void _llk_math_face_compressed_mm_mop_config_(const std::uint32_t face_r_dim)
{
    // MVMUL's instr_mod19 SrcB broadcast, using it for a single row reduces the SrcB alignment requirement from 8 to 1
    const std::uint32_t mvmul_single_row = (face_r_dim == 1) ? 1 : 0;
    // SrcB rows per activation face: what ADDR_MOD_1 and ADDR_MOD_3 step, and so what the incB / revB skips step.
    const std::uint32_t srcb_incr = face_r_dim;
    // Dest rows the face-1 MVMULs of the ct_dim == 1 split (revB and both endings) write past the counter, so
    // their rows land one face above face 0's while the counter itself only steps 8 (see the split layout).
    constexpr std::uint32_t face1_dst = ct_dim == 1 ? _llk_math_face_compressed_mm_split_face1_dst_ : 0;

    // One instruction per code-sequence character. Lower case means no data for this face, so it is
    // skipped: the counters move exactly as the matching MVMUL's addrmod would move them, by INCRWC or
    // SETRWC, with the SrcB steps relative to its CR (which the addrmods keep equal to the counter). Upper
    // case multiplies it in and the ADDR_MOD picks what advances afterwards -- see
    // _llk_math_face_compressed_mm_addrmod_config_.
    auto instr_for_code = [mvmul_single_row, srcb_incr](char code)
    {
        switch (code)
        {
            case 'h':
                TTI_INCRWC(0 /*rwc_cr*/, _llk_math_face_compressed_mm_dest_step_rows_ /*rwc_d*/, 0 /*rwc_b*/, 0 /*rwc_a*/);
                break; // skip, half of noB's face step (all of it at ct_dim == 1)
            case 'N':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_0, 0);
                break; // mul, noB
            case 'i':
                TT_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0 /*rwc_d*/, srcb_incr /*rwc_b*/, 0 /*rwc_a*/, p_setrwc::SET_BD);
                break; // skip, incB: SrcB += srcb_incr through its CR, dest and its CR back to 0
            case 'I':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_1, face1_dst);
                break; // mul, incB
            case 'c':
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                break; // skip, clrB
            case 'C':
                TT_MVMUL(p_setrwc::CLR_AB, mvmul_single_row, ADDR_MOD_2, face1_dst);
                break; // mul, clrB
            case 'r':
                TT_INCRWC(p_setrwc::CR_B /*rwc_cr*/, 0 /*rwc_d*/, srcb_incr /*rwc_b*/, 0 /*rwc_a*/);
                break; // skip, revB: SrcB += srcb_incr through its CR, dest holds
            case 'R':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_3, face1_dst);
                break; // mul, revB
            default:
                LLK_ASSERT(false, "Invalid code for math instruction");
                break;
        }
    };

    if constexpr (ct_dim == 1)
    {
        _llk_math_face_compressed_mm_one_seq_.load(instr_for_code);
    }
    else
    {
        _llk_math_face_compressed_mm_multi_seq_.load(instr_for_code);
    }

    constexpr std::uint32_t op0m = _llk_math_face_compressed_mm_multi_seq_.fragment("hh");
    constexpr std::uint32_t op0M = _llk_math_face_compressed_mm_multi_seq_.fragment("N");
    constexpr std::uint32_t op0s = _llk_math_face_compressed_mm_one_seq_.fragment("hr");
    constexpr std::uint32_t op1s = _llk_math_face_compressed_mm_one_seq_.fragment("h");
    constexpr std::uint32_t op0S = _llk_math_face_compressed_mm_one_seq_.fragment("hr");
    constexpr std::uint32_t op1S = _llk_math_face_compressed_mm_one_seq_.fragment("N");

    // Every MOP operand must resolve to a real replay instruction; a 0 means its fragment
    // is not a substring of the code sequence (otherwise silent until it breaks at runtime).
    static_assert(
        op0m != 0 && op0M != 0 && op0s != 0 && op1s != 0 && op0S != 0 && op1S != 0,
        "face_compressed_mm (math): a MOP operand fragment is not a substring of the code sequence");

    ckernel_unpack_template tmp = ckernel_unpack_template(
        ct_dim == 1,               // unpackB    = only for CT==1
        false,                     // unpackHalo = false
        ct_dim == 1 ? op0s : op0m, // A
        TT_OP_NOP,                 // A1    (unused)
        TT_OP_NOP,                 // A2    (unused)
        TT_OP_NOP,                 // A3    (unused)
        ct_dim == 1 ? op0S : op0M, // skipA
        op1s,                      // B     (only for CT==1)
        op1S                       // skipB (only for CT==1)
    );
    tmp.program();
}

/**
 * @brief Configure the math thread for a face-granular compressed matmul.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8.
 * @note Call this before @ref _llk_math_face_compressed_mm_ with the same ct_dim. This thread has no
 *       uninit: it leaves behind only addrmods, the MOP and the replay buffer, which the next op's init
 *       reprograms anyway.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_face_compressed_mm_init_.
 */
template <std::uint32_t ct_dim = 1>
inline void _llk_math_face_compressed_mm_init_(const std::uint32_t face_r_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (math): ct_dim must be in [1, 16]");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 8, "face_compressed_mm (math): unsupported face_r_dim (expected 1 or 8)");

    _llk_math_face_compressed_mm_addrmod_config_<ct_dim>(face_r_dim);
    _llk_math_face_compressed_mm_mop_config_<ct_dim>(face_r_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Select the decode table pair for one output width.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8. Only ct_dim == 1 reads it: its level-2 table
 *                    carries the endIncB skip, whose SrcB step is this value, so it comes in two copies.
 * @return The level-1 and level-2 tables, which expand a 6-bit meta into its two instructions.
 * @note Three regimes, because the header values a width can produce differ: an odd width can advance SrcB
 *       in the middle of a meta, which an even one never does, and ct_dim == 1 splits the accumulation on
 *       top of that.
 * @note Assumes ct_dim and face_r_dim are in range. @ref _llk_math_face_compressed_mm_, its only caller,
 *       asserts them.
 */
template <std::uint32_t ct_dim>
inline std::array<const std::uint32_t*, 2> _llk_math_face_compressed_mm_tables_(const std::uint32_t face_r_dim)
{
    if constexpr (ct_dim == 1)
    {
        const std::uint32_t* l2_table = face_r_dim == 1 ? _llk_math_face_compressed_mm_one_l2_table_<1 /*srcb_incr*/>.data()
                                                        : _llk_math_face_compressed_mm_one_l2_table_<8 /*srcb_incr*/>.data();
        return {{_llk_math_face_compressed_mm_one_l1_table_.data(), l2_table}};
    }
    else if constexpr (ct_dim % 2 == 0)
    {
        return {{_llk_math_face_compressed_mm_even_l1_table_.data(), _llk_math_face_compressed_mm_even_l2_table_.data()}};
    }
    else
    {
        return {{_llk_math_face_compressed_mm_odd_l1_table_.data(), _llk_math_face_compressed_mm_odd_l2_table_.data()}};
    }
}

/**
 * @brief Multiply the unpacked activation by the compressed weight faces, accumulating into DST.
 *
 * Streams the math meta section, expanding each 6-bit meta through the two tables into the instructions
 * that multiply its four faces and step SrcB. For ct_dim == 1 the accumulation runs as two halves per face,
 * which finalize merges at the end.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam finalize: Merge the split-accumulation partials, values = <true/false>. Only ct_dim == 1 splits,
 *                   so it has no effect at any other width.
 * @param base_address_meta: L1 address of the meta buffer. The math metas are its first section, 6 bits
 *                           each at a 6-bit stride, five per word.
 * @param face_r_dim: Activation rows per face, 1 or 8.
 * @param dst_index: Tile index in DST that the result is written to.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref _llk_math_face_compressed_mm_init_ first, with the same ct_dim and the same face_r_dim: init
 *       programs the addrmods and the replayed skips from face_r_dim, and this call picks the endIncB skip
 *       table from it, so a mismatch mis-steps SrcB.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_face_compressed_mm_.
 */
template <std::uint32_t ct_dim = 1, bool finalize = true>
inline void _llk_math_face_compressed_mm_(
    const std::uint32_t base_address_meta, const std::uint32_t face_r_dim, const std::uint32_t dst_index, const std::uint32_t kt_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (math): ct_dim must be in [1, 16]");
    LLK_ASSERT(kt_dim >= 2 && kt_dim <= 256 && kt_dim % 2 == 0, "face_compressed_mm (math): kt_dim must be an even number in [2, 256]");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 8, "face_compressed_mm (math): unsupported face_r_dim (expected 1 or 8)");

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Geometry of the math meta section: one 6-bit meta per four faces, packed at its own width, so five
    // fit per word with two bits spare. The unpack index words share a bit between neighbours to fit six;
    // these do not overlap.
    constexpr std::uint32_t meta_index_bits  = 6;
    constexpr std::uint32_t meta_stride_bits = 6;
    constexpr std::uint32_t meta_index_mask  = (1u << meta_index_bits) - 1;
    constexpr std::uint32_t metas_per_word   = 5;
    static_assert((metas_per_word - 1) * meta_stride_bits + meta_index_bits <= 32, "five 6-bit metas at a 6-bit stride must fit one 32-bit word");

    const std::uint32_t iters      = kt_dim * ct_dim;
    const std::uint32_t full_iters = iters / metas_per_word; // whole meta words
    const std::uint32_t rem_iters  = iters % metas_per_word; // metas in the trailing word
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    const auto tables                   = _llk_math_face_compressed_mm_tables_<ct_dim>(face_r_dim);
    const std::uint32_t* const l1_table = tables[0];
    const std::uint32_t* const l2_table = tables[1];

    for (std::uint32_t i = 0; i < full_iters; ++i)
    {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> (0 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx1 = (meta >> (1 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx2 = (meta >> (2 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx3 = (meta >> (3 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx4 = (meta >> (4 * meta_stride_bits)) & meta_index_mask;

        TT_INSN(l1_table[idx0]);
        TT_INSN(l2_table[idx0]);
        TT_INSN(l1_table[idx1]);
        TT_INSN(l2_table[idx1]);
        TT_INSN(l1_table[idx2]);
        TT_INSN(l2_table[idx2]);
        TT_INSN(l1_table[idx3]);
        TT_INSN(l2_table[idx3]);
        TT_INSN(l1_table[idx4]);
        TT_INSN(l2_table[idx4]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i)
    {
        std::uint32_t idx0        = meta & meta_index_mask;
        TT_INSN(l1_table[idx0]);
        TT_INSN(l2_table[idx0]);
        meta >>= meta_stride_bits;
    }

    if constexpr (ct_dim == 1 && finalize)
    {
        // Add each face's odd partial, rows 8 and 24, onto its even accumulation, rows 0 and 16: the partial goes
        // through SrcB, the accumulation stays in dest at its full width. The sequence left both counters at 0,
        // ADDR_MOD_5 holds them, and the unpack side left SrcA and SrcB zeroed and valid.
        constexpr std::uint32_t partial    = _llk_math_face_compressed_mm_split_partial_rows_;
        constexpr std::uint32_t face1_slot = FACE_R_DIM;
        constexpr std::uint32_t face1_srcb = _llk_math_face_compressed_mm_finalize_face1_srcb_row_;

        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD); // wait for both operands
        // Park face 0's odd partial at SrcB row 0 and face 1's at SrcB row face1_srcb.
        TTI_MOVD2B(p_mov::DEST_NORM, 0 /*SrcB row*/, ADDR_MOD_5, p_movd2b::MOV_4_ROWS, partial);
        TTI_MOVD2B(p_mov::DEST_NORM, face1_srcb /*SrcB row*/, ADDR_MOD_5, p_movd2b::MOV_4_ROWS, face1_slot + partial);
        if (face_r_dim == 8)
        {
            // Rows 4-7 of each partial.
            TTI_MOVD2B(p_mov::DEST_NORM, 4 /*SrcB row*/, ADDR_MOD_5, p_movd2b::MOV_4_ROWS, partial + 4);
            TTI_MOVD2B(p_mov::DEST_NORM, face1_srcb + 4 /*SrcB row*/, ADDR_MOD_5, p_movd2b::MOV_4_ROWS, face1_slot + partial + 4);
        }
        // Accumulate each partial onto its face; SrcA is zero. ADDR_MOD_4 steps SrcB from face 0's partial to
        // face 1's in between, ADDR_MOD_2 resets the counters afterwards.
        TTI_ELWADD(p_elwise::CLR_NONE, p_elwise::DEST_ACCUM_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_4, 0);
        TTI_ELWADD(p_elwise::CLR_AB, p_elwise::DEST_ACCUM_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, face1_slot);
    }
}
