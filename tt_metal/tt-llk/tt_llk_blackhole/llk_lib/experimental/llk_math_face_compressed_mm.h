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

// Dest rows from a face's accumulation to its split partial: consecutive MVMULs land in different rows so
// the FPU does not stall on the read-after-write. A CT == 1 tile is two faces wide, so dest ends up holding
// accumulations at rows 0 and 16 and their partials at 8 and 24, and finalize steps SrcB by this same amount
// to merge each pair.
constexpr std::int16_t _llk_math_face_compressed_mm_split_acc_partial_rows_ = 8;

// The math thread owns the replay buffer from replay_buf_offset up; the SFPU owns everything below it.
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_base_ = ckernel::math::replay_buf_offset;
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_slot_ = ckernel::REPLAY_BUF_SIZE - _llk_math_face_compressed_mm_replay_base_;

// The math code sequences: one character is one instruction, recorded into the replay buffer at init (see
// instr_for_code in _llk_math_face_compressed_mm_mop_config_ for the encodings). The letter names the
// addrmod that runs with it, so it says what moves after the face is handled:
//
//   n / N   noB    dest advances one face, both sources hold
//   i / I   incB   SrcB advances one activation row block, dest restarts
//   c / C   clrB   end of a K block: the SrcB and dest counters reset
//   r / R   revB   like incB, but dest rewinds to the other split-accumulation partial (ct_dim == 1)
//
// Lower case skips a face with no data, advancing the counters without multiplying; upper case issues the
// MVMUL. Each string is an overlap-packing of exactly the fragments its tables ask for, sized to the
// instructions this thread owns:
//
//   multi_seq (ct_dim >= 2) needs n i c N I C singly, plus ni Ni nI NI for the two-face midIncB only an
//   odd width emits. They pack into the first eight characters at 0, 2, 4 and 6, leaving the clrB endings
//   in the tail.
//
//   one_seq (ct_dim == 1) needs the I and C endings, nr for the MOP operands, and the six three-face
//   midRevB fragments -- {n,N}{r,R}{n,N} less nrn and nrN, which the MOP issues directly. They pack as
//   Nrn at 2, nRn at 4, nRN at 6, NrN at 8, NRN at 10, NRn at 12, and nr at 14.
static constexpr auto _llk_math_face_compressed_mm_multi_seq_ =
    ckernel::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("niNinINIncNcnCNC");
static constexpr auto _llk_math_face_compressed_mm_one_seq_ =
    ckernel::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("ICNrnRnRNrNRNRnr");

static constexpr auto _llk_math_face_compressed_mm_even_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m) -> std::uint32_t
    {
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // nnn case encoded in the mop
    });

static constexpr auto _llk_math_face_compressed_mm_even_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch (hdr)
        {
            case 0b00:
                return case_encode("n", face4); // noB
            case 0b01:
                return case_encode("i", face4); // endIncB
            case 0b10:
                return case_encode("c", face4); // endClrB
            default:
                return TT_OP_NOP; // invalid hdr
        }
    });

static constexpr auto _llk_math_face_compressed_mm_odd_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr      = m & 0b11;
        const std::uint32_t faces12  = (m >> 2) & 0b011;
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (hdr)
        {
            case 0b00:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // noB
            case 0b01:
                return case_encode("ni", faces12); // midIncB only two faces in this case
            case 0b10:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // endIncB
            case 0b11:
                return TT_OP_MOP(p_mop::MASK_LOOP, 2, faces123); // endClrB
            default:
                return 0;
        }
    });

static constexpr auto _llk_math_face_compressed_mm_odd_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr     = m & 0b11;
        const std::uint32_t faces34 = (m >> 4) & 0b11;
        const std::uint32_t face4   = (m >> 5) & 0b01;
        switch (hdr)
        {
            case 0b00:
                return case_encode("n", face4); // noB
            case 0b01:
                return TT_OP_MOP(p_mop::MASK_LOOP, 1, faces34); // midIncB the mop walks two faces
            case 0b10:
                return case_encode("i", face4); // endIncB
            case 0b11:
                return case_encode("c", face4); // endClrB
            default:
                return 0;
        }
    });

static constexpr auto _llk_math_face_compressed_mm_one_l1_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (faces123)
        {
            case 0b000:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 0); // "nrn" encoded in the mop
            case 0b100:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 1); // "nrN" encoded in the mop
            default:
                return case_encode("nrn", faces123); // always midRevB
        }
    });

static constexpr auto _llk_math_face_compressed_mm_one_l2_table_ = ckernel::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch ((hdr << 1) | face4)
        {
            case 0b000:
                return TT_OP_ZEROACC_ADDRMOD_ONLY(ADDR_MOD_1); // endIncB zero face4 (i)
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
 * @brief Program the five address modifiers the code sequence moves the sources and dest with.
 *
 * Every instruction in the sequence names one of these, and it decides what moves afterwards: ADDR_MOD_0
 * holds both sources and advances dest one face, ADDR_MOD_1 and ADDR_MOD_3 step SrcB to the next activation
 * row block, ADDR_MOD_2 clears the SrcB and dest counters, and ADDR_MOD_4 walks SrcB onto the partial that
 * finalize merges.
 *
 * @param face_r_dim: Activation rows per face, 1 or 8. This is the SrcB increment, so ADDR_MOD_1 and
 *                    ADDR_MOD_3 are programmed from one of two compile-time arms.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 */
inline void _llk_math_face_compressed_mm_addrmod_config_(const std::uint32_t face_r_dim)
{
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // noB: hold both sources, advance dest by one face

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
            .srcb = {.incr = 1, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1); // incB: advance SrcB one activation row block, reset dest to the beginning

        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 1, .clr = 0, .cr = 0},
            .dest = {.incr = -_llk_math_face_compressed_mm_split_acc_partial_rows_, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_3); // revB: advance SrcB one activation row block, rewind dest to the other partial
    }
    else
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 0},
        }
            .set(ADDR_MOD_1); // incB: advance SrcB one activation row block, reset dest to the beginning

        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = -_llk_math_face_compressed_mm_split_acc_partial_rows_, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_3); // revB: advance SrcB one activation row block, rewind dest to the other partial
    }

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = _llk_math_face_compressed_mm_split_acc_partial_rows_, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_4); // finalize: advance SrcB onto the partial MOVD2B parked there, hold dest
}

/**
 * @brief Record the math code sequence into the replay buffer and program the MOP that replays it.
 *
 * One instruction per character: lower case skips a face, advancing the counters with ZEROACC or SETRWC,
 * and upper case multiplies it in with MVMUL. ct_dim == 1 records the split-accumulation sequence, every
 * other width the multi-tile one.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8. Only 1 takes MVMUL's single-row SrcB broadcast,
 *                    which drops the SrcB alignment requirement from 8 rows to 1.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 * @ref _llk_math_face_compressed_mm_addrmod_config_ programs the addrmods these instructions name.
 */
template <std::uint32_t ct_dim>
inline void _llk_math_face_compressed_mm_mop_config_(const std::uint32_t face_r_dim)
{
    // MVMUL's instr_mod19 SrcB broadcast, using it for a single row reduces the SrcB alignment requirement from 8 to 1
    const std::uint32_t mvmul_single_row = (face_r_dim == 1) ? 1 : 0;

    // One instruction per code-sequence character. Lower case means no data for this face, so it is
    // skipped: the counters advance via ZEROACC or SETRWC without multiplying. Upper case multiplies it
    // in. The ADDR_MOD then picks what advances afterwards -- see _llk_math_face_compressed_mm_addrmod_config_.
    auto instr_for_code = [mvmul_single_row](char code)
    {
        switch (code)
        {
            case 'n':
                TTI_ZEROACC_ADDRMOD_ONLY(ADDR_MOD_0);
                break; // skip, noB
            case 'N':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_0, 0);
                break; // mul, noB
            case 'i':
                TTI_ZEROACC_ADDRMOD_ONLY(ADDR_MOD_1);
                break; // skip, incB
            case 'I':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_1, 0);
                break; // mul, incB
            case 'c':
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
                break; // skip, clrB
            case 'C':
                TT_MVMUL(p_setrwc::CLR_AB, mvmul_single_row, ADDR_MOD_2, 0);
                break; // mul, clrB
            case 'r':
                TTI_ZEROACC_ADDRMOD_ONLY(ADDR_MOD_3);
                break; // skip, revB
            case 'R':
                TT_MVMUL(p_setrwc::CLR_A, mvmul_single_row, ADDR_MOD_3, 0);
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

    constexpr std::uint32_t op0m = _llk_math_face_compressed_mm_multi_seq_.fragment("n");
    constexpr std::uint32_t op0M = _llk_math_face_compressed_mm_multi_seq_.fragment("N");
    constexpr std::uint32_t op0s = _llk_math_face_compressed_mm_one_seq_.fragment("nr");
    constexpr std::uint32_t op1s = _llk_math_face_compressed_mm_one_seq_.fragment("n");
    constexpr std::uint32_t op0S = _llk_math_face_compressed_mm_one_seq_.fragment("nr");
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

    _llk_math_face_compressed_mm_addrmod_config_(face_r_dim);
    _llk_math_face_compressed_mm_mop_config_<ct_dim>(face_r_dim);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Select the decode table pair for one output width.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @return The level-1 and level-2 tables, which expand a 6-bit meta into its two instructions.
 * @note Three regimes, because the header values a width can produce differ: an odd width can advance SrcB
 *       in the middle of a meta, which an even one never does, and ct_dim == 1 splits the accumulation on
 *       top of that.
 * @note Assumes ct_dim is in range. @ref _llk_math_face_compressed_mm_, its only caller, asserts it.
 */
template <std::uint32_t ct_dim>
inline constexpr std::array<const std::uint32_t*, 2> _llk_math_face_compressed_mm_tables_()
{
    if constexpr (ct_dim == 1)
    {
        return {{_llk_math_face_compressed_mm_one_l1_table_.data(), _llk_math_face_compressed_mm_one_l2_table_.data()}};
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
 * @note Call @ref _llk_math_face_compressed_mm_init_ first, with the same ct_dim.
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

    constexpr auto tables                   = _llk_math_face_compressed_mm_tables_<ct_dim>();
    constexpr const std::uint32_t* l1_table = tables[0];
    constexpr const std::uint32_t* l2_table = tables[1];

    for (std::uint32_t i = 0; i < full_iters; ++i)
    {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> (0 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx1 = (meta >> (1 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx2 = (meta >> (2 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx3 = (meta >> (3 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx4 = (meta >> (4 * meta_stride_bits)) & meta_index_mask;

        std::uint32_t data0 = l1_table[idx0];
        std::uint32_t data1 = l2_table[idx0];
        std::uint32_t data2 = l1_table[idx1];
        std::uint32_t data3 = l2_table[idx1];
        std::uint32_t data4 = l1_table[idx2];
        std::uint32_t data5 = l2_table[idx2];
        std::uint32_t data6 = l1_table[idx3];
        std::uint32_t data7 = l2_table[idx3];
        std::uint32_t data8 = l1_table[idx4];
        std::uint32_t data9 = l2_table[idx4];

        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        ckernel::instrn_buffer[0] = data2;
        ckernel::instrn_buffer[0] = data3;
        ckernel::instrn_buffer[0] = data4;
        ckernel::instrn_buffer[0] = data5;
        ckernel::instrn_buffer[0] = data6;
        ckernel::instrn_buffer[0] = data7;
        ckernel::instrn_buffer[0] = data8;
        ckernel::instrn_buffer[0] = data9;
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i)
    {
        std::uint32_t idx0        = meta & meta_index_mask;
        std::uint32_t data0       = l1_table[idx0];
        std::uint32_t data1       = l2_table[idx0];
        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        meta >>= meta_stride_bits;
    }

    if constexpr (ct_dim == 1 && finalize)
    {
        // Merge each face's split partial into its accumulation.
        constexpr std::int16_t partial = _llk_math_face_compressed_mm_split_acc_partial_rows_;

        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD); // wait for both operands
        // Move both partials into SrcB, face 0 at row 0 and face 1 at row 8.
        // ADDR_MOD_0 advances dest by one face between the two, and ADDR_MOD_2 resets it afterwards.
        TTI_MOVD2B(p_mov::DEST_NORM, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, partial);
        TTI_MOVD2B(p_mov::DEST_NORM, partial, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, partial);
        if (face_r_dim == 8)
        {
            // Rows 4-7 of each partial, walking the counters the same way as the pair above.
            TTI_MOVD2B(p_mov::DEST_NORM, 4, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, partial + 4);
            TTI_MOVD2B(p_mov::DEST_NORM, partial + 4, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, partial + 4);
        }
        // Add each partial into its accumulation, at dest 0 and 16, with ADDR_MOD_4 stepping SrcB from the
        // first to the second in between.
        TTI_ELWADD(p_elwise::CLR_NONE, p_elwise::DEST_ACCUM_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_4, 0);
        TTI_ELWADD(p_elwise::CLR_AB, p_elwise::DEST_ACCUM_EN, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 16);
    }
}
