// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_common.h"

#if TENSIX_GATHER != 1
#error "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the saturn variant"
#endif

/**
 * @brief Program the address modifiers for the MOVB2D-gather MOP.
 *
 * The output columns are walked through the fidelity-phase counter. A 64-element (8-bit) vector steps SrcB back and
 * forth by 4 rows (ckernel::ADDR_MOD_0/ckernel::ADDR_MOD_1) with a fidelity step of 1; a 32-element vector keeps SrcB fixed with a
 * fidelity step of 2. ckernel::ADDR_MOD_2 closes the vector by clearing the SrcB and fidelity counters.
 *
 * @param is_64_elem: True for a 64-element (Int8/UInt8) vector, false for a 32-element vector
 */
inline void _llk_math_gather_compress_addrmod_(const bool is_64_elem)
{
    if (is_64_elem)
    {
        constexpr std::uint8_t srcb_window_step = 4;

        ckernel::addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = srcb_window_step, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ckernel::ADDR_MOD_0);

        ckernel::addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = static_cast<std::uint8_t>(64 - srcb_window_step), .clr = 0, .cr = 0}, // wraparound to make -4
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ckernel::ADDR_MOD_1);
    }
    else
    {
        ckernel::addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 2, .clr = 0}}
            .set(ckernel::ADDR_MOD_0);
    }

    ckernel::addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ckernel::ADDR_MOD_2);
}

/**
 * @brief Load the MOVB2D sequence for one vector into the replay buffer and program a MOP that replays it once.
 *
 * A vector is 8 MOVB2Ds (64-element) or 2 MOVB2Ds (32-element), followed by a CLEARDVALID that releases SrcB
 * back to the unpacker.
 *
 * @param is_64_elem: True for a 64-element (Int8/UInt8) vector, false for a 32-element vector
 */
inline void _llk_math_gather_compress_mop_config_(const bool is_64_elem)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    constexpr std::uint32_t REPLAY_BUF_LEN_32_EL = 3; // 2 MOVB2D + CLEARDVALID
    constexpr std::uint32_t REPLAY_BUF_LEN_64_EL = 9; // 8 MOVB2D + CLEARDVALID
    const std::uint32_t replay_buf_len           = is_64_elem ? REPLAY_BUF_LEN_64_EL : REPLAY_BUF_LEN_32_EL;

    if (is_64_elem)
    {
        ckernel::load_replay_buf<0, REPLAY_BUF_LEN_64_EL>(
            []
            {
                for (std::uint32_t i = 0; i < 3; i++)
                {
                    TTI_MOVB2D(
                        ckernel::p_mov::DEST_NORM,
                        0 /*src*/,
                        ckernel::ADDR_MOD_0,
                        ckernel::p_mov_src_to_dest::MOV_4_ROWS,
                        ckernel::p_movb2d::BCAST_OFF,
                        0 /*dst*/);
                    TTI_MOVB2D(
                        ckernel::p_mov::DEST_NORM,
                        0 /*src*/,
                        ckernel::ADDR_MOD_1,
                        ckernel::p_mov_src_to_dest::MOV_4_ROWS,
                        ckernel::p_movb2d::BCAST_OFF,
                        0 /*dst*/);
                }
                TTI_MOVB2D(
                    ckernel::p_mov::DEST_NORM, 0 /*src*/, ckernel::ADDR_MOD_0, ckernel::p_mov_src_to_dest::MOV_4_ROWS, ckernel::p_movb2d::BCAST_OFF, 0 /*dst*/);
                TTI_MOVB2D(
                    ckernel::p_mov::DEST_NORM, 0 /*src*/, ckernel::ADDR_MOD_2, ckernel::p_mov_src_to_dest::MOV_4_ROWS, ckernel::p_movb2d::BCAST_OFF, 0 /*dst*/);
                TTI_CLEARDVALID(
                    ckernel::p_cleardvalid::CLR_SRCB_VLD,
                    ckernel::p_cleardvalid::CLR_NONE /*cleardvalid_S*/,
                    0 /*dest_dvalid_reset*/,
                    0 /*dest_dvalid_client_bank_reset*/,
                    0 /*dest_pulse_last*/,
                    0 /*reset*/);
            });
    }
    else
    {
        ckernel::load_replay_buf<0, REPLAY_BUF_LEN_32_EL>(
            []
            {
                TTI_MOVB2D(
                    ckernel::p_mov::DEST_NORM, 0 /*src*/, ckernel::ADDR_MOD_0, ckernel::p_mov_src_to_dest::MOV_4_ROWS, ckernel::p_movb2d::BCAST_OFF, 0 /*dst*/);
                TTI_MOVB2D(
                    ckernel::p_mov::DEST_NORM, 0 /*src*/, ckernel::ADDR_MOD_2, ckernel::p_mov_src_to_dest::MOV_4_ROWS, ckernel::p_movb2d::BCAST_OFF, 0 /*dst*/);
                TTI_CLEARDVALID(
                    ckernel::p_cleardvalid::CLR_SRCB_VLD,
                    ckernel::p_cleardvalid::CLR_NONE /*cleardvalid_S*/,
                    0 /*dest_dvalid_reset*/,
                    0 /*dest_dvalid_client_bank_reset*/,
                    0 /*dest_pulse_last*/,
                    0 /*reset*/);
            });
    }

    ckernel::ckernel_template temp(
        MOP_OUTER_LOOP,
        MOP_INNER_LOOP,
        TT_OP_REPLAY(0 /*start_idx*/, replay_buf_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/));
    temp.program_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Initialize the FPU for a gather/compress: address modifiers, replay buffer/MOP, and the A/B/D counters.
 *
 * The vector size follows the source format: Int8/UInt8 vectors hold 64 elements, all others 32.
 *
 * The gather/compress feature can only write to rows 0-1 (16-bit formats) or 0-3 (8-bit formats) of dest, so each
 * @ref _llk_math_gather_compress_ call moves one vector, and every vector is its own dest section: close the section
 * after each call, the same as after any other math op.
 *
 * Also disables Src zero-flag detection (ALU_ACC_CTRL_Zero_Flag_disabled_src), so a zero index in the exponent slot
 * is not read as a zero flag by the MOVB2Ds.
 *
 * @tparam EN_32BIT_DEST: Must be false; the MOVB2Ds write only the high 16 bits of a 32-bit dest datum
 * @param src_format: SrcB data format
 * @note Configure the ALU formats first. Pair with
 * @ref _llk_unpack_gather_compress_init_ (T0) and @ref _llk_pack_gather_compress_init_ (T2). Call
 * @ref _llk_math_gather_compress_uninit_ before the next math op's init.
 */
template <bool EN_32BIT_DEST>
inline void _llk_math_gather_compress_init_(const DataFormat src_format)
{
    static_assert(!EN_32BIT_DEST, "Gather/compress supports 16-bit dest only");
    LLK_ASSERT(
        src_format == DataFormat::Int8 || src_format == DataFormat::UInt8 || src_format == DataFormat::Float16_b,
        "Gather/compress supports Int8, UInt8, and Float16_b formats");

    const bool is_64_elem = (src_format == DataFormat::Int8) || (src_format == DataFormat::UInt8);

    _llk_math_gather_compress_addrmod_(is_64_elem);
    _llk_math_gather_compress_mop_config_(is_64_elem);

    TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, 0, ckernel::p_stall::WAIT_SFPU, ckernel::p_stall::MATH);
    ckernel::cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 1); // Keeps a zero index in the exponent slot from being reinterpreted as a zero-flag.

    ckernel::math::_reset_counters_<ckernel::p_setrwc::SET_ABD_F>();
}

/**
 * @brief Move one gathered/compressed vector from SrcB into dest tile 0.
 *
 * @note Call @ref _llk_math_gather_compress_init_ first. Call once per vector, each in its own dest section. Pairs
 * with @ref _llk_unpack_gather_compress_ (T0) and @ref _llk_pack_gather_compress_ (T2).
 */
inline void _llk_math_gather_compress_()
{
    // Set the tile-address and run MOP
    ckernel::math::_set_dst_write_addr_<ckernel::trisc::DstTileShape::Tile32x32>(0);
    ckernel::ckernel_template::run_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Re-enable Src zero-flag detection (ALU_ACC_CTRL_Zero_Flag_disabled_src), disabled by
 * @ref _llk_math_gather_compress_init_.
 *
 * Waits for math to finish first, so the last MOVB2Ds still run with zero-flag detection disabled.
 *
 * @note Call after the last @ref _llk_math_gather_compress_, then call the next math op's init.
 */
inline void _llk_math_gather_compress_uninit_()
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, 0, ckernel::p_stall::WAIT_SFPU, ckernel::p_stall::MATH);
    ckernel::cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 0);
}
