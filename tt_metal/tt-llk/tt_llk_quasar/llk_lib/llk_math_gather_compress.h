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

using namespace ckernel;

/**
 * @brief Program the address modifiers for the MOVB2D-gather MOP.
 *
 * The output columns are walked through the fidelity-phase counter. A 64-element (8-bit) vector steps SrcB back and
 * forth by 4 rows (ADDR_MOD_0/ADDR_MOD_1) with a fidelity step of 1; a 32-element vector keeps SrcB fixed with a
 * fidelity step of 2. ADDR_MOD_2 closes the vector by clearing the SrcB and fidelity counters.
 *
 * @param is_64_elem: True for a 64-element (Int8/UInt8) vector, false for a 32-element vector
 */
inline void _llk_math_gather_compress_addrmod_(const bool is_64_elem)
{
    if (is_64_elem)
    {
        constexpr std::uint8_t srcb_window_step = 4;

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = srcb_window_step, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = static_cast<std::uint8_t>(64 - srcb_window_step), .clr = 0, .cr = 0}, // wraparound to make -4
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 2, .clr = 0}}
            .set(ADDR_MOD_0);
    }

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_2);
}

/**
 * @brief Load the per-vector MOVB2D sequence into the replay buffer and program a MOP that replays it once per vector.
 *
 * Each vector is 8 MOVB2Ds (64-element) or 2 MOVB2Ds (32-element), followed by a CLEARDVALID that releases SrcB
 * back to the unpacker and pulses the FPU dest-dvalid, handing dest to the packer.
 *
 * @param is_64_elem: True for a 64-element (Int8/UInt8) vector, false for a 32-element vector
 * @param num_vectors: Number of vectors moved per @ref _llk_math_gather_compress_ call (MOP outer loop count)
 */
inline void _llk_math_gather_compress_mop_config_(const bool is_64_elem, const std::uint32_t num_vectors)
{
    LLK_ASSERT(num_vectors > 0, "num_vectors must be non-zero");

    const std::uint32_t MOP_OUTER_LOOP     = num_vectors;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    constexpr std::uint32_t REPLAY_BUF_LEN_32_EL = 3; // 2 MOVB2D + CLEARDVALID
    constexpr std::uint32_t REPLAY_BUF_LEN_64_EL = 9; // 8 MOVB2D + CLEARDVALID
    const std::uint32_t replay_buf_len           = is_64_elem ? REPLAY_BUF_LEN_64_EL : REPLAY_BUF_LEN_32_EL;

    if (is_64_elem)
    {
        load_replay_buf<0, REPLAY_BUF_LEN_64_EL>(
            []
            {
                for (std::uint32_t i = 0; i < 3; i++)
                {
                    TTI_MOVB2D(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0, 0);
                    TTI_MOVB2D(0, 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0, 0);
                }
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0, 0);
                TTI_MOVB2D(0, 0, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 0, 0);
                TTI_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, p_cleardvalid::FPU /*dest_pulse_last*/, 0);
            });
    }
    else
    {
        load_replay_buf<0, REPLAY_BUF_LEN_32_EL>(
            []
            {
                TTI_MOVB2D(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0, 0);
                TTI_MOVB2D(0, 0, ADDR_MOD_2, p_movd2b::MOV_4_ROWS, 0, 0);
                TTI_CLEARDVALID(p_cleardvalid::CLR_SRCB_VLD, 0, 0, 0, p_cleardvalid::FPU /*dest_pulse_last*/, 0);
            });
    }

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialize the FPU for a gather/compress: address modifiers, replay buffer/MOP, and the A/B/D counters.
 *
 * The vector size follows the source format: Int8/UInt8 vectors hold 64 elements, all others 32.
 *
 * Every vector sets the FPU dest-dvalid, handing dest to the packer, which clears it per vector
 * (@ref _llk_pack_gather_compress_init_). This is required because the gather/compress feature can only write to
 * rows 0-1 of dest.
 *
 * @param src_format: SrcB data format, as a DataFormat value
 * @param num_vectors: Number of vectors moved per @ref _llk_math_gather_compress_ call; must match the unpack and pack threads
 * @note Configure the ALU formats and the FPU -> PACK dest-dvalid chain first. Pair with
 * @ref _llk_unpack_gather_compress_init_ (T0) and @ref _llk_pack_gather_compress_init_ (T2).
 */
inline void _llk_math_gather_compress_init_(const std::uint32_t src_format, const std::uint32_t num_vectors = 1)
{
    static_assert(TENSIX_GATHER == 1, "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the quasar_gather variant");
    LLK_ASSERT(
        src_format == static_cast<std::uint32_t>(DataFormat::Int8) || src_format == static_cast<std::uint32_t>(DataFormat::UInt8) ||
            src_format == static_cast<std::uint32_t>(DataFormat::Float16_b),
        "Gather/compress supports Int8, UInt8, and Float16_b formats");

    const bool is_64_elem = (src_format == static_cast<std::uint32_t>(DataFormat::Int8)) || (src_format == static_cast<std::uint32_t>(DataFormat::UInt8));

    _llk_math_gather_compress_addrmod_(is_64_elem);
    _llk_math_gather_compress_mop_config_(is_64_elem, num_vectors);

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Move the gathered/compressed vectors from SrcB into rows 0-1 of dest-tile 0, running the MOP once.
 *
 * @note Call @ref _llk_math_gather_compress_init_ first. Pairs with @ref _llk_unpack_gather_compress_ (T0) and
 * @ref _llk_pack_gather_compress_ (T2).
 */
inline void _llk_math_gather_compress_()
{
    // Set the tile-address and run MOP
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
