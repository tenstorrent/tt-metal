// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_pack_common.h"
#include "llk_unpack_common.h"

using namespace ckernel;

/**
 * @brief Configure the auto loop registers for unpacking srcs
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @tparam INSTRN_LOOP_COUNT: The number of times to loop the instructions in the auto-loop
 */
template <std::uint8_t INSTRN_COUNT, std::uint8_t INSTRN_LOOP_COUNT>
inline void _llk_unpack_srcs_config_()
{
    static_assert(INSTRN_LOOP_COUNT <= 256, "INSTRN_LOOP_COUNT maximum is 256");
    static_assert(INSTRN_COUNT <= 4, "INSTRN_COUNT maximum is 4");
    cfg_rmw(THCON_UNPACKER2_REG0_INSTRN_LOOP_COUNT_RMW, INSTRN_LOOP_COUNT - 1);
    cfg_rmw(THCON_UNPACKER2_REG0_INSTRN_COUNT_RMW, INSTRN_COUNT - 1);
}

template <std::uint8_t INSTRN_COUNT, std::uint8_t INSTRN_LOOP_COUNT>
inline void _llk_pack_srcs_config_()
{
    static_assert(INSTRN_LOOP_COUNT <= 256, "INSTRN_LOOP_COUNT maximum is 256");
    static_assert(INSTRN_COUNT <= 4, "INSTRN_COUNT maximum is 4");
    cfg_rmw(THCON_PACKER1_REG0_INSTRN_LOOP_COUNT_RMW, INSTRN_LOOP_COUNT - 1);
    cfg_rmw(THCON_PACKER1_REG0_INSTRN_COUNT_RMW, INSTRN_COUNT - 1);
}

/**
 * @brief Unpacks a single operand (unary) to SrcS. Each SrcS slice is only 8*16*16bit.
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0-31
 * @param start_l1_tile_idx: The tile index into the l1 input buffer
 * that unpacker can start unpacking from
 */
template <std::uint8_t INSTRN_COUNT>
inline void _llk_unpack_srcs_unary_(
    const std::uint8_t buf_desc_id, const std::uint32_t start_l1_tile_idx

)
{
    // Set src (l1 input) counter to face index offset
    // Tile can only be a maximum of 8x16 rows for 16 bit, 4*16 for 32 bit
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, start_l1_tile_idx);

    for (std::uint32_t i = 0; i < INSTRN_COUNT; i++)
    {
        TT_UNPACR2_TILE_INC(0b0, 0b1 /*SrcS l1 increment*/, buf_desc_id, 0b1 /*Set dvalid*/);
    }
}

/**
 * @brief Unpacks two operands (binary) to SrcS from separate L1 buffers.
 * Both operands share the same L1 source counter. The first operand increments
 * the SrcS tile counter (to advance to the next slice) without incrementing L1
 * or setting dvalid. The second operand increments L1 and sets dvalid (which
 * also resets the SrcS tile counter).
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @param buf_desc_id_0: Buffer descriptor ID for the first operand (L1 buffer A)
 * @param buf_desc_id_1: Buffer descriptor ID for the second operand (L1 buffer B)
 * @param start_l1_tile_idx: The tile index into the l1 input buffer
 * that unpacker can start unpacking from (shared across both operands)
 */
template <std::uint8_t INSTRN_COUNT>
inline void _llk_unpack_srcs_binary_(
    const std::uint8_t buf_desc_id_0, const std::uint8_t buf_desc_id_1, const std::uint32_t start_l1_tile_idx

)
{
    // Set src (l1 input) counter to face index offset (shared for both operands)
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, start_l1_tile_idx);

    for (std::uint32_t i = 0; i < INSTRN_COUNT; i++)
    {
        // First operand: increment SrcS tile counter (advance to next slice),
        // no L1 increment, no dvalid
        TT_UNPACR2_TILE_INC(0b1 /*SrcS tile inc*/, 0b0 /*no L1 inc*/, buf_desc_id_0, 0b0 /*no dvalid*/);
        // Second operand: no SrcS tile counter increment,
        // L1 increment, set dvalid (also resets SrcS tile counter)
        TT_UNPACR2_TILE_INC(0b0 /*no SrcS tile inc*/, 0b1 /*L1 inc*/, buf_desc_id_1, 0b1 /*Set dvalid*/);
    }
}

/**
 * @brief Packs tiles from SrcS to L1, Each SrcS slice is only 8*16*16bit
 * @tparam INSTRN_COUNT: The number of instructions to place in the auto-loop, auto-loop will then
 * be looped by INSTRN_LOOP_COUNT set in the hw_config
 * @param buf_desc_id: The buffer descriptor ID where the buffer information is
 * stored in the buffer descriptor table, values = 0-31
 * @param start_l1_tile_idx: The tile index into the l1 output buffer
 * that packer can start packing to
 */
template <std::uint8_t INSTRN_COUNT>
inline void _llk_pack_srcs_(
    const std::uint8_t buf_desc_id, const std::uint32_t start_l1_tile_idx

)
{
    // Set dst (l1 output) counter to face index offset
    // Tile can only be a maximum of 8x16 rows for 16 bit, 4*16 for 32 bit
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK1, start_l1_tile_idx);

    for (std::uint32_t i = 0; i < INSTRN_COUNT; i++)
    {
        TT_PACR1_TILE_INC(0b1 /*DstS l1 increment*/, 0b0 /*SrcS l1 increment*/, buf_desc_id, 0b1 /*Set dvalid*/);
    }
}
