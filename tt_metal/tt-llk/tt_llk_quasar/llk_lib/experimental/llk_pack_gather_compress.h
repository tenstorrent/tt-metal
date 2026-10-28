// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

#if TENSIX_GATHER != 1
#error "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the saturn variant"
#endif

/**
 * @brief Program a MOP that issues one PACR0_TILE_INC, packing one vector from dest tile 0 into one L1 tile.
 *
 * The PACR leaves the dest dvalid set: the caller closes the dest section after each vector.
 *
 * @param buf_desc_id: Buffer descriptor of the output, one vector per L1 tile
 */
inline void _llk_pack_gather_compress_mop_config_(const std::uint32_t buf_desc_id)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t pack_vector_instrn = TT_OP_PACR0_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*ClrDatValid*/);

    ckernel::ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_vector_instrn);
    temp.program_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Initialize the packer for a gather/compress: the one-vector pack MOP.
 *
 * @param buf_desc_id: Buffer descriptor of the output, one vector per L1 tile
 * @note Configure the packer formats first. Pair with @ref _llk_unpack_gather_compress_init_ (T0) and
 * @ref _llk_math_gather_compress_init_ (T1).
 */
inline void _llk_pack_gather_compress_init_(const std::uint32_t buf_desc_id)
{
    TTI_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_FACE_ROW_SEL, ckernel::p_pacr::PACK0, 0);
    _llk_pack_gather_compress_mop_config_(buf_desc_id);
}

/**
 * @brief Pack one gather/compress vector from dest tile 0 into L1 tile l1_vector_idx.
 *
 * @param l1_vector_idx: L1 tile index of the vector
 * @note Call @ref _llk_pack_gather_compress_init_ first. Call once per vector, then close the dest section, the same
 * as after any other pack. Pairs with @ref _llk_unpack_gather_compress_ (T0) and @ref _llk_math_gather_compress_ (T1).
 */
inline void _llk_pack_gather_compress_(const std::uint32_t l1_vector_idx)
{
    TTI_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_SEL, ckernel::p_pacr::PACK0, 0);
    TT_SET_DST_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_SEL, ckernel::p_pacr::PACK0, l1_vector_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(ckernel::instrn_buffer);
}
