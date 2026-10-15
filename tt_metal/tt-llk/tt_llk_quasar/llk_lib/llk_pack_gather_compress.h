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

using namespace ckernel;

/**
 * @brief Program a MOP that issues one PACR0_TILE_INC per vector, from dest tile 0 into consecutive L1 tiles,
 * clearing the dest dvalid after each one to release dest back to math.
 *
 * @param buf_desc_id: Buffer descriptor of the output, one vector per L1 tile
 * @param num_vectors: Number of vectors packed per @ref _llk_pack_gather_compress_ call (MOP outer loop count)
 */
inline void _llk_pack_gather_compress_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t num_vectors)
{
    LLK_ASSERT(num_vectors > 0, "num_vectors must be non-zero");

    const std::uint32_t MOP_OUTER_LOOP     = num_vectors;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const std::uint32_t pack_vector_instrn = TT_OP_PACR0_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc*/, buf_desc_id, 1 /*ClrDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, pack_vector_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initialize the packer for a gather/compress: the per-vector pack MOP.
 *
 * Every vector clears the dest dvalid, releasing dest back to math, which pulses it per vector
 * (@ref _llk_math_gather_compress_init_).
 *
 * @param buf_desc_id: Buffer descriptor of the output, one vector per L1 tile
 * @param num_vectors: Number of vectors packed per @ref _llk_pack_gather_compress_ call; must match the unpack and math threads
 * @note Configure the packer formats and the FPU -> PACK dest-dvalid chain first. Pair with
 * @ref _llk_unpack_gather_compress_init_ (T0) and @ref _llk_math_gather_compress_init_ (T1).
 */
inline void _llk_pack_gather_compress_init_(const std::uint32_t buf_desc_id, const std::uint32_t num_vectors = 1)
{
    static_assert(TENSIX_GATHER == 1, "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the quasar_gather variant");
    _llk_pack_gather_compress_mop_config_(buf_desc_id, num_vectors);
}

/**
 * @brief Pack the gather/compress vectors from dest tile 0 to L1, the first into tile start_l1_vector_idx.
 *
 * @param start_l1_vector_idx: L1 tile index of the first vector; vector v goes to start_l1_vector_idx + v
 * @note Call @ref _llk_pack_gather_compress_init_ first.
 */
inline void _llk_pack_gather_compress_(const std::uint32_t start_l1_vector_idx)
{
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, 0);
    TT_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_pacr::PACK0, start_l1_vector_idx);

    // Runs MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
