// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/**
 * @brief Unpack a kt_dim x ct_dim block of in1 tiles into SrcA and the matching in0 tiles into SrcB, preceded by
 *        the mask row when mask_chunk is set.
 *
 * @tparam read_transposed: Walk the in1 tiles column by column (ct_dim tiles with a stride of row_stride, then the
 *                          next tile) instead of row by row, values = <true/false>
 * @param base_address_a: L1 address of in1 (SrcA), in the 16-byte-word encoding of L1_ADDRESS().
 * @param base_address_b: L1 address of in0 (SrcB), in the same encoding.
 * @param base_address_mask: L1 address of the mask row, in the same encoding; read only when mask_chunk is set. The
 *                           read spans ct_dim * 2 SrcB rows of FACE_C_DIM datums, one row per output face.
 * @param tile_index_a: First in1 tile to read.
 * @param tile_index_b: First in0 tile to read.
 * @param tile_size_a: in1 tile size, in 16-byte words.
 * @param tile_size_b: in0 tile size, in 16-byte words.
 * @param kt_dim: Inner dimension in tiles.
 * @param ct_dim: Output width in tiles, 1 to 16; must match the math call.
 * @param mask_chunk: Unpack the mask row into SrcB ahead of the block so the math side can seed DEST from it; must
 *                    match the math call.
 * @param operandB_face_r_dim: in0 face height, 1, 2, 4 or 8. Must equal the unpB_face_r_dim given to
 *                             @ref _llk_unpack_AB_custom_mm_init_: every masked call restores it as unpacker 1's X
 *                             end after the mask read, so a wrong value here leaves the in0 unpacks that follow
 *                             reading the wrong number of rows per face. Required for that reason: no default.
 * @param row_stride: in1 tiles per stored row when read_transposed; zero uses kt_dim.
 * @note Call @ref _llk_unpack_AB_custom_mm_init_ first.
 * @note On the math thread, pair with @ref _llk_math_sdpa_custom_mm_ (same ct_dim and mask_chunk).
 */
template <bool read_transposed = false>
inline void _llk_unpack_AB_sdpa_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_mask,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim,
    const bool mask_chunk,
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t row_stride = 0)
{
    const std::uint32_t in1_row = row_stride ? row_stride : kt_dim;
    if constexpr (read_transposed)
    {
        LLK_ASSERT(in1_row >= kt_dim, "row_stride must contain the contraction tiles");
    }
    volatile std::uint32_t* cfg         = get_cfg_pointer();
    const std::uint32_t block_increment = read_transposed ? in1_row * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * in1_row) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    if (mask_chunk)
    {
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = base_address_mask;
        // The math side broadcasts ct_dim * 2 SrcB rows into DEST, one per output face, so the mask unpack
        // must write that many rows. The init programmed unpacker 1's X end for operandB_face_r_dim rows;
        // left there, every output face >= operandB_face_r_dim would be seeded from stale SrcB rows instead
        // of the mask. Program the extent for the mask, then restore it for the matmul unpacks that follow.
        LLK_ASSERT(ct_dim * 2 <= 64, "mask_chunk requires ct_dim*2 <= 64 SrcB rows");
        TT_SETADCXX(p_setadc::UNP_B, ct_dim * 2 * FACE_C_DIM - 1, 0x0);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcB, 0b00000000, 1, 1);
        TT_SETADCXX(p_setadc::UNP_B, operandB_face_r_dim * FACE_C_DIM - 1, 0x0);
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim);
}
