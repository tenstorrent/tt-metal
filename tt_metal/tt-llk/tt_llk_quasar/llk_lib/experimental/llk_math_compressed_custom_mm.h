// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_compressed_custom_mm_common.h"
#include "llk_math_common.h"
#include "llk_math_matmul.h"

using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/*************************************************************************
 * LLK MATH COMPRESSED CUSTOM MM
 *************************************************************************/

// Same MVMUL stream as the standard matmul with rt_dim = 1: the MOP programmed by _llk_math_matmul_init_ does one
// Tile x Tile multiply (4 MVMULs per fidelity phase for an in0 of up to 8 rows), releases SrcA and moves Dest to the
// next output tile. The only difference is that a zero weight tile has no SrcA bank to consume, so this thread
// skips its MOP and moves Dest past the tile instead.

/**
 * @brief Initializes addrmods and the MOP for compressed matrix multiply: Input0 (SrcB) x compressed Input1 (SrcA).
 *
 * With rt_dim = 1 the matmul reuses SrcB across the row and steps Dest by one tile, whatever ct_dim is, so the
 * configuration does not depend on ct_dim.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param src_a_format: SrcA register format, the format every weight tile unpacks to.
 * @param src_b_format: SrcB register format.
 * @param src_b_shape: Input 0/SrcB tile shape, [{1, 2, 4, 8}, 32].
 * @note On the unpack thread, pair with @ref _llk_unpack_compressed_custom_mm_init_ (T0); on the pack thread, pair
 *       with @ref _llk_pack_matmul_init_ (T2).
 * @note @ref _llk_math_compressed_custom_mm_ runs the configured matmul.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE = ckernel::MathFidelity::LoFi>
inline void _llk_math_compressed_custom_mm_init_(const DataFormat src_a_format, const DataFormat src_b_format, const TensorShape src_b_shape)
{
    LLK_ASSERT(_llk_compressed_custom_mm_is_valid_src_b_shape_(src_b_shape), "compressed_custom_mm supports in0 tile shapes [{1, 2, 4, 8}, 32] only");

    _llk_math_matmul_init_<MATH_FIDELITY_TYPE>(src_a_format, src_b_format, 1 /*ct_dim*/, 1 /*rt_dim*/, src_b_shape, DEFAULT_TENSOR_SHAPE);
}

/**
 * @brief Does compressed matrix multiply: Output [1, ct_dim] += Input0 [1, kt_dim] x Input1 [kt_dim, ct_dim], in tiles.
 *
 * Accumulates the ct_dim output tiles in Dest, starting at dst_index, over all kt_dim. Zero weight tiles, as
 * marked in the metadata, contribute nothing and are skipped.
 *
 * @param base_address_meta: Byte address of the kt_dim * ct_dim metadata words, the same words the unpack thread
 *        reads. Not L1_ADDRESS(): the RISC-V dereferences it.
 * @param dst_index: Index of the first output tile in Dest, in units of output tiles.
 * @param kt_dim: Number of tiles in the common dimension, values >= 1.
 * @param ct_dim: Number of tiles in the column dimension of Input 1, values = 1 - 16.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ before this function.
 * @note On the unpack thread, pair with @ref _llk_unpack_compressed_custom_mm_ with the same metadata.
 */
inline void _llk_math_compressed_custom_mm_(
    const std::uint32_t base_address_meta, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim)
{
    const std::uint32_t* meta             = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    const std::uint32_t dst_rows_per_tile = 1U << ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX];

    _set_dst_write_addr_by_rows_(dst_index);

    for (std::uint32_t k = 0; k < kt_dim; k++)
    {
        bool src_b_consumed = false;

        for (std::uint32_t c = 0; c < ct_dim; c++)
        {
            if (_llk_compressed_custom_mm_is_zero_tile_(*meta++))
            {
                // No SrcA bank to multiply: move Dest (and its carriage return) to the next output tile, as the
                // tile-done MVMUL would have.
                TT_SETRWC(p_setrwc::CLR_NONE, 0, (c + 1) * dst_rows_per_tile, p_setrwc::SET_D);
            }
            else
            {
                ckernel_template::run_bank0_sw_cntl(instrn_buffer);
                src_b_consumed = true;
            }
        }

        if (!src_b_consumed)
        {
            // Every weight tile of the row was zero, so no MVMUL waited for this row's SrcB. Wait for it here,
            // so the release below clears the bank the unpacker filled and not the one it is filling next.
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::MATH, p_stall::SRCB_VLD);
        }

        // Release this row's SrcB and rewind Dest to the first output tile for the next k.
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
    }
}
