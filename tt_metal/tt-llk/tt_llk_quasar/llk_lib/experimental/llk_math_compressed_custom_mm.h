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
// next output tile. Zero weight tiles need no special case here: the unpack thread hands them over as zeroed SrcA
// banks. Multiplying by zero costs one MOP of FPU time, but it keeps the metadata off this thread, whose RISC-V
// loop (an L1 load and a branch per tile) was slower than the FPU work it skipped.

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
 * Accumulates the ct_dim output tiles in Dest, starting at dst_index, over all kt_dim. The unpack thread turns
 * every zero weight tile into a zeroed SrcA bank, so this multiplies every tile and does not read the metadata.
 *
 * @param dst_index: Index of the first output tile in Dest, in units of output tiles.
 * @param kt_dim: Number of tiles in the common dimension, values >= 1.
 * @param ct_dim: Number of tiles in the column dimension of Input 1, values = 1 - 16.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ before this function.
 * @note On the unpack thread, pair with @ref _llk_unpack_compressed_custom_mm_ with the same kt_dim and ct_dim.
 */
inline void _llk_math_compressed_custom_mm_(const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim)
{
    _set_dst_write_addr_by_rows_(dst_index);

    for (std::uint32_t k = 0; k < kt_dim; k++)
    {
        for (std::uint32_t c = 0; c < ct_dim; c++)
        {
            ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        }

        // Release this row's SrcB and rewind Dest to the first output tile for the next k.
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
    }
}
