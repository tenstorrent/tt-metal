// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

#if TENSIX_GATHER != 1
#error "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the saturn variant"
#endif

/**
 * @brief Program a MOP that issues one UNPACR1_STRIDE into SrcB, unpacking one vector.
 *
 * The UNPACR1_STRIDE sets the SrcB dvalid and spans one vector: 4 16-datum L1 rows for an Int8/UInt8 (64-element)
 * vector, 2 rows otherwise.
 *
 * @param buf_desc_id: Buffer descriptor of the operand, with the Strided (16x1x1) geometry
 * @param unpack_dst_format: SrcB data format
 */
inline void _llk_unpack_gather_compress_mop_config_(const std::uint32_t buf_desc_id, const DataFormat unpack_dst_format)
{
    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const bool is_8bit_format = (unpack_dst_format == DataFormat::Int8) || (unpack_dst_format == DataFormat::UInt8);

    const std::uint32_t row_16_inc_log2 = is_8bit_format ? 2 : 1;

    const std::uint32_t unpack_vector_instrn = TT_OP_UNPACR1_STRIDE(
        0 /*Src_Reg_Y_Cntr_Incr*/,
        row_16_inc_log2 /*L1_Tile_Idx_or_Tile_Idx_Inc*/,
        1 /*Tile_Idx_Inc*/,
        0 /*Row_Mask_Reg_Sel*/,
        0 /*L1_16datums_Row_Index*/,
        buf_desc_id,
        1 /*SetDatValid*/);

    ckernel::ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_vector_instrn);
    temp.program_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Program the UNPACKER1 ARG-FIFO/stride configuration that UNPACR1_STRIDE gather/compress reads.
 *
 * @tparam MODE: GATHER (index per output element) or COMPRESS (mask per input element)
 */
template <ckernel::GatherCompressModeSelect MODE>
inline void _llk_unpack_gather_compress_hw_configure_()
{
    ckernel::cfg_rmw(THCON_UNPACKER1_REG0_ENABLE_ARG_FIFO_RMW, 1);                                           // enable index FIFO
    ckernel::cfg_rmw(THCON_UNPACKER1_REG0_ARG_FIFO_UNPACR_STRIDE_MODE_RMW, static_cast<std::uint8_t>(MODE)); // program the register to describe mode
    ckernel::cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);                                  // Use a fixed stride in L1
    ckernel::cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, 1);                                    // Step through contiguous rows
    ckernel::cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);                                                 // A prior unary transpose leaves it set
}

/**
 * @brief Initialize the unpacker for a gather/compress: ARG-FIFO/stride configuration and MOP.
 *
 * @tparam MODE: Operation, values = <GATHER/COMPRESS>
 * @param buf_desc_id: Buffer descriptor of the operand, with the Strided (16x1x1) geometry
 * @param unpack_dst_format: SrcB data format; Int8/UInt8 select 64-element vectors, others 32
 * @note Pair with @ref _llk_math_gather_compress_init_ (T1) and @ref _llk_pack_gather_compress_init_ (T2). Call
 * @ref _llk_unpack_gather_compress_uninit_ before the next unpack op's init.
 */
template <ckernel::GatherCompressModeSelect MODE>
inline void _llk_unpack_gather_compress_init_(const std::uint32_t buf_desc_id, const DataFormat unpack_dst_format)
{
    LLK_ASSERT(
        unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8 || unpack_dst_format == DataFormat::Float16_b,
        "Gather/compress supports Int8, UInt8, and Float16_b formats");

    _llk_unpack_gather_compress_hw_configure_<MODE>();
    _llk_unpack_gather_compress_mop_config_(buf_desc_id, unpack_dst_format);
}

/**
 * @brief Unpack one gather/compress vector into SrcB, from 16-datum row l1_row_16_idx of the operand.
 *
 * @param l1_row_16_idx: 16-datum L1 row of the operand where the vector starts (16B for Int8/UInt8, 32B for
 * Float16_b)
 * @note Call @ref _llk_unpack_gather_compress_init_ first. Call this to queue the MOP, then push the vector's
 * index/mask entry into the UNPACKER1 ARG FIFO: the UNPACR1_STRIDE stalls until the entry arrives. Call once per
 * vector, in step with @ref _llk_math_gather_compress_ (T1) and @ref _llk_pack_gather_compress_ (T2).
 */
inline void _llk_unpack_gather_compress_(const std::uint32_t l1_row_16_idx = 0)
{
    TT_SET_SRC_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_SEL, ckernel::p_unpacr::UNP_B, l1_row_16_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(ckernel::p_set_inc_sel::TILE_SEL, ckernel::p_unpacr::UNP_B, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Undo @ref _llk_unpack_gather_compress_hw_configure_: disable the UNPACKER1 ARG FIFO.
 *
 * Without this, any later UNPACR1_STRIDE (e.g. a strided tilize into SrcB) would block on the ARG FIFO. The stride
 * registers are left as-is, since every strided init reprograms them.
 *
 * Waits for UNPACKER1 to finish first, so the last UNPACR1_STRIDEs still run with the ARG FIFO enabled.
 *
 * @note Call after the last @ref _llk_unpack_gather_compress_, then call the next unpack op's init.
 */
inline void _llk_unpack_gather_compress_uninit_()
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, 0, 0, ckernel::p_stall::UNPACK1);
    ckernel::cfg_rmw(THCON_UNPACKER1_REG0_ENABLE_ARG_FIFO_RMW, 0);
}
