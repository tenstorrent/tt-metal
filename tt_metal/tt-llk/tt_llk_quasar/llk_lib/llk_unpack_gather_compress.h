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

using namespace ckernel;

/**
 * @brief Program a MOP that issues one UNPACR1_STRIDE into SrcB per vector.
 *
 * Each UNPACR1_STRIDE sets the SrcB dvalid and advances the L1 tile index by one vector: 4 16-byte rows for an
 * Int8/UInt8 (64-element) vector, 2 rows otherwise.
 *
 * @param buf_desc_id: Buffer descriptor of the operand, with the Strided (16x1x1) geometry
 * @param unpack_dst_format: SrcB data format, as a DataFormat value
 * @param num_vectors: Number of vectors unpacked per @ref _llk_unpack_gather_compress_ call (MOP outer loop count)
 */
inline void _llk_unpack_gather_compress_mop_config_(const std::uint32_t buf_desc_id, const std::uint32_t unpack_dst_format, const std::uint32_t num_vectors)
{
    LLK_ASSERT(num_vectors > 0, "num_vectors must be non-zero");

    const std::uint32_t MOP_OUTER_LOOP     = num_vectors;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    const bool is_8bit_format =
        (unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Int8)) || (unpack_dst_format == static_cast<std::uint32_t>(DataFormat::UInt8));

    const std::uint32_t row_16_inc_log2 = is_8bit_format ? 2 : 1;

    const std::uint32_t unpack_vector_instrn = TT_OP_UNPACR1_STRIDE(
        0 /*Src_Reg_Y_Cntr_Incr*/,
        row_16_inc_log2 /*L1_Tile_Idx_or_Tile_Idx_Inc*/,
        1 /*Tile_Idx_Inc*/,
        0 /*Row_Mask_Reg_Sel*/,
        0 /*L1_16datums_Row_Index*/,
        buf_desc_id,
        1 /*SetDatValid*/);

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, unpack_vector_instrn);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Program the UNPACKER1 ARG-FIFO/stride configuration that UNPACR1_STRIDE gather/compress reads.
 *
 * @tparam MODE: GATHER (index per output element) or COMPRESS (mask per input element)
 * @param unpack_dst_format: SrcB data format
 * @note Also sets ALU_ACC_CTRL_Zero_Flag_disabled_src, so a zero index in the exponent slot is not read as a zero flag.
 */
template <GatherCompressModeSelect MODE>
inline void _llk_unpack_gather_compress_hw_configure_(const std::uint32_t unpack_dst_format)
{
    cfg_rmw(THCON_UNPACKER1_REG0_ENABLE_ARG_FIFO_RMW, 1);                                           // enable index FIFO
    cfg_rmw(THCON_UNPACKER1_REG0_ARG_FIFO_UNPACR_STRIDE_MODE_RMW, static_cast<std::uint8_t>(MODE)); // program the register to describe mode
    cfg_rmw(THCON_UNPACKER1_REG1_UNPACK_STRIDE_VAL_SOURCE_RMW, 0);                                  // Use a fixed stride in L1
    cfg_rmw(THCON_UNPACKER1_REG2_UNPACK_STRIDE_OFFSET_0_RMW, 1);                                    // Step through contiguous rows
    cfg_rmw(THCON_UNPACKER1_REG0_OUT_DATA_FORMAT_RMW, static_cast<std::uint8_t>(unpack_dst_format));
    cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 1); // Keeps a zero index in the exponent slot from being reinterpreted as a zero-flag.
}

/**
 * @brief Initialize the unpacker for a gather/compress: ARG-FIFO/stride configuration and MOP.
 *
 * @tparam MODE: Operation, values = <GATHER/COMPRESS>
 * @param buf_desc_id: Buffer descriptor of the operand, with the Strided (16x1x1) geometry
 * @param unpack_dst_format: SrcB data format, as a DataFormat value; Int8/UInt8 select 64-element vectors, others 32
 * @param num_vectors: Number of vectors unpacked per @ref _llk_unpack_gather_compress_ call; must match the math and pack threads
 * @note Pair with @ref _llk_math_gather_compress_init_ (T1) and @ref _llk_pack_gather_compress_init_ (T2).
 */
template <GatherCompressModeSelect MODE>
inline void _llk_unpack_gather_compress_init_(const std::uint32_t buf_desc_id, const std::uint32_t unpack_dst_format, const std::uint32_t num_vectors = 1)
{
    static_assert(TENSIX_GATHER == 1, "Gather/compress needs a Tensix built with the gather unit (TENSIX_GATHER); build for the quasar_gather variant");
    LLK_ASSERT(
        unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Int8) || unpack_dst_format == static_cast<std::uint32_t>(DataFormat::UInt8) ||
            unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Float16_b),
        "Gather/compress supports Int8, UInt8, and Float16_b formats");

    _llk_unpack_gather_compress_hw_configure_<MODE>(unpack_dst_format);
    _llk_unpack_gather_compress_mop_config_(buf_desc_id, unpack_dst_format, num_vectors);
}

/**
 * @brief Unpack the gather/compress vectors into SrcB, starting at 16-byte row l1_row_16_idx of the operand.
 *
 * @param l1_row_16_idx: 16-byte L1 row of the operand where the first vector starts
 * @note Call @ref _llk_unpack_gather_compress_init_ first, and push one index/mask vector per UNPACR into the
 * UNPACKER1 ARG FIFO before every call: each UNPACR1_STRIDE consumes its FIFO entry.
 */
inline void _llk_unpack_gather_compress_(const std::uint32_t l1_row_16_idx = 0)
{
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, l1_row_16_idx);
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Undo @ref _llk_unpack_gather_compress_hw_configure_: disable the UNPACKER1 ARG FIFO and re-enable the
 * SrcB zero flag.
 *
 * Without this, any later UNPACR1_STRIDE (e.g. a strided tilize into SrcB) would block on the ARG FIFO. The stride
 * registers are left as-is, since every strided init reprograms them, and OUT_DATA_FORMAT is left at the gather
 * format, so reconfigure SrcB if the next op's format differs.
 *
 * @note Call after the last @ref _llk_unpack_gather_compress_ once its UNPACRs have drained.
 */
inline void _llk_unpack_gather_compress_uninit_()
{
    cfg_rmw(THCON_UNPACKER1_REG0_ENABLE_ARG_FIFO_RMW, 0);
    cfg_rmw(ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW, 0);
}
