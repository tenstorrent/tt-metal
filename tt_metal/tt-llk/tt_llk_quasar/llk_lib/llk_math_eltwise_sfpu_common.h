// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_sfpu.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::math;

/** @brief Configure the SFPU address modes for elementwise ops. */
inline void _eltwise_sfpu_configure_addrmod_()
{
    _sfpu_configure_addrmod_();
}

/**
 * @brief Begin an SFPU elementwise op on the math thread for the given dest tile.
 *
 * @tparam TILE_SHAPE: Destination tile shape used to calculate the write address.
 * @param tile_index: Tile index into the destination register to operate on.
 * @note Pair with @ref _llk_math_eltwise_sfpu_done_ once the op has run.
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE = ckernel::trisc::DstTileShape::Tile32x32>
inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t tile_index)
{
    _llk_math_sfpu_start_<TILE_SHAPE>(tile_index);
}

/**
 * @brief Finish the current SFPU elementwise op on the math thread.
 *
 * @note Call after the @ref _llk_math_eltwise_sfpu_start_ that opened the op.
 */
inline void _llk_math_eltwise_sfpu_done_()
{
    _llk_math_sfpu_done_();
}

/**
 * @brief Clear the SrcA/SrcB valid flags after the SFPU has consumed them.
 *
 * @tparam SRCS_RD_DONE: Clear the read-valid flags
 * @tparam SRCS_WR_DONE: Clear the write-valid flags
 */
template <bool SRCS_RD_DONE, bool SRCS_WR_DONE>
inline void _llk_math_eltwise_sfpu_srcs_clear_vlds_()
{
    _llk_math_sfpu_srcs_clear_vlds_<SRCS_RD_DONE, SRCS_WR_DONE>();
}

/** @brief Advance the SFPU destination address by one face. */
inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    _llk_math_sfpu_inc_dst_face_addr_();
}

/** @brief Initialize the math thread for SFPU elementwise operations. */
inline void _llk_math_eltwise_sfpu_init_()
{
    _llk_math_sfpu_init_();
}

/**
 * @brief Number of 16-column faces across one row of a Dest slot.
 *
 * A 32x32 slot holds 2x2 faces and a 32x16 slot holds 2x1 faces. Narrower slots hold less than a
 * face per row and return 0, which the SFPU face walk rejects.
 *
 * @tparam SLOT: Destination tile shape
 */
template <ckernel::trisc::DstTileShape SLOT>
constexpr int _llk_math_eltwise_sfpu_slot_faces_c_()
{
    return SLOT == ckernel::trisc::DstTileShape::Tile32x32 ? MAX_NUM_FACES_C_DIM : SLOT == ckernel::trisc::DstTileShape::Tile32x16 ? 1 : 0;
}

/**
 * @brief Advance the SFPU Dest address past NUM_FACES faces without processing them.
 *
 * @tparam NUM_FACES: Number of faces to skip, expanded at compile time into straight-line increments
 */
template <int NUM_FACES>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_skip_faces_()
{
    if constexpr (NUM_FACES > 0)
    {
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_skip_faces_<NUM_FACES - 1>();
    }
}

/**
 * @brief Call an SFPU functor once per face of the tile described by TENSOR_SHAPE.
 *
 * The tile starts at face 0 of the current Dest slot and is walked with the slot's face stride.
 * Faces are visited in row-major order and faces outside the shape are skipped. In a 32x32 slot a
 * 16x32 tile visits faces 0 and 1 and a 32x16 tile visits faces 0 and 2; in a 32x16 slot the two
 * faces of a 32x16 tile are contiguous. The walk always ends at the end of the slot.
 *
 * The static_asserts limit TENSOR_SHAPE to full-face tiles of at most 2x2 faces, all of which are in the
 * math TensorShape coverage table, so no runtime LLK_VALIDATE_TENSOR_SHAPE_MATH check is needed.
 *
 * @tparam TENSOR_SHAPE: Tile to process; must be made of full 16x16 faces and fit in SLOT
 * @tparam SLOT: Destination tile shape the tile lives in
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor to run on each face; processes one face per call
 * @param args: Arguments passed to every sfpu_func call
 */
template <TensorShape TENSOR_SHAPE, ckernel::trisc::DstTileShape SLOT = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_for_each_face_(Callable&& sfpu_func, Args&&... args)
{
    constexpr int SLOT_FACES_C = _llk_math_eltwise_sfpu_slot_faces_c_<SLOT>();
    static_assert(SLOT_FACES_C > 0, "SFPU face walk requires a Dest slot at least 16 columns wide");
    static_assert(TENSOR_SHAPE.face_r_dim == MAX_FACE_R_DIM && TENSOR_SHAPE.face_c_dim == MAX_FACE_C_DIM, "SFPU face walk requires full 16x16 faces");
    static_assert(TENSOR_SHAPE.num_faces_r_dim >= 1 && TENSOR_SHAPE.num_faces_r_dim <= MAX_NUM_FACES_R_DIM, "SFPU face walk supports 1 or 2 rows of faces");
    static_assert(TENSOR_SHAPE.num_faces_c_dim >= 1 && TENSOR_SHAPE.num_faces_c_dim <= SLOT_FACES_C, "TensorShape does not fit in the Dest slot");

    constexpr int FACES_R = TENSOR_SHAPE.num_faces_r_dim;
    constexpr int FACES_C = TENSOR_SHAPE.num_faces_c_dim;

    if constexpr (FACES_C == SLOT_FACES_C)
    {
        // Whole rows of faces are contiguous in Dest.
#pragma GCC unroll 0
        for (int face = 0; face < FACES_R * FACES_C; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        // One face per row of faces; skip the rest of the row.
#pragma GCC unroll 0
        for (int face_r = 0; face_r < FACES_R; face_r++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_skip_faces_<SLOT_FACES_C - FACES_C>();
        }
    }

    // Skip the rows of faces outside the shape.
    _llk_math_eltwise_sfpu_skip_faces_<(MAX_NUM_FACES_R_DIM - FACES_R) * SLOT_FACES_C>();
}

/**
 * @brief Legacy VectorMode dispatch, kept as a shim over @ref _llk_math_eltwise_sfpu_for_each_face_.
 *
 * RC walks the whole slot, R its top row of faces and C its left column of faces; in a 32x32 slot
 * that is a 32x32, 16x32 and 32x16 tile. Any other mode (None, RC_custom) calls sfpu_func once, for
 * functors that walk Dest themselves.
 *
 * A slot narrower than 32x16 is a single 16-row block, so there is nothing to walk: RC, C, None and
 * RC_custom call sfpu_func once, over the whole tile. R would need the top 16 rows of the tile
 * alone, which that block does not separate, so it is unsupported there and runs nothing.
 *
 * @tparam SLOT: Destination tile shape the tile lives in
 * @tparam Callable: Type of the SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU functor to run
 * @param vector_mode: Legacy face selection, values = <RC/R/C/None/RC_custom>
 * @param args: Arguments passed to sfpu_func
 * @note Remove together with VectorMode; tracked in tt-metal issues #36281 and #57603.
 */
template <ckernel::trisc::DstTileShape SLOT = ckernel::trisc::DstTileShape::Tile32x32, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    constexpr std::uint8_t SLOT_FACES_C = _llk_math_eltwise_sfpu_slot_faces_c_<SLOT>();

    if constexpr (SLOT_FACES_C == 0)
    {
        LLK_ASSERT(vector_mode != VectorMode::R, "VectorMode::R is not supported for Dest slots narrower than 32x16");
        if (vector_mode != VectorMode::R)
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
        }
    }
    else
    {
        constexpr TensorShape SHAPE_RC = make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, SLOT_FACES_C);
        constexpr TensorShape SHAPE_R  = make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, 1, SLOT_FACES_C);
        constexpr TensorShape SHAPE_C  = make_tensor_shape(MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, 1);

        if (vector_mode == VectorMode::RC)
        {
            _llk_math_eltwise_sfpu_for_each_face_<SHAPE_RC, SLOT>(sfpu_func, args...);
        }
        else if (vector_mode == VectorMode::R)
        {
            _llk_math_eltwise_sfpu_for_each_face_<SHAPE_R, SLOT>(sfpu_func, args...);
        }
        else if (vector_mode == VectorMode::C)
        {
            _llk_math_eltwise_sfpu_for_each_face_<SHAPE_C, SLOT>(sfpu_func, args...);
        }
        else
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
        }
    }
}

/**
 * @brief Determine the stochastic-rounding conversion mode for a source -> cast format pair.
 *
 * @tparam SRC_FMT: Source data format
 * @tparam CAST_FMT: Target (cast) data format
 * @return sfp_stochrnd_mod selector for the conversion.
 */
template <DataFormat SRC_FMT, DataFormat CAST_FMT>
inline constexpr std::uint32_t _sfpu_stochround_conversion_()
{
    if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Float16)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16A;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Float16_b)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_FP16B;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::UInt8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_UINT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Int8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_INT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Int32 && CAST_FMT == DataFormat::UInt8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::INT32_TO_UINT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Int32 && CAST_FMT == DataFormat::Int8)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::INT32_TO_INT8;
    }
    else if constexpr (SRC_FMT == DataFormat::Float32 && CAST_FMT == DataFormat::Int16)
    {
        return ckernel::p_sfpu::sfp_stochrnd_mod::FP32_TO_INT16;
    }
    else
    {
        static_assert(
            !std::is_same_v<decltype(SRC_FMT), DataFormat>,
            "Unsupported DataFormats for stochround conversion"); // need the condition to depend on the template parameter... compiler things
    }
}

/**
 * @brief Compile-time sfpmem type selector from a DataFormat.
 *
 * Formats with no dedicated SFPU load/store mode (fp8, the MX block formats, 4-bit ints) map to
 * sfpmem::DEFAULT — the implied/default format the producing engine left in the register-file
 * format config (ISA: HW derives it from ALU_FORMAT_SPEC_REG / ACC_CTRL).
 *
 * @tparam FMT: Data format to map
 * @return sfpmem type parameter for FMT, or sfpmem::DEFAULT for formats with no dedicated mode.
 * @note Runtime equivalent: @ref _sfpu_sfpmem_type_. Keep both in sync when adding a format.
 */
template <DataFormat FMT>
inline constexpr std::uint32_t _sfpu_sfpmem_type_()
{
    if constexpr (FMT == DataFormat::Float16)
    {
        return ckernel::p_sfpu::sfpmem::FP16A;
    }
    else if constexpr (FMT == DataFormat::Float16_b)
    {
        return ckernel::p_sfpu::sfpmem::FP16B;
    }
    else if constexpr (FMT == DataFormat::Float32 || FMT == DataFormat::Tf32)
    {
        return ckernel::p_sfpu::sfpmem::FP32;
    }
    else if constexpr (FMT == DataFormat::Int32)
    {
        return ckernel::p_sfpu::sfpmem::INT32;
    }
    else if constexpr (FMT == DataFormat::Int16)
    {
        return ckernel::p_sfpu::sfpmem::INT16;
    }
    else if constexpr (FMT == DataFormat::Int8)
    {
        return ckernel::p_sfpu::sfpmem::INT8;
    }
    else if constexpr (FMT == DataFormat::UInt8)
    {
        return ckernel::p_sfpu::sfpmem::UINT8;
    }
    else if constexpr (FMT == DataFormat::UInt16)
    {
        return ckernel::p_sfpu::sfpmem::UINT16;
    }
    else
    {
        // No dedicated SFPU mode (fp8, MX block formats, 4-bit ints): fall back to the implied/
        // default register-file format. Matches the runtime overload's default case.
        return ckernel::p_sfpu::sfpmem::DEFAULT;
    }
}

/**
 * @brief Runtime counterpart to _sfpu_sfpmem_type_<FMT>() — runtime sfpmem type parameter from DataFormat.
 *
 * Use for any DataFormat-driven path (unpack dst, pack src, reg_data_format, etc.). Unknown
 * values return sfpmem::DEFAULT (ISA: HW may derive format from ALU_FORMAT_SPEC_REG / ACC_CTRL).
 * When adding a format, update this switch and the template above together.
 *
 * @param fmt: Data format to map
 * @return sfpmem type parameter, or sfpmem::DEFAULT for unknown formats.
 */
inline std::uint32_t _sfpu_sfpmem_type_(DataFormat fmt)
{
    switch (fmt)
    {
        case DataFormat::Float16:
            return ckernel::p_sfpu::sfpmem::FP16A;
        case DataFormat::Float16_b:
            return ckernel::p_sfpu::sfpmem::FP16B;
        case DataFormat::Float32:
        case DataFormat::Tf32:
            return ckernel::p_sfpu::sfpmem::FP32;
        case DataFormat::Int32:
            return ckernel::p_sfpu::sfpmem::INT32;
        case DataFormat::Int16:
            return ckernel::p_sfpu::sfpmem::INT16;
        case DataFormat::Int8:
            return ckernel::p_sfpu::sfpmem::INT8;
        case DataFormat::UInt8:
            return ckernel::p_sfpu::sfpmem::UINT8;
        case DataFormat::UInt16:
            return ckernel::p_sfpu::sfpmem::UINT16;
        default:
            return ckernel::p_sfpu::sfpmem::DEFAULT;
    }
}

/**
 * @brief Same as _sfpu_sfpmem_type_(DataFormat) for raw enum underlying values (e.g. UInt16 = 130).
 *
 * @param data_format_raw: Underlying integer value of a DataFormat enumerator
 * @return sfpmem type parameter, or sfpmem::DEFAULT for unknown formats.
 */
inline std::uint32_t _sfpu_sfpmem_type_(std::uint32_t data_format_raw)
{
    return _sfpu_sfpmem_type_(static_cast<DataFormat>(data_format_raw));
}
