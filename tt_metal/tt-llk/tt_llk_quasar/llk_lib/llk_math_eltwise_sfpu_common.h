// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_sfpu.h"
#include "llk_defs.h"

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
 * @param tile_index: Tile index into the destination register to operate on.
 * @note Pair with @ref _llk_math_eltwise_sfpu_done_ once the op has run.
 */
inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t tile_index)
{
    _llk_math_sfpu_start_(tile_index);
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
 * @brief Apply an SFPU op across the dest faces selected by the vector mode.
 *
 * Invokes sfpu_func once per active face and advances the dest face address, walking only the
 * faces the mode covers: RC = all 4 faces, R = faces 0/1 (row vector), C = faces 0/2 (column
 * vector), None = a single call on the current face.
 *
 * @tparam Callable: Type of the per-face SFPU functor
 * @tparam Args: Argument types forwarded to the functor
 * @param sfpu_func: SFPU op to run on each selected face
 * @param vector_mode: Faces to cover, values = <RC/R/C/None>
 * @param args: Arguments forwarded to sfpu_func
 * @todo Revisit vector mode handling — tracked in tt-metal issue #36281.
 */
template <typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    if (vector_mode == VectorMode::RC)
    {
        // All 4 faces
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else if (vector_mode == VectorMode::R)
    {
        // Face0 + Face1 (row vector)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        // Skip Face2 + Face3
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    else if (vector_mode == VectorMode::C)
    {
        // Face0 + Face2 (column vector)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
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
