// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"

template <class F, class... ARGS>
inline void _llk_math_eltwise_unary_sfpu_params_(F&& sfpu_func, std::uint32_t dst_tile_index, ARGS&&... args)
{
    _llk_math_eltwise_sfpu_params_(std::forward<F>(sfpu_func), dst_tile_index, std::forward<ARGS>(args)...);
}

/**
 * @brief Determines the stochround conversion type based on source and cast data formats
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
 * @brief Compile-time sfpmem type parameter from DataFormat.
 * @see _sfpu_sfpmem_type_(DataFormat) for the runtime equivalent.
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
        static_assert(
            !std::is_same_v<decltype(FMT), DataFormat>,
            "Unsupported DataFormat for sfpmem type determination"); // need the condition to depend on the template parameter... compiler things
    }
}

/**
 * @brief Runtime counterpart to _sfpu_sfpmem_type_<FMT>() — runtime sfpmem type parameter from DataFormat.
 *
 * Use for any DataFormat-driven path (unpack dst, pack src, reg_data_format, etc.). Unknown
 * values return sfpmem::DEFAULT (ISA: HW may derive format from ALU_FORMAT_SPEC_REG / ACC_CTRL).
 * When adding a format, update this switch and the template above together.
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
        case DataFormat::UInt8:
            return ckernel::p_sfpu::sfpmem::UINT8;
        case DataFormat::UInt16:
            return ckernel::p_sfpu::sfpmem::UINT16;
        default:
            return ckernel::p_sfpu::sfpmem::DEFAULT;
    }
}

/** @brief Same as _sfpu_sfpmem_type_(DataFormat) for raw enum underlying values (e.g. UInt16 = 130). */
inline std::uint32_t _sfpu_sfpmem_type_(std::uint32_t data_format_raw)
{
    return _sfpu_sfpmem_type_(static_cast<DataFormat>(data_format_raw));
}
