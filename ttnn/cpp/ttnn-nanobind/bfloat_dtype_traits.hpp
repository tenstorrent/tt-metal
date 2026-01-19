// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/tensor_impl.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <>
struct dtype_traits<::bfloat16> {
    static constexpr dlpack::dtype value{
        static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat),  // type code
        16,                                                               // size in bits
        1                                                                 // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat16");
};

template <>
struct dtype_traits<tt::tt_metal::tensor_impl::bfloat8_b> {
    static constexpr dlpack::dtype value{
        static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat),  // type code
        8,                                                                // size in bits
        1                                                                 // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat8_b");
};

template <>
struct dtype_traits<tt::tt_metal::tensor_impl::bfloat4_b> {
    static constexpr dlpack::dtype value{
        static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat),  // type code
        4,                                                                // size in bits
        1                                                                 // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat4_b");
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
