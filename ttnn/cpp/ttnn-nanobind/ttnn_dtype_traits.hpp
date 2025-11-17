// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <ttnn/tensor/types.hpp>

template <nanobind::dlpack::dtype nb_dtype, typename = int>
struct py_to_ttnn_dtype {
    constexpr static auto ttnn_dtype = tt::tt_metal::DataType::INVALID;
};

//
// type maps between python libs and ttnn
//
// torch ingress -> ttnn label inferred
// float64       -> N/A
// float32       -> float32
// float16       -> bfloat16
// bfloat16      -> bfloat16
// uint64        -> N/A
// int64         -> uint32
// uint32        -> N/A
// int32         -> int32
// uint16        -> N/A
// int16         -> uint16
// uint8         -> uint8
// int8          -> N/A

// ttnn label    -> torch convert ingress
// uint8         -> uint8
// uint16        -> int16
// int32         -> int32
// uint32        -> int32
// bfloat4_b     -> float32
// bfloat8_b     -> float32
// bfloat16      -> bfloat16

// numpy ingress -> ttnn inferred
// float64       -> N/A
// float32       -> float32
// float16       -> N/A
// uint64        -> N/A
// int64         -> uint32
// uint32        -> N/A
// int32         -> int32
// uint16        -> N/A
// int16         -> uint16
// uint8         -> ubyte
// int8          -> N/A

// ttnn label    -> numpy convert ingress
// uint8         -> ubyte
// uint16        -> int16
// int32         -> int32
// uint32        -> int32
// bfloat4_b     -> float32
// bfloat8_b     -> float32
// float32       -> float32

template <tt::tt_metal::DataType dt, typename = int /*SFINAE*/>
struct ttnn_dtype_traits {
    using underlying_type = void;
    static constexpr nanobind::dlpack::dtype value{.code = 0, .bits = 0, .lanes = 0};
    static constexpr auto name = nanobind::detail::descr<0>();
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::INVALID> {
    using underlying_type = void;
    static constexpr nanobind::dlpack::dtype value{.code = 0, .bits = 0, .lanes = 0};
    static constexpr auto name = nanobind::detail::const_name("INVALID");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::BFLOAT16> {
    using underlying_type = ::bfloat16;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat), .bits = 16, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("BFLOAT16");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::FLOAT32> {
    using underlying_type = float;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Float), .bits = 32, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("FLOAT32");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::UINT32> {
    using underlying_type = std::uint32_t;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::UInt), .bits = 32, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("UINT32");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::BFLOAT8_B> {
    using underlying_type = float;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat), .bits = 8, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("BFLOAT8_B");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::BFLOAT4_B> {
    using underlying_type = float;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Bfloat), .bits = 4, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("BFLOAT4_B");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::UINT8> {
    using underlying_type = std::uint8_t;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::UInt), .bits = 8, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("UINT8");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::UINT16> {
    using underlying_type = std::uint16_t;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::UInt), .bits = 16, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("UINT16");
};

template <>
struct ttnn_dtype_traits<tt::tt_metal::DataType::INT32> {
    using underlying_type = std::int32_t;
    static constexpr nanobind::dlpack::dtype value{
        .code = static_cast<std::uint8_t>(nanobind::dlpack::dtype_code::Int), .bits = 32, .lanes = 1};
    static constexpr auto name = nanobind::detail::const_name("INT32");
};
