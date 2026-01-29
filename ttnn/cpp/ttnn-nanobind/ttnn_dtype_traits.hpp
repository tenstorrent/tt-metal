// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <bit>  // bit_cast

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <ttnn/tensor/types.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/unreachable.hpp>

namespace ttnn {
namespace ttnn_dtype_traits::detail {
//
// type maps between python libs and ttnn
//
// Pytorch 2.3 introduced unsigned int types, which is < the version used in this repo
// https://github.com/pytorch/pytorch/pull/116594
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

// ttnn label    -> torch convert egress
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

// ttnn label    -> numpy convert egress
// uint8         -> ubyte
// uint16        -> int16
// int32         -> int32
// uint32        -> int32
// bfloat4_b     -> float32
// bfloat8_b     -> float32
// float32       -> float32

namespace nb = nanobind;
namespace nbd = nb::detail;
namespace nbdlp = nb::dlpack;

using tt::tt_metal::DataType;

constexpr uint32_t nbdlp_dtype_to_int(nbdlp::dtype dt) noexcept {
    // uint8_t code, uint8_t bits, uint16_t lanes
    return std::bit_cast<uint32_t>(dt);
}

static_assert(
    []() constexpr noexcept {
        constexpr nbdlp::dtype start{1, 2, 3};
        constexpr auto to_int = std::bit_cast<uint32_t>(start);
        constexpr auto to_dtype = std::bit_cast<nbdlp::dtype>(to_int);
        return start == to_dtype;
    }(),
    "nanobind::dlpack::dtype definition changed! No longer trivially mappable to uint32_t!");

// map dtypes to a uint32 so we can switch over them
enum class DtypeID : uint32_t {
    UINT64 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 64, .lanes = 1}),
    INT64 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Int), .bits = 64, .lanes = 1}),
    UINT32 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 32, .lanes = 1}),
    INT32 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Int), .bits = 32, .lanes = 1}),
    UINT16 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 16, .lanes = 1}),
    INT16 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Int), .bits = 16, .lanes = 1}),
    UINT8 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 8, .lanes = 1}),
    INT8 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Int), .bits = 8, .lanes = 1}),
    FLOAT64 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 64, .lanes = 1}),
    FLOAT32 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 32, .lanes = 1}),
    FLOAT16 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 16, .lanes = 1}),
    BFLOAT16 = nbdlp_dtype_to_int(
        nbdlp::dtype{.code = static_cast<std::uint8_t>(nbdlp::dtype_code::Bfloat), .bits = 16, .lanes = 1}),
    INVALID = 0,
};

template <DtypeID nb_dtype, typename = int>  // sfinae to allow for partial specializations
struct py_to_ {
    constexpr static auto ttnn_DataType = DataType::INVALID;
};

// for some reason ttnn maps float16 to bfloat16
template <>
struct py_to_<DtypeID::FLOAT16> {
    constexpr static auto ttnn_DataType = DataType::BFLOAT16;
};

template <>
struct py_to_<DtypeID::FLOAT32> {
    constexpr static auto ttnn_DataType = DataType::FLOAT32;
};

template <>
struct py_to_<DtypeID::BFLOAT16> {
    constexpr static auto ttnn_DataType = DataType::BFLOAT16;
};

template <>
struct py_to_<DtypeID::INT64> {
    constexpr static auto ttnn_DataType = DataType::UINT32;
};

template <>
struct py_to_<DtypeID::INT32> {
    constexpr static auto ttnn_DataType = DataType::INT32;
};

template <>
struct py_to_<DtypeID::INT16> {
    constexpr static auto ttnn_DataType = DataType::UINT16;
};

template <>
struct py_to_<DtypeID::UINT8> {
    constexpr static auto ttnn_DataType = DataType::UINT8;
};

template <>
struct py_to_<DtypeID::UINT32> {
    constexpr static auto ttnn_DataType = DataType::UINT32;
};

template <>
struct py_to_<DtypeID::UINT16> {
    constexpr static auto ttnn_DataType = DataType::UINT16;
};

template <DataType dt>
struct ttnn_datatype_traits {
    using underlying_type = void;
    static constexpr nbdlp::dtype value{.code = 0, .bits = 0, .lanes = 0};
    static constexpr auto name = nbd::descr<0>();
};

template <>
struct ttnn_datatype_traits<DataType::INVALID> {
    using underlying_type = void;
    static constexpr nbdlp::dtype value{.code = 0, .bits = 0, .lanes = 0};
    static constexpr auto name = nbd::const_name("INVALID");
};

template <>
struct ttnn_datatype_traits<DataType::BFLOAT16> {
    using underlying_type = ::bfloat16;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Bfloat), .bits = 16, .lanes = 1};
    static constexpr auto name = nbd::const_name("BFLOAT16");
};

template <>
struct ttnn_datatype_traits<DataType::FLOAT32> {
    using underlying_type = float;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 32, .lanes = 1};
    static constexpr auto name = nbd::const_name("FLOAT32");
};

template <>
struct ttnn_datatype_traits<DataType::UINT32> {
    using underlying_type = std::uint32_t;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 32, .lanes = 1};
    static constexpr auto name = nbd::const_name("UINT32");
};

// bfloat8/4_b are treated as float32 on the host side
template <>
struct ttnn_datatype_traits<DataType::BFLOAT8_B> {
    using underlying_type = float;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 32, .lanes = 1};
    static constexpr auto name = nbd::const_name("BFLOAT8_B");
};

template <>
struct ttnn_datatype_traits<DataType::BFLOAT4_B> {
    using underlying_type = float;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Float), .bits = 32, .lanes = 1};
    static constexpr auto name = nbd::const_name("BFLOAT4_B");
};

// use this if we get proper 3rd party support for bfloat8/4_b
// template <>
// struct ttnn_datatype_traits<DataType::BFLOAT8_B> {
//    using underlying_type = float;
//    static constexpr nbdlp::dtype value{
//        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Bfloat), .bits = 8, .lanes = 1};
//    static constexpr auto name = nbd::const_name("BFLOAT8_B");
//};
//
// template <>
// struct ttnn_datatype_traits<DataType::BFLOAT4_B> {
//    using underlying_type = float;
//    static constexpr nbdlp::dtype value{
//        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Bfloat), .bits = 4, .lanes = 1};
//    static constexpr auto name = nbd::const_name("BFLOAT4_B");
//};

template <>
struct ttnn_datatype_traits<DataType::UINT8> {
    using underlying_type = std::uint8_t;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 8, .lanes = 1};
    static constexpr auto name = nbd::const_name("UINT8");
};

template <>
struct ttnn_datatype_traits<DataType::UINT16> {
    using underlying_type = std::uint16_t;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::UInt), .bits = 16, .lanes = 1};
    static constexpr auto name = nbd::const_name("UINT16");
};

template <>
struct ttnn_datatype_traits<DataType::INT32> {
    using underlying_type = std::int32_t;
    static constexpr nbdlp::dtype value{
        .code = static_cast<std::uint8_t>(nbdlp::dtype_code::Int), .bits = 32, .lanes = 1};
    static constexpr auto name = nbd::const_name("INT32");
};

[[nodiscard]]
constexpr nbdlp::dtype get_dtype_from_ttnn_datatype(DataType dt) noexcept {
    switch (dt) {
        case DataType::BFLOAT16: return ttnn_datatype_traits<DataType::BFLOAT16>::value;
        case DataType::FLOAT32: return ttnn_datatype_traits<DataType::FLOAT32>::value;
        case DataType::UINT32: return ttnn_datatype_traits<DataType::UINT32>::value;
        case DataType::BFLOAT8_B: return ttnn_datatype_traits<DataType::BFLOAT8_B>::value;
        case DataType::BFLOAT4_B: return ttnn_datatype_traits<DataType::BFLOAT4_B>::value;
        case DataType::UINT8: return ttnn_datatype_traits<DataType::UINT8>::value;
        case DataType::UINT16: return ttnn_datatype_traits<DataType::UINT16>::value;
        case DataType::INT32: return ttnn_datatype_traits<DataType::INT32>::value;
        case DataType::INVALID: [[fallthrough]];
        default: TT_THROW("get_dtype_from_ttnn_datatype: got INVALID or unhandled DataType.");
    }

    return {};
}

[[nodiscard]]
constexpr DataType get_ttnn_datatype_from_dtype(nbdlp::dtype dt) noexcept {
    switch (static_cast<DtypeID>(nbdlp_dtype_to_int(dt))) {
        case DtypeID::UINT64: return py_to_<DtypeID::UINT64>::ttnn_DataType;
        case DtypeID::INT64: return py_to_<DtypeID::INT64>::ttnn_DataType;
        case DtypeID::UINT32: return py_to_<DtypeID::UINT32>::ttnn_DataType;
        case DtypeID::INT32: return py_to_<DtypeID::INT32>::ttnn_DataType;
        case DtypeID::UINT16: return py_to_<DtypeID::UINT16>::ttnn_DataType;
        case DtypeID::INT16: return py_to_<DtypeID::INT16>::ttnn_DataType;
        case DtypeID::UINT8: return py_to_<DtypeID::UINT8>::ttnn_DataType;
        case DtypeID::INT8: return py_to_<DtypeID::INT8>::ttnn_DataType;
        case DtypeID::FLOAT64: return py_to_<DtypeID::FLOAT64>::ttnn_DataType;
        case DtypeID::FLOAT32: return py_to_<DtypeID::FLOAT32>::ttnn_DataType;
        case DtypeID::FLOAT16: return py_to_<DtypeID::FLOAT16>::ttnn_DataType;
        case DtypeID::BFLOAT16: return py_to_<DtypeID::BFLOAT16>::ttnn_DataType;
        default:
            TT_THROW("get_ttnn_datatype_from_dtype: got unexpected dlpack dtype. code: {}, bits: {}", dt.code, dt.bits);
    }

    return DataType::INVALID;
}

constexpr PyDType get_PyDType_from_dtype(nb::dlpack::dtype dt) {
    switch (static_cast<DtypeID>(nbdlp_dtype_to_int(dt))) {
        case DtypeID::UINT64: return PyDType::UINT64;
        case DtypeID::INT64: return PyDType::INT64;
        case DtypeID::UINT32: return PyDType::UINT32;
        case DtypeID::INT32: return PyDType::INT32;
        case DtypeID::UINT16: return PyDType::UINT16;
        case DtypeID::INT16: return PyDType::INT16;
        case DtypeID::UINT8: return PyDType::UINT8;
        case DtypeID::INT8: return PyDType::INT8;
        case DtypeID::FLOAT64: return PyDType::FLOAT64;
        case DtypeID::FLOAT32: return PyDType::FLOAT32;
        case DtypeID::FLOAT16: return PyDType::FLOAT16;
        case DtypeID::BFLOAT16: return PyDType::BFLOAT16;
        default:
            TT_THROW("get_ttnn_datatype_from_dtype: got unexpected dlpack dtype. code: {}, bits: {}", dt.code, dt.bits);
    }

    ttsl::unreachable();
}

}  // namespace ttnn_dtype_traits::detail

using ttnn_dtype_traits::detail::get_dtype_from_ttnn_datatype;
using ttnn_dtype_traits::detail::get_PyDType_from_dtype;
using ttnn_dtype_traits::detail::get_ttnn_datatype_from_dtype;
}  // namespace ttnn
