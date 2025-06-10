// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
namespace host_buffer {

template <typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_FATAL(
            tensor.dtype() == DataType::UINT32 or tensor.dtype() == DataType::BFLOAT8_B or
                tensor.dtype() == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            tensor.dtype());
    } else if constexpr (std::is_same_v<T, int32_t>) {
        TT_FATAL(tensor.dtype() == DataType::INT32, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<T, float>) {
        TT_FATAL(tensor.dtype() == DataType::FLOAT32, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT16, "Incorrect data type {}", tensor.dtype());
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        TT_FATAL(tensor.dtype() == DataType::UINT8, "Incorrect data type {}", tensor.dtype());
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType");
    }
}

HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
tt::stl::Span<const T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

}  // namespace host_buffer
}  // namespace tt::tt_metal
