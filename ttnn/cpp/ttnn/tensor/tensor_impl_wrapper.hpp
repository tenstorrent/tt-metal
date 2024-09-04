// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_impl.hpp"

namespace tt::tt_metal::tensor_impl {

// Utility to convert runtime DataType to compile-time constant and dispatch the function call
template <typename Func, typename... Args>
auto dispatch(DataType dtype, Func &&func, Args &&...args) {
    switch (dtype) {
        case DataType::BFLOAT16: return func.template operator()<bfloat16>(static_cast<Args &&>(args)...);
        case DataType::FLOAT32: return func.template operator()<float>(static_cast<Args &&>(args)...);
        case DataType::INT32: return func.template operator()<int32_t>(static_cast<Args &&>(args)...);
        case DataType::UINT32: return func.template operator()<uint32_t>(static_cast<Args &&>(args)...);
        case DataType::UINT16: return func.template operator()<uint16_t>(static_cast<Args &&>(args)...);
        case DataType::UINT8: return func.template operator()<uint8_t>(static_cast<Args &&>(args)...);
        case DataType::BFLOAT8_B: return func.template operator()<bfloat8_b>(static_cast<Args &&>(args)...);
        case DataType::BFLOAT4_B: return func.template operator()<bfloat4_b>(static_cast<Args &&>(args)...);
        default: TT_THROW("Unsupported data type");
    }
}

#define AS_LAMBDA(func) []<typename T>(auto &&...args) { return func<T>(std::forward<decltype(args)>(args)...); }

#define WRAP_FUNCTION(func)                                                                                         \
    template <typename... Args>                                                                                     \
    auto func##_wrapper(Args &&...args) {                                                                           \
        return dispatch(                                                                                            \
            std::get<0>(std::forward_as_tuple(args...)).get_dtype(), AS_LAMBDA(func), std::forward<Args>(args)...); \
    }

inline uint32_t packed_buffer_size_bytes_wrapper(DataType dtype, uint32_t volume_unpacked_data) {
    return dispatch(dtype, AS_LAMBDA(packed_buffer_size_bytes), volume_unpacked_data);
}

WRAP_FUNCTION(to_host)
WRAP_FUNCTION(extract_shard)
WRAP_FUNCTION(to_host_sharded)
WRAP_FUNCTION(to_device)
WRAP_FUNCTION(to_layout)
WRAP_FUNCTION(pad)
WRAP_FUNCTION(unpad)
WRAP_FUNCTION(to_string)

#undef WRAP_FUNCTION
#undef AS_LAMBDA

}  // namespace tt::tt_metal::tensor_impl
