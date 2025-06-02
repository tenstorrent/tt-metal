// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

std::vector<std::byte> pack_nd_sharded_data(
    tt::stl::Span<const std::byte> data, const TensorSpec& tensor_spec, size_t element_size_bytes);

std::vector<std::byte> unpack_nd_sharded_data(
    tt::stl::Span<const std::byte> sharded_data, const TensorSpec& tensor_spec, size_t element_size_bytes);

template <typename T>
std::vector<std::byte> pack_nd_sharded_data(tt::stl::Span<const T> data, const TensorSpec& tensor_spec) {
    return pack_nd_sharded_data(tt::stl::as_bytes(data), tensor_spec, sizeof(T));
}

template <typename T>
std::vector<std::byte> unpack_nd_sharded_data(tt::stl::Span<const T> sharded_data, const TensorSpec& tensor_spec) {
    return unpack_nd_sharded_data(tt::stl::as_bytes(sharded_data), tensor_spec, sizeof(T));
}

}  // namespace tt::tt_metal
