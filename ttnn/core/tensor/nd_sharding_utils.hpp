// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

std::vector<uint8_t> pack_sharded_data(
    tt::stl::Span<uint8_t> data, const TensorSpec& tensor_spec, size_t element_size_bytes);

std::vector<uint8_t> unpack_sharded_data(
    tt::stl::Span<uint8_t> sharded_data, const TensorSpec& tensor_spec, size_t element_size_bytes);

template <typename T>
std::vector<uint8_t> pack_sharded_data(tt::stl::Span<T> data, const TensorSpec& tensor_spec) {
    return pack_sharded_data(
        tt::stl::Span<uint8_t>(reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(T)),
        tensor_spec,
        sizeof(T));
}

template <typename T>
std::vector<uint8_t> unpack_sharded_data(tt::stl::Span<T> sharded_data, const TensorSpec& tensor_spec) {
    return unpack_sharded_data(
        tt::stl::Span<uint8_t>(reinterpret_cast<uint8_t*>(sharded_data.data()), sharded_data.size() * sizeof(T)),
        tensor_spec,
        sizeof(T));
}

}  // namespace tt::tt_metal
