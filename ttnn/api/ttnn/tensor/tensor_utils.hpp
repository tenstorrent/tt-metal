// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {

tt::tt_metal::Shape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape);

int compute_flat_indices(tt::stl::Span<const int> indices, tt::stl::Span<const size_t> strides);

std::size_t compute_buffer_size(const tt::tt_metal::Shape& shape, DataType data_type, const Tile& tile);

constexpr auto compute_flat_input_index = [](const auto& indices, const auto& strides) {
    uint32_t flat_index = 0;
    for (auto i = 0; i < indices.size(); i++) {
        flat_index += indices[i] * strides[i];
    }
    return flat_index;
};

// Returns true if architecture is GRAYSKULL.
bool is_arch_gs(const tt::ARCH& arch);

// Returns true if architecture is WORMHOLE_B0.
bool is_arch_whb0(const tt::ARCH& arch);

// Returns true if tensor has Host storage.
bool is_cpu_tensor(const Tensor& tensor);

// Returns true if tensor is on device.
bool is_device_tensor(const Tensor& tensor);

template <class T>
uint32_t get_batch_size(const T& shape) {
    uint32_t result = 1;
    for (int i = 0; i < (int)shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

}  // namespace tt_metal
}  // namespace tt
