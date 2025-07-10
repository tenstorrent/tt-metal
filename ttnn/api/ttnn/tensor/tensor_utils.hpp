// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "types.hpp"

namespace tt {

namespace tt_metal {

ttnn::Shape infer_dims_for_reshape(const Tensor& tensor, tt::stl::Span<const int32_t> shape);

int compute_flat_indices(tt::stl::Span<const int> indices, tt::stl::Span<const uint32_t> strides);

std::size_t compute_buffer_size(const ttnn::Shape& shape, DataType data_type, const Tile& tile);

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

// Returns true if tensor has MultiDeviceHost storage.
// TODO: #19177 - Remove this once host and multi-device host tensors are unified.
bool is_multi_device_host_tensor(const Tensor& tensor);

// Returns true if tensor is on device.
bool is_device_tensor(const Tensor& tensor);

// Given a multi-device host tensor and a function that transforms a tensor, applies the function to all per-device
// tensors.
Tensor transform(const Tensor& tensor, const std::function<Tensor(const Tensor&)>& transform_func);

// Given a multi-device host tensor and a callable, applies the function to all per-device tensors.
void apply(const Tensor& tensor, const std::function<void(const Tensor&)>& callable);

template <class T>
uint32_t get_batch_size(const T& shape) {
    uint32_t result = 1;
    for (int i = 0; i < (int)shape.rank() - 2; i++) {
        result *= shape[i];
    }
    return result;
}

// Useful information about how a shard_shape cuts a 2D shape
// - num_shards_height: Number of shards along the height (including partial last shard, if any)
// - last_shard_height: Height of last partial shard (if None, it will be same as full shard shape height)
// - num_shards_width: Number of shards along the width (including partial last shard, if any)
// - last_shard_width: Width of last partial shard (if None, it will be same as full shard shape width)
struct ShardDivisionSpec {
    size_t num_shards_height = 0;
    size_t last_shard_height = 0;
    size_t num_shards_width = 0;
    size_t last_shard_width = 0;
};

// Returns ShardDivisionSpecs given 2D shape and shard_shape
ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape);

}  // namespace tt_metal
}  // namespace tt
