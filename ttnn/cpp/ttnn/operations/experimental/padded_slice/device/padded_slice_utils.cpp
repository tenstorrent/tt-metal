// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_utils.hpp"

#include <bit>
#include <cstdint>
#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor) {
    auto shard_spec = tensor.shard_spec().value();
    auto core_grid = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    uint32_t num_cores_channels = 1;
    if (tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        if (rm_orientation) {
            num_cores_channels = core_grid.bounding_box().grid_size().x;
        } else {
            num_cores_channels = core_grid.bounding_box().grid_size().y;
        }
    } else if (tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = core_grid.num_cores();
    }
    return num_cores_channels;
}

uint32_t pad_value_to_element_bits(const std::optional<float>& pad_value, DataType output_dtype) {
    if (!pad_value.has_value()) {
        return 0;
    }
    const float value = *pad_value;
    switch (output_dtype) {
        case DataType::FLOAT32: return std::bit_cast<uint32_t>(value);
        case DataType::BFLOAT16: return fp32_to_bf16_bits_round_to_nearest_even(value);
        case DataType::INT32: return static_cast<uint32_t>(static_cast<int32_t>(value));
        case DataType::UINT32: return static_cast<uint32_t>(value);
        case DataType::UINT16: return static_cast<uint16_t>(value);
        case DataType::UINT8: return static_cast<uint8_t>(value);
        default: TT_THROW("pad_value is not supported for output dtype {} in padded_slice", output_dtype);
    }
}

}  // namespace ttnn::operations::experimental::detail
