// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nd_sharding_utils.hpp"

#include <algorithm>

namespace tt::tt_metal {

std::vector<uint8_t> prepare_sharded_data(
    tt::stl::Span<uint8_t> data, const TensorSpec& tensor_spec, size_t element_size_bytes) {
    if (tensor_spec.padded_shape().volume() == 0) {
        return {};
    }

    const auto& memory_config = tensor_spec.memory_config();
    const auto& shape = tensor_spec.padded_shape();
    const auto& physical_shape = tensor_spec.physical_shape();
    const auto& strides = tensor_spec.compute_strides();
    const auto& shard_spec = memory_config.nd_shard_spec().value();
    const auto& shard_shape = shard_spec.shard_shape;
    auto shard_size = shard_shape.volume();
    size_t shard_width = shard_shape[-1];

    size_t num_shards = 1;
    tt::stl::SmallVector<size_t> num_shards_per_dim(shape.rank());
    for (size_t i = 0; i < shape.rank(); i++) {
        num_shards_per_dim[i] = (shape[i] + shard_shape[i] - 1) / shard_shape[i];
        num_shards *= num_shards_per_dim[i];
    }
    size_t num_cores = shard_spec.grid.num_cores();
    size_t num_shards_per_core = (num_shards + num_cores - 1) / num_cores;

    tt::stl::SmallVector<size_t> shard_strides(shape.rank());
    shard_strides.back() = 1;
    for (int i = static_cast<int>(shape.rank()) - 2; i >= 0; i--) {
        shard_strides[i] = shard_strides[i + 1] * shard_shape[i + 1];
    }

    tt::stl::SmallVector<size_t> shard_index_strides(shape.rank());
    shard_index_strides.back() = 1;
    for (int i = static_cast<int>(num_shards_per_dim.size()) - 2; i >= 0; i--) {
        shard_index_strides[i] = shard_index_strides[i + 1] * num_shards_per_dim[i + 1];
    }

    std::vector<uint8_t> sharded_data(num_shards_per_core * num_cores * shard_size * element_size_bytes);

    for (size_t row_idx = 0; row_idx < physical_shape.height(); row_idx++) {
        for (size_t col_block_idx = 0; col_block_idx < num_shards_per_dim.back(); col_block_idx++) {
            size_t element_idx = row_idx * physical_shape.width() + col_block_idx * shard_width;

            size_t src_offset = 0;
            size_t offset_within_shard = 0;
            size_t shard_idx = 0;
            size_t element_idx_tmp = element_idx;
            for (int i = static_cast<int>(shape.rank()) - 1; i >= 0; i--) {
                size_t element_coord = element_idx_tmp % shape[i];
                src_offset += element_coord * strides[i];
                size_t shard_coord = element_coord / shard_shape[i];
                shard_idx += shard_coord * shard_index_strides[i];
                size_t coord_within_shard = element_coord % shard_shape[i];
                offset_within_shard += coord_within_shard * shard_strides[i];
                element_idx_tmp /= shape[i];
            }

            size_t core_idx = shard_idx % num_cores;
            size_t shard_idx_within_core = shard_idx / num_cores;
            size_t shard_offset = (num_shards_per_core * core_idx + shard_idx_within_core) * shard_size;
            size_t dst_offset = shard_offset + offset_within_shard;

            size_t num_bytes_to_copy = shard_width * element_size_bytes;
            bool last_shard_in_row = col_block_idx == num_shards_per_dim.back() - 1;
            if (last_shard_in_row && shape[-1] % shard_width != 0) {
                num_bytes_to_copy = (shape[-1] % shard_width) * element_size_bytes;
            }

            std::memcpy(
                sharded_data.data() + dst_offset * element_size_bytes,
                data.data() + src_offset * element_size_bytes,
                num_bytes_to_copy);
        }
    }

    return sharded_data;
}

}  // namespace tt::tt_metal
