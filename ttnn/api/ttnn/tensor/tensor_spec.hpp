// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

namespace tt::tt_metal {

class TensorSpec final {
public:
    TensorSpec(ttnn::Shape logical_shape, TensorLayout tensor_layout);
    TensorSpec(TensorSpec&&) noexcept = default;
    TensorSpec& operator=(TensorSpec&&) = default;
    TensorSpec(const TensorSpec&) = default;
    TensorSpec& operator=(const TensorSpec&) = default;
    bool operator==(const TensorSpec&) const = default;
    bool operator!=(const TensorSpec&) const = default;

    const ttnn::Shape& logical_shape() const { return logical_shape_; }
    const TensorLayout& tensor_layout() const { return tensor_layout_; }
    DataType data_type() const { return tensor_layout_.get_data_type(); }
    Layout layout() const { return tensor_layout_.get_layout(); }
    PageConfig page_config() const { return tensor_layout_.get_page_config(); }
    const MemoryConfig& memory_config() const { return tensor_layout_.get_memory_config(); }
    const ttnn::Shape& padded_shape() const { return cached_padded_shape_; }
    const Shape2D& logical_2d_shape() const { return cached_logical_2d_shape_; }
    const Shape2D& physical_shape() const { return cached_physical_shape_; }

    Tile tile() const { return tensor_layout_.get_tile(); }

    /// Shards TensorSpec across the specified dimensions.
    /// This would result in the shard shape to be minimal (typically 1 or tile size) in the sharded dimensions.
    TensorSpec sharded_across_dims(
        tt::stl::Span<const int32_t> dims,
        CoreRangeSet grid,
        ShardOrientation orientation = ShardOrientation::ROW_MAJOR) const;
    /// Shards TensorSpec across all dimensions except for the specified ones.
    /// This would result in the shard shape to be minimal (typically 1 or tile size) in all dimensions except for the
    /// specified ones.
    TensorSpec sharded_across_dims_except(
        tt::stl::Span<const int32_t> dims,
        CoreRangeSet grid,
        ShardOrientation orientation = ShardOrientation::ROW_MAJOR) const;
    /// Performs 2D height sharding for TensorSpec.
    /// This flattens the tensor into a 2D shape and splits it along the height to achieve as close to equal
    /// distribution as possible, while maintaining just 1 shard per core.
    TensorSpec height_sharded(CoreRangeSet grid, ShardOrientation orientation = ShardOrientation::ROW_MAJOR) const;
    /// Performs 2D width sharding for TensorSpec.
    /// This flattens the tensor into a 2D shape and splits it along the width to achieve as close to equal distribution
    /// as possible, while maintaining just 1 shard per core.
    TensorSpec width_sharded(CoreRangeSet grid, ShardOrientation orientation = ShardOrientation::ROW_MAJOR) const;
    /// Performs 2D block sharding for TensorSpec.
    /// This flattens the tensor into a 2D shape and splits it into 2D contiguous blocks, putting each block onto the
    /// corresponding core in 2D grid.
    TensorSpec block_sharded(CoreRange grid, ShardOrientation orientation = ShardOrientation::ROW_MAJOR) const;

    enum class ShardShapeAlignment {
        /// No shard shape alignment will be performed. If the shard shape is not following the alignment requirements,
        /// an exception will be thrown.
        NONE,
        /// Shard shape will be automatically aligned to the minimum required alignment. The Required alignment may
        /// cause higher memory usage and lower read/write performance for some use cases.
        REQUIRED,
        /// Shard shape will be automatically aligned to the recommended alignment, trying to achieve optimal
        /// performance and memory usage.
        RECOMMENDED,
    };
    /// Performs arbitrary sharding for TensorSpec using the specified shard shape, grid, shard shape alignment, and
    /// other optional parameters.
    TensorSpec sharded(
        Shape shard_shape,
        CoreRangeSet grid,
        ShardShapeAlignment shard_alignment,
        ShardOrientation orientation = ShardOrientation::ROW_MAJOR,
        ShardDistributionStrategy shard_distribution_strategy = ShardDistributionStrategy::ROUND_ROBIN_1D) const;
    /// Performs arbitrary sharding for TensorSpec using the specified shard spec and shard shape alignment.
    TensorSpec sharded(NdShardSpec nd_shard_spec, ShardShapeAlignment shard_alignment) const;

    Strides compute_strides() const { return tensor_layout_.compute_strides(logical_shape_); }
    BufferShardingArgs compute_buffer_sharding_args() const {
        return tensor_layout_.compute_buffer_sharding_args(logical_shape_);
    }
    size_t compute_packed_buffer_size_bytes() const {
        return tensor_layout_.compute_packed_buffer_size_bytes(logical_shape_);
    }
    size_t compute_page_size_bytes() const { return tensor_layout_.compute_page_size_bytes(logical_shape_); }

    size_t compute_consumed_memory_bytes_per_bank(const IDevice& device) const {
        return tensor_layout_.compute_consumed_memory_bytes_per_bank(logical_shape_, device);
    }
    size_t compute_consumed_memory_bytes_per_bank(size_t page_alignment, size_t num_banks) const {
        return tensor_layout_.compute_consumed_memory_bytes_per_bank(logical_shape_, page_alignment, num_banks);
    }

    TensorSpec with_memory_config(MemoryConfig memory_config) const;

    static constexpr auto attribute_names = std::forward_as_tuple("logical_shape", "tensor_layout");
    auto attribute_values() const { return std::forward_as_tuple(logical_shape_, tensor_layout_); }

private:
    void populate_sharding_specs();
    MemoryConfig populate_nd_shard_spec_from_legacy() const;
    std::optional<MemoryConfig> populate_legacy_shard_spec_from_nd() const;

    ttnn::Shape logical_shape_;
    TensorLayout tensor_layout_;

    ttnn::Shape cached_padded_shape_;
    Shape2D cached_logical_2d_shape_;
    Shape2D cached_physical_shape_;
};

}  // namespace tt::tt_metal
