// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape2d.hpp>

#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/alignment.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal {

class IDevice;

using Strides = std::vector<size_t>;

// TensorLayout describes how a tensor is laid out in memory
// It takes datatype, layout (eg. TILE vs. RM), memory (eg. DRAM vs. L1), sharding (ie. how you want to cut your logical
// shape) And provides information required to physically lay out the tensor in memory
class TensorLayout {
public:
    TensorLayout(
        DataType dtype,
        const PageConfig& page_config,
        const MemoryConfig& memory_config,
        const Alignment& alignment = {});

    // static method makes it easy to find and remove all of its usages in the codebase - thats why it is not a
    // constructor
    [[deprecated("Use of Padded Shape is deprecated")]]
    static TensorLayout fromPaddedShape(
        DataType dtype,
        const PageConfig& page_config,
        const MemoryConfig& memory_config,
        const ttnn::Shape& logical_shape,
        const ttnn::Shape& padded_shape);

    Layout get_layout() const { return page_config_.get_layout(); }
    Tile get_tile() const { return page_config_.get_tile(); }
    PageConfig get_page_config() const { return page_config_; }
    DataType get_data_type() const { return dtype_; }
    const MemoryConfig& get_memory_config() const { return memory_config_; }
    const Alignment& get_alignment() const { return alignment_; }

    Strides compute_strides(const ttnn::Shape& shape) const;

    BufferShardingArgs compute_buffer_sharding_args(const ttnn::Shape& shape) const;

    size_t compute_packed_buffer_size_bytes(const ttnn::Shape& shape) const;
    size_t compute_page_size_bytes(const ttnn::Shape& shape) const;

    size_t compute_consumed_memory_bytes_per_bank(const ttnn::Shape& shape, const IDevice& device) const;
    size_t compute_consumed_memory_bytes_per_bank(
        const ttnn::Shape& shape, size_t page_alignment, size_t num_banks) const;

    // This method is deprecated and should be replaced with get_strides() / get_physical_size()
    // It computes padded shape on the fly from shape and alignment
    [[deprecated("Use of LegacyPaddedShape is deprecated. Please use get_physical_size() or get_strides() instead.")]]
    ttnn::Shape compute_padded_shape(const ttnn::Shape& shape) const;

    // Flattens input shape into height and width
    // - Height is accumulated over all dims except last
    // - Width is equal to the last dim
    Shape2D compute_logical_2d_shape(const ttnn::Shape& shape) const;

    // Returns number of elements laid out in physically memory across H:W dimensions
    //  W is row width aligned to page width and shard width, depends on data type
    //  H is all dimensions except W multiplied and aligned to tile and shard height
    Shape2D compute_physical_shape(const ttnn::Shape& shape) const;

    // Returns logical shard shape from shard spec shape
    Shape2D get_logical_shard_shape() const;

    // Returns physical shard shape based on ShardMode, shard shape, and alignment
    Shape2D get_physical_shard_shape() const;

    Shape2D compute_page_shape(const Shape2D& physical_size) const;
    size_t compute_page_size_bytes(const Shape2D& page_size) const;

    TensorLayout with_memory_config(MemoryConfig memory_config) const {
        TensorLayout result = *this;
        result.memory_config_ = std::move(memory_config);
        return result;
    }

    bool operator==(const TensorLayout&) const = default;
    bool operator!=(const TensorLayout&) const = default;

    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "page_config", "memory_config", "alignment");
    auto attribute_values() const { return std::forward_as_tuple(dtype_, page_config_, memory_config_, alignment_); }

    static TensorLayout restore_from_serialized(
        DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

private:
    void initialize_alignment();

    DataType dtype_ = DataType::BFLOAT16;
    PageConfig page_config_;
    MemoryConfig memory_config_;
    Alignment alignment_;
};

}  // namespace tt::tt_metal
