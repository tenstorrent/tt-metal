// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "alignment.hpp"
#include "size.hpp"
#include "page_config.hpp"

#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace tt::tt_metal {

using Strides = std::vector<size_t>;

// TensorLayout describes how a tensor is laid out in memory
// It takes datatype, layout (eg. TILE vs. RM), memory (eg. DRAM vs. L1), sharding (ie. how you want to cut your logical shape)
// And provides information required to physically lay out the tensor in memory
class TensorLayout {
public:
    TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config);

    // static method makes it easy to find and remove all of its usages in the codebase - thats why it is not a constructor
    [[deprecated("Use of Legacy Padded Shape is deprecated")]]
    static TensorLayout fromLegacyPaddedShape(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::Shape& legacy_shape);

    Layout get_layout() const { return page_config_.is_row_major() ? Layout::ROW_MAJOR : Layout::TILE; }
    PageConfig get_page_config() const { return page_config_; }
    DataType get_data_type() const { return dtype_; }
    const MemoryConfig& get_memory_config() const { return memory_config_; }
    const Alignment& get_alignment() const { return alignment_; }

    Strides compute_strides(const ttnn::SimpleShape& shape) const;

    std::optional<ShardSpecBuffer> compute_shard_spec_buffer(const ttnn::SimpleShape& shape) const;

    size_t compute_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const;
    size_t compute_page_size_bytes(const ttnn::SimpleShape& shape) const;

    // This method is deprecated and should be replaced with get_strides() / get_physical_size()
    // It computes padded shape on the fly from shape and alignment
    [[deprecated("Use of LegacyPaddedShape is deprecated. Please use get_physical_size() or get_strides() instead.")]]
    ttnn::SimpleShape compute_padded_shape(const ttnn::SimpleShape& shape) const;

    // Returns number of elements laid out in physically memory across H:W dimensions
    //  W is row width aligned to page width and shard width, depends on data type
    //  H is all dimensions except W multiplied and aligned to tile and shard height
    Size compute_physical_shape(const ttnn::SimpleShape& shape) const;

private:
    // Private to not expose alignment parameter to the public API
    TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

    void initialize_alignment();
    void validate_alignment() const;

    Size compute_page_shape(const Size& physical_size) const;
    size_t compute_page_size_bytes(const Size& page_size) const;

    DataType dtype_ = DataType::BFLOAT16;
    PageConfig page_config_;
    MemoryConfig memory_config_;
    Alignment alignment_;
};

} // tt::tt_metal
