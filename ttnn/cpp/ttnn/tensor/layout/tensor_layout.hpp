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

    // This is a static method instead of a constructor to make it easy to find and remove all of its usages in the codebase.
    [[deprecated("Use of LegacyPaddedShape is deprecated. Please use constructor with Alignment instead.")]]
    static TensorLayout fromLegacyPaddedShape(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::SimpleShape& legacy_padded_shape);

    Layout get_layout() const { return m_page_config.is_row_major() ? Layout::ROW_MAJOR : Layout::TILE; }
    PageConfig get_page_config() const { return m_page_config; }
    DataType get_data_type() const { return m_dtype; }
    const MemoryConfig& get_memory_config() const { return m_memory_config; }
    const ttnn::Alignment& get_alignment() const { return m_alignment; }

    Strides get_strides(const ttnn::SimpleShape& shape) const;

    std::optional<ShardSpecBuffer> get_shard_spec_buffer(const ttnn::SimpleShape& shape) const;

    size_t get_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const;
    size_t get_page_size_bytes(const ttnn::SimpleShape& shape) const;

    // This method is deprecated and should be replaced with get_strides() / get_physical_size()
    // It computes padded shape on the fly from shape and alignment
    [[deprecated("Use of LegacyPaddedShape is deprecated. Please use get_physical_size() or get_strides() instead.")]]
    ttnn::SimpleShape get_padded_shape(const ttnn::SimpleShape& shape) const;

    // Returns number of elements laid out in physically memory across H:W dimensions
    //  W is row width aligned to page width and shard width, depends on data type
    //  H is all dimensions except W multiplied and aligned to tile and shard height
    Size get_physical_shape(const ttnn::SimpleShape& shape) const;

private:
    // Private to not expose alignment parameter to the public API
    TensorLayout(DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const ttnn::Alignment& alignment);

    void initialize_alignment();
    void validate_alignment() const;

    uint32_t get_header_size_bytes() const;
    uint32_t get_page_elements_count(const ttnn::SimpleShape& shape) const;

    Size get_page_shape(const Size& physical_size) const;
    size_t get_page_size_bytes(const Size& page_size) const;

    DataType m_dtype = DataType::BFLOAT16;
    PageConfig m_page_config;
    MemoryConfig m_memory_config;
    ttnn::Alignment m_alignment;
};

} // tt::tt_metal
