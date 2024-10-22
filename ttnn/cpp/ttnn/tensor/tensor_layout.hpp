// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/tile/tile.hpp"
#include "types.hpp"
#include "enum_types.hpp"

#include <cstddef>
#include <ostream>
#include <variant>

#include "shape.hpp"
#include "alignment.hpp"
#include "size.hpp"
#include "page_config.hpp"

namespace tt::tt_metal {

using Strides = std::vector<size_t>;

// Alignment is a physical row alignment for each dimension of the tensor (except innermost dimension - there its a column alignment).
// Example:
//  Given Tensor with shape (2, 3, 10, 16) and Alignment (A:768, B:300, C:32, 32) we will physically allocate
//     W = round_up(16, 32) = 32
//     H_R1 = round_up(10, C:32) = 32
//     H_R2 = round_up(R1:32 * 3, B:300) = round_up(96, 300) = 300
//     H_R3 = = round_up(R2:300 * 2, A:768) = round_up(600, 768) = 768
//     PHY = [H:768, W:32]
//
// LegacyPaddedShape is a padding shape for each dimension of the tensor.
// All tensor calculations (like physical_size or strides) are done using Alignment, so internally we always convert LegacyPaddedShape to Alignment.
// Example:
//   Given tensor with shape (2, 3, 10, 16) and LegacyPaddedShape (2, 4, 32, 32).
//   PHY_LEGACY = [2*4*32, 32] = [256, 32]
//   Lets convert LegacyPaddedShape to Alignment:
//   Alignment = (2*4*32, 4*32, 32, 32) = (A:256, B:128, C:32, 32)
//   W = round_up(16, 32) = 32
//   H_R1 = round_up(10, C:32) = 32
//   H_R2 = round_up(R1:32 * 3, B:128) = round_up(96, 128) = 128
//   H_R3 = = round_up(R2:128 * 2, A:256) = round_up(256, 256) = 256
//   PHY_ALIGNMENT = [H:256, W:32]
//   As you can see, PHY_LEGACY and PHY_ALIGNMENT are the same.
//
// Warning:
//   Alignment can encode strides of a higher granularity than LegacyPaddedShape.
//   It means Alignment can represnet whatever value of LegacyPaddedShape can, but not the other way around.
//   Most of the codebase today uses LegacyPaddedShape and get_padded_shape()
//   As we migrate to Alignment we can't use get_padded_shape anymore, because it can't represent aligned shapes.
//   So get_padded_shape() must be replaced with get_physical_size() and get_strides()
//
// Note:
//   This class is a work in progress. Many of its public methods have to be moved to private or even pImpl.
class TensorLayout {
public:
    TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const Alignment& alignment = {});

    // This method is not a constructor to make it easy to find and remove all of its usages in the codebase.
    [[deprecated("Use of LegacyPaddedShape is deprecated. Please use constructor with Alignment instead.")]]
    static TensorLayout fromLegacyPaddedShape(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape);

    Layout get_layout() const { return mPageConfig.is_row_major() ? Layout::ROW_MAJOR : Layout::TILE; }
    PageConfig get_page_config() const { return mPageConfig; }
    DataType get_data_type() const { return mDataType; }
    const MemoryConfig& get_memory_config() const { return mMemoryConfig; }
    const Alignment& get_alignment() const { return mAlignment; }

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
    void initialize_alignment();
    void validate_alignment() const;

    uint32_t get_header_size_bytes() const;
    uint32_t get_page_elements_count(const ttnn::SimpleShape& shape) const;

    Size get_page_shape(const Size& physical_size) const;
    size_t get_page_size_bytes(const Size& page_size) const;

    DataType mDataType = DataType::BFLOAT16;
    PageConfig mPageConfig;
    MemoryConfig mMemoryConfig;
    Alignment mAlignment;
};

} // tt::tt_metal
