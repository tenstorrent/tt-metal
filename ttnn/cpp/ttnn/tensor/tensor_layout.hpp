// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/tile/tile.hpp"
#include "types.hpp"
#include "enum_types.hpp"

#include <cstddef>
#include <ostream>
#include <variant>

namespace tt::tt_metal {

class Size {
public:
    Size(size_t height, size_t width);
    Size(const std::pair<size_t, size_t>& size);
    Size(const std::array<size_t, 2>& size);

    operator std::pair<size_t, size_t>() const;
    operator std::array<size_t, 2>() const;
    operator std::array<uint32_t, 2>() const;

    Size operator/(const Size& rhs) const;
    Size operator*(size_t scalar) const;
    Size operator%(const Size& rhs) const;

    // comparison operator
    bool operator==(const Size& rhs) const;

    size_t height() const;
    size_t width() const;

    static constexpr auto attribute_names = std::forward_as_tuple("height", "width");
    auto attribute_values() const { return std::forward_as_tuple(mHeight, mWidth); }

private:
    size_t mHeight = 0;
    size_t mWidth = 0;
};

// Creating as a class to differentiate the use between LegacyPadding represented as SimpleShape and Alignment.
// This class has to eventually become its own class
class Alignment : public ttnn::SimpleShape {
public:
    using ttnn::SimpleShape::SimpleShape;
    size_t size() const { return rank(); }
};

using Strides = std::vector<size_t>;

struct RowMajorPageConfig {
    Alignment createDefaultAlignment(DataType dataType) const;
    void validateAlignment(const Alignment& alignment, DataType dataType) const;
    Size get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const;
};
struct TilePageConfig {
    Tile tile;

    TilePageConfig(const Tile& tile = Tile());

    Alignment createDefaultAlignment(DataType dataType) const;
    void validateAlignment(const Alignment& alignment, DataType dataType) const;
    Size get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const;
};

class PageConfig {
public:
    using Config = std::variant<RowMajorPageConfig, TilePageConfig>;

    PageConfig(const Config& config);
    PageConfig(Layout layout);
    PageConfig(Layout layout, const std::optional<Tile>& tile);

    Alignment createDefaultAlignment(DataType dataType) const;
    void validateAlignment(const Alignment& alignment, DataType dataType) const;
    Size get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const;
    bool isRowMajor() const;

private:
    Config mConfig;
};

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

    Layout get_layout() const { return mPageConfig.isRowMajor() ? Layout::ROW_MAJOR : Layout::TILE; }
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
    // Private constructor to create TensorLayout from LegacyPaddedShape
    TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape);

    // For the case when Aligmnet is not provided or is empty
    // This method will initialize Alignment to reflect requirements of Layout/DType/Sharding(currently not supported)
    void initializeAlignment();
    void validateAlignment() const;

    uint32_t get_header_size_bytes() const;
    uint32_t get_page_elements_count(const ttnn::SimpleShape& shape) const;

    // Returns number of elements in a page
    // For SINGLE_BANK layout returns physical size
    // For ROW_MAJOR width is equal to physical width
    Size get_page_shape(const Size& physical_size) const;
    size_t get_page_size_bytes(const Size& page_size) const;

    DataType mDataType = DataType::BFLOAT16;
    PageConfig mPageConfig;
    MemoryConfig mMemoryConfig;
    Alignment mAlignment;
};

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Alignment &value);
std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size);

} // tt::tt_metal
