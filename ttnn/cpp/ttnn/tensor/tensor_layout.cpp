// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"
#include <variant>
#include <vector>
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "types.hpp"

namespace tt::tt_metal {

namespace {
size_t round_up(size_t value, size_t multiple) {
    // can be faster if multiple is power of 2
    // return (value + multiple - 1) & ~(multiple - 1);
    return ((value + multiple - 1) / multiple) * multiple;
};

Alignment legacyPaddedShapeToAlignment(const ttnn::SimpleShape& legacyPaddedShape) {
    const auto rank = legacyPaddedShape.rank();
    std::vector<uint32_t> values(rank);

    if(rank >= 1) {
        values[rank - 1] = legacyPaddedShape[rank - 1];
    }
    if(rank >= 2) {
        values[rank - 2] = legacyPaddedShape[rank - 2];
    }
    for (int i = rank - 3; i >= 0; i--) {
        values[i] = legacyPaddedShape[i] * values[i + 1];
    }

    Alignment result(values);
    return result;
}
}

namespace utils {
size_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            TT_THROW("element_size_bytes() should not be used for BFLOAT8_B and BFLOAT4_B types becaues of how they are packed");

        default:
            TT_THROW("Unsupported data type!");
    }
}
}

Size::Size(size_t height, size_t width) : mHeight(height), mWidth(width) {}
Size::Size(const std::pair<size_t, size_t>& size) : mHeight(size.first), mWidth(size.second) {}
Size::Size(const std::array<size_t, 2>& size) : mHeight(size[0]), mWidth(size[1]) {}


Size Size::operator/(const Size& rhs) const {
    return Size(mHeight / rhs.mHeight, mWidth / rhs.mWidth);
}

Size Size::operator*(size_t scalar) const {
    return Size(mHeight * scalar, mWidth * scalar);
}

Size Size::operator%(const Size& rhs) const {
    return Size(mHeight % rhs.mHeight,  mWidth % rhs.mWidth);
}

Size::operator std::pair<size_t, size_t>() const {
    return {mHeight, mWidth};
}

Size::operator std::array<size_t, 2>() const {
    return {mHeight, mWidth};
}

Size::operator std::array<uint32_t, 2>() const {
    return {static_cast<uint32_t>(mHeight), static_cast<uint32_t>(mWidth)};
}

size_t Size::height() const { return mHeight; }
size_t Size::width() const { return mWidth; }

bool Size::operator==(const Size& rhs) const {
    return mHeight == rhs.mHeight && mWidth == rhs.mWidth;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Alignment &value) {
    os << "Alignment([";
    for (size_t i = 0; i < value.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << value[i];
    }
    os << "])";
    return os;
}


Alignment RowMajorPageConfig::createDefaultAlignment(DataType dataType) const {
{
    TT_FATAL(dataType != DataType::BFLOAT4_B && dataType != DataType::BFLOAT8_B, "BFLOAT4_B and BFLOAT8_B data types are not supported for ROW_MAJOR layout");
    return Alignment({sizeof(uint32_t) / utils::element_size_bytes(dataType)});}
}

Alignment TilePageConfig::createDefaultAlignment(DataType dataType) const {
    return Alignment({tile.get_height(), tile.get_width()});
}

void RowMajorPageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    TT_FATAL(alignment.size() > 0, "Alignment should have at least 1 dimension for Row Major layout");
    uint32_t widthAlignment = alignment[-1];
    uint32_t element_size = utils::element_size_bytes(dataType);
    uint32_t page_alignment = sizeof(uint32_t) / element_size;
    TT_FATAL((widthAlignment % page_alignment) == 0,
        "Wrong custom Tensor Layout alignment {}. For Row Major layout with element size {}bytes the innermost dimension must align to {}. This is because Buffer data is packed as uint32_t (4 bytes).",
        alignment, element_size, page_alignment);
}

void TilePageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    TT_FATAL(alignment.size() >= 2, "Alignment should have at least 2 dimensions for Tile layout");
    const auto widthAlignment = alignment[-1];
    TT_FATAL(widthAlignment % tile.get_width() == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width {}.", alignment, config.tile.get_width());
    auto heightAlignment = alignment[-2];
    TT_FATAL((heightAlignment % tile.get_height()) == 0,
        "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of tile height {}.", alignment, config.tile.get_height());
}

Size RowMajorPageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    if (memoryConfig.shard_spec.has_value()) {
        const auto& shard_spec = memoryConfig.shard_spec.value();
        const auto& shard_shape = shard_spec.shape;
        return Size(1, shard_shape[1]);
    }
    return Size(1, physical_size.width());
}

Size TilePageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    return Size(tile.get_height(), tile.get_width());
}


PageConfig::PageConfig(const Config& config)
    : mConfig(config) {
}

PageConfig::PageConfig(Layout layout) {
    if(layout == Layout::ROW_MAJOR) {
        mConfig = RowMajorPageConfig();
    }
    else {
        mConfig =  TilePageConfig();
    }
}

Alignment PageConfig::createDefaultAlignment(DataType dataType) const {
    return std::visit([&](const auto& config) constexpr { return config.createDefaultAlignment(dataType); }, mConfig);
}

void PageConfig::validateAlignment(const Alignment& alignment, DataType dataType) const {
    std::visit([&](const auto& config) constexpr { config.validateAlignment(alignment, dataType); }, mConfig);
}

Size PageConfig::get_page_shape(const Size& physical_size, const MemoryConfig& memoryConfig) const {
    return std::visit([&](const auto& config) constexpr { return config.get_page_shape(physical_size, memoryConfig); }, mConfig);
}

bool PageConfig::isRowMajor() const {
    return std::holds_alternative<RowMajorPageConfig>(mConfig);
}


TensorLayout::TensorLayout(DataType dataType, const PageConfig& pageConfig, const MemoryConfig& memoryConfig, const Alignment& alignment)
    : mDataType(dataType),
      mPageConfig(pageConfig),
      mMemoryConfig(memoryConfig),
      mAlignment(alignment) {

    initializeAlignment();
    validateAlignment();
}

// Private constructor to create TensorLayout from LegacyPaddedShape
TensorLayout::TensorLayout(DataType dataType, Layout layout, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape)
    : TensorLayout(dataType, PageConfig(layout), memoryConfig, legacyPaddedShapeToAlignment(legacyPaddedShape)) {
    mLegacyPaddedShape = legacyPaddedShape;
}

TensorLayout TensorLayout::fromLegacyPaddedShape(DataType dataType, Layout layout, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape) {
    return TensorLayout(dataType, layout, memoryConfig, legacyPaddedShape);
}

void TensorLayout::initializeAlignment() {
    if(mAlignment.size() != 0)
        return;

    mAlignment = mPageConfig.createDefaultAlignment(mDataType);
}

void TensorLayout::validateAlignment() const
{
    return mPageConfig.validateAlignment(mAlignment, mDataType);
}

std::optional<ShardSpecBuffer> TensorLayout::get_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!mMemoryConfig.is_sharded())
        return std::nullopt;

    TT_FATAL(mMemoryConfig.shard_spec.has_value(), "MemoryConfig should have Shard Spec specified for sharded memory layout");

    auto& shard_spec = mMemoryConfig.shard_spec.value();
    Size physical_shape = get_physical_shape(shape);
    Size page_shape = get_page_shape(physical_shape);
    Size tensor2d_size = physical_shape / page_shape; // looks like shard grid
    ShardSpecBuffer shard_spec_buffer(shard_spec, std::array<uint32_t, 2>(page_shape), std::array<uint32_t, 2>(tensor2d_size));

    return shard_spec_buffer;
}

size_t TensorLayout::get_packed_buffer_size_bytes(const ttnn::SimpleShape& shape) const {
    const Size physical_size = get_physical_shape(shape);
    const Size page_shape = get_page_shape(physical_size);
    const Size size_modulo = physical_size % page_shape;
    TT_FATAL(size_modulo.height() == 0 && size_modulo.width() == 0, "Physical size {} should be multiple of page size {}", physical_size, page_size);

    const size_t physical_area = physical_size.height() * physical_size.width();
    const size_t page_area = page_shape.height() * page_shape.width();

    size_t page_count = physical_area / page_area;
    size_t page_size_bytes = get_page_size_bytes(page_shape);

    return page_count * page_size_bytes;
}

size_t TensorLayout::get_page_size_bytes(const ttnn::SimpleShape& shape) const {
    auto physical_size = get_physical_shape(shape);
    auto page_shape = get_page_shape(physical_size);
    return get_page_size_bytes(page_shape);
}

size_t TensorLayout::get_page_size_bytes(const Size& page_size) const {
    const size_t elements_in_page = page_size.height() * page_size.width();
    uint32_t page_size_bytes = get_header_size_bytes();

    switch (mDataType) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT16:
        case DataType::UINT8:
            page_size_bytes += elements_in_page * utils::element_size_bytes(mDataType);
            break;

        case DataType::BFLOAT8_B:
            page_size_bytes += elements_in_page;
            break;

        case DataType::BFLOAT4_B:
            TT_FATAL(elements_in_page % 2 == 0, "BFLOAT4_B should have even number of elements in a page");
            page_size_bytes += elements_in_page / 2;
            break;

        default:
            TT_THROW("Unsupported data type!");
    }

    TT_FATAL(page_size_bytes != 0, "Page size should not be zero");

    return page_size_bytes;
}

Size TensorLayout::get_physical_shape(const ttnn::SimpleShape& shape) const {
    TT_FATAL(shape.rank() > 2, "Shape should have at least 2 dimensions");
    TT_FATAL(mAlignment.size() <= shape.rank(), "Alignment rank should be less than or equal to the rank of the shape");

    const int rank = static_cast<int>(shape.rank());
    size_t width = round_up(shape[-1], mAlignment[-1]);
    size_t height = 1;
    for (int i = -2; i >= -rank; --i) {
        height *= shape[i];
        if (mAlignment.size() >= static_cast<size_t>(-i)) {
            height = round_up(height, mAlignment[i]);
        }
    }

    if(mMemoryConfig.shard_spec.has_value())
    {
        auto& shard_spec = mMemoryConfig.shard_spec.value();
        height = round_up(height, shard_spec.shape[0]);
        width = round_up(width, shard_spec.shape[1]);
    }

    Size size{height, width};

    return size;
}

Size TensorLayout::get_page_shape(const Size& physical_size) const {
    if(mMemoryConfig.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        return physical_size;
    }

    return mPageConfig.get_page_shape(physical_size, mMemoryConfig);
}

uint32_t TensorLayout::get_header_size_bytes() const {
    switch (mDataType) {
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
            return 64;
        default:
            return 0;
    }
}

Strides TensorLayout::get_strides(const ttnn::SimpleShape& shape) const {
    const int rank = static_cast<int>(shape.rank());
    const int alignmentRank = static_cast<int>(mAlignment.size());

    Strides strides(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];

        const int alignment_index = i - (rank - alignmentRank) + 1;
        if(alignment_index >= 0) {
            strides[i] = round_up(strides[i], mAlignment[alignment_index]);
        }
    }

    return strides;
}

ttnn::SimpleShape TensorLayout::get_padded_shape(const ttnn::SimpleShape& shape) const
{
    TT_FATAL(mLegacyPaddedShape.has_value(), "Use get_physical_size() or get_strides(). Calling get_padded_shape() is not allowed for TensorLayout created w/o LegacyPaddedShape. ");
    return mLegacyPaddedShape.value();
}

} // namespace tt::tt_metal
