// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "types.hpp"

namespace tt::tt_metal {

namespace {
size_t round_up(size_t value, size_t multiple) {
    // can be faster if multiple is power of 2
    // return (value + multiple - 1) & ~(multiple - 1);
    return ((value + multiple - 1) / multiple) * multiple;
};
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

// does not have to be a member, but it is easier to find if it is
Size Size::aligned_to_tile(const Size& tile) {
    size_t height = round_up(mHeight, tile.height());
    size_t width = round_up(mWidth, tile.width());
    return Size(height, width);
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

TensorLayout::TensorLayout(DataType dataType, const Size& tileSize, const MemoryConfig& memoryConfig, const Alignment& alignment)
    : mDataType(dataType),
      mTileSize(tileSize),
      mMemoryConfig(memoryConfig),
      mAlignment(alignment) {

    mLayout = mTileSize.height() == 1 ? Layout::ROW_MAJOR : Layout::TILE;
    initializeAlignment();
    validateCustomAlignment();
}

TensorLayout::TensorLayout(DataType dataType, Layout layout, const MemoryConfig& memoryConfig, const Alignment& alignment)
    : mLayout(layout),
      mDataType(dataType),
      mMemoryConfig(memoryConfig),
      mAlignment(alignment) {

    if (mLayout == Layout::TILE) {
        mTileSize = Size(32, 32);
    }
    else {
        mTileSize = Size(1, 1); // width 1 makes no sense? need juggestions
    }

    initializeAlignment();
    validateCustomAlignment();
}

namespace {

// This function is used to convert the legacy padded shape to Alignment
// This is a temporary solution until we get rid of the LegacyPaddedShape
// Keep in mind:
//    The benefit of Alignment compared to LegacyPaddedShape is that it allows
//    to pad between Channels and Batches with a row granularity.
// Examples:
// 1. Given LegacyPaddedShape (1, 1, 16, 20), alignment is (1, 1, 16, 20)
//    means Shape (1, 1, 16, 16) which in theory takes physical space (1 * 1 * 16, 16)
//    will take (1 * 1 * 16, 20)
//
// 2. Given LegacyPaddedShape (1, 1, 32, 32), alignment is (1, 1, 32, 32)
//    Shape (1, 1, 16, 16)
//      which in theory takes physical space (1 * 1 * 16, 16)
//      will take (1 * 1 * 32, 32)
//
// 3. Given LegacyPaddedShape (2, 3, 32, 32), alignment is (2, 3*32, 32, 32).
//    Shape (2, 2, 16, 16)
//      which in theory takes physical space (2 * 2 * 16, 16)
//      will take (2 * 3 * 32, 32)

Alignment legacyPaddedShapeToAlignment(const ttnn::SimpleShape& legacyPaddedShape) {
    std::vector<uint32_t> values;
    const auto rank = legacyPaddedShape.rank();
    values.resize(rank);

    if(rank > 1) {
        values[rank - 1] = legacyPaddedShape[rank - 1];
    }
    if(rank > 2) {
        values[rank - 2] = legacyPaddedShape[rank - 2];
    }
    for (size_t i = rank - 3; i >= 0; i--) {
        values[i] = legacyPaddedShape[i] * values[i + 1];
    }

    Alignment result(values);
    return result;
}
}

// Private constructor to create TensorLayout from LegacyPaddedShape
TensorLayout::TensorLayout(DataType dataType, Layout layout, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape)
    : TensorLayout(dataType, layout, memoryConfig, legacyPaddedShapeToAlignment(legacyPaddedShape)) {
    mLegacyPaddedShape = legacyPaddedShape;
}

TensorLayout TensorLayout::fromLegacyPaddeShape(DataType dataType, Layout layout, const MemoryConfig& memoryConfig, const ttnn::SimpleShape& legacyPaddedShape) {
    return TensorLayout(dataType, layout, memoryConfig, legacyPaddedShape);
}

void TensorLayout::initializeAlignment() {
    if(mAlignment.size() != 0)
        return;

    switch(mLayout) {
        case Layout::ROW_MAJOR: {
            mAlignment = Alignment({sizeof(uint32_t) % element_size_bytes()});
            break;
        }
        case Layout::TILE: {
            mAlignment = Alignment({mTileSize.height(), mTileSize.width()});
            break;
        }
        case Layout::INVALID:
            TT_THROW("Invalid Layout");
    }
}

void TensorLayout::validateCustomAlignment() const
{
    if(mAlignment.size() == 0)
        return;

    switch(mLayout) {
        case Layout::ROW_MAJOR: {
            if(mAlignment.size() > 0) {
                auto widthAlignment = mAlignment[-1];
                auto element_size = element_size_bytes();
                auto page_alignment = sizeof(uint32_t) % element_size;
                TT_FATAL(widthAlignment % page_alignment == 0,
                "Wrong custom Tensor Layout alignment {}. For Row Major layout with element size {}bytes the innermost dimension must align to {}. This is because Buffer data is packes as uint32_t (4 bytes).",
                    mAlignment,
                    element_size,
                    page_alignment);
            }
            break;
        }
        case Layout::TILE: {
            if(mAlignment.size() > 1) {
                auto widthAlignment = mAlignment[-1];
                TT_FATAL(widthAlignment % mTileSize.width() == 0,
                "Wrong custom Tensor Layout alignment {}. For Tile layout innermost dimension should be multiple of tile width {}.", mAlignment, mTileSize.width());
            }
            if(mAlignment.size() > 2) {
                auto heightAlignment = mAlignment[-2];
                TT_FATAL(heightAlignment % mTileSize.height() == 0,
                "Wrong custom Tensor Layout alignment {}. For Tile layout second innermost dimension should be multiple of tile height {}.", mAlignment, mTileSize.height());
            }
            break;
        }
        case Layout::INVALID:
            TT_THROW("Invalid Layout");
    }
}

Size TensorLayout::get_physical_size(const ttnn::SimpleShape& shape) const {
    TT_FATAL(shape.rank() > 2, "Shape should have at least 2 dimensions");

    auto padded_shape = get_padded_shape(shape);

    size_t width = padded_shape[-1];
    size_t height = padded_shape[-2];
    for (size_t i = 0; i < padded_shape.rank() - 2; i++) {
        height *= padded_shape[i];
        height = round_up(height, mAlignment[i]);
    }

    Size size{height, width};
    return size;
}

Size TensorLayout::get_sharded_page_size() const {
    TT_FATAL(mMemoryConfig.shard_spec.has_value(), "MemoryConfig should have Shard Spec");
    const auto& shard_spec = mMemoryConfig.shard_spec.value();
    const auto& shard_shape = shard_spec.shape;

    switch (mLayout) {
        case Layout::ROW_MAJOR:
            return Size(1, shard_shape[1]);
        case Layout::TILE: {
            return mTileSize;
        }
        default: TT_THROW("Unsupported layout to write to device");
    }
}

std::optional<ShardSpecBuffer> TensorLayout::get_shard_spec_buffer(const ttnn::SimpleShape& shape) const {
    if (!mMemoryConfig.is_sharded())
        return std::nullopt;

    TT_FATAL(mMemoryConfig.shard_spec.has_value(), "MemoryConfig should have Shard Spec specified for sharded memory layout");

    auto& shard_spec = mMemoryConfig.shard_spec.value();
    Size physical_size = get_physical_size(shape);
    Size page_shape = get_sharded_page_size();
    Size tensor2d_size = physical_size / page_shape;
    ShardSpecBuffer shard_spec_buffer(shard_spec, std::array<uint32_t, 2>(page_shape), std::array<uint32_t, 2>(tensor2d_size));

    return shard_spec_buffer;
}

size_t TensorLayout::get_packed_buffer_size(const ttnn::SimpleShape& shape) const {
    Size physical_size = get_physical_size(shape);
    return physical_size.height() * physical_size.width() * element_size_bytes();
}

uint32_t TensorLayout::get_page_elements_count(const ttnn::SimpleShape& shape) const {
    if(mMemoryConfig.memory_layout == TensorMemoryLayout::SINGLE_BANK) {
        auto physical_size = get_physical_size(shape);
        return physical_size.height() * physical_size.width();
    }

    uint32_t elements_in_page = 0;
    switch(mLayout) {
        case Layout::ROW_MAJOR:
            elements_in_page = shape[-1];
            break;
        case Layout::TILE:
            elements_in_page = mTileSize.height() * mTileSize.width();
            break;
        default:
            TT_THROW("Unsupported layout");
    }

    return elements_in_page;
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

uint32_t TensorLayout::element_size_bytes() const {
    switch (mDataType) {
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

size_t TensorLayout::get_page_size_bytes(const ttnn::SimpleShape& shape) const {
    uint32_t page_size_bytes = get_header_size_bytes();
    uint32_t elements_in_page = get_page_elements_count(shape);

    switch (mDataType) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT16:
        case DataType::UINT8:
            page_size_bytes += elements_in_page * element_size_bytes();
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

    //TT_FATAL(total_size_bytes % page_size_bytes == 0);
    TT_FATAL(page_size_bytes != 0, "Page size should not be zero");

    return page_size_bytes;
}

Alignment TensorLayout::get_strides(const ttnn::SimpleShape& shape) const {
    TT_FATAL(mAlignment.size() <= shape.rank(), "Alignment rank should be less than or equal to the rank of the shape");

    std::vector<uint32_t> strides;
    strides.resize(shape.rank());

    for (size_t i = shape.rank() - 1; i >= 0; i--) {
        if (i == shape.rank() - 1) {
            strides[i] = 1;
        } else {
            strides[i] = strides[i + 1] * shape[i + 1];
            if(mAlignment.size() > i) {
                strides[i] = round_up(strides[i], mAlignment[i]);
            }
        }
    }

    Alignment result(strides);
    return result;
}

ttnn::SimpleShape TensorLayout::get_padded_shape(const ttnn::SimpleShape& shape) const
{
    TT_FATAL(mLegacyPaddedShape.has_value(), "Use get_physical_size() or get_strides(). Calling get_padded_shape() is not allowed for TensorLayout created w/o LegacyPaddedShape. ");
    return mLegacyPaddedShape.value();
}

} // namespace tt::tt_metal
