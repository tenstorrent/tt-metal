// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"
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

TensorLayout::TensorLayout(DataType dataType, const Size& tileSize, const MemoryConfig& memoryConfig)
    : mDataType(dataType),
      mTileSize(tileSize),
      mMemoryConfig(memoryConfig) {

    mLayout = mTileSize.height() == 1 ? Layout::ROW_MAJOR : Layout::TILE;
}

TensorLayout::TensorLayout(DataType dataType, Layout layout, const MemoryConfig& memoryConfig)
    : mLayout(layout),
      mDataType(dataType),
      mMemoryConfig(memoryConfig) {

    if (mLayout == Layout::TILE) {
        mTileSize = Size(32, 32);
    }
    else {
        mTileSize = Size(1, 1); // width 1 makes no sense? need juggestions
    }
}

Size TensorLayout::get_physical_size(const ttnn::SimpleShape& shape) const {
    TT_FATAL(shape.rank() > 2, "Shape should have at least 2 dimensions");

    size_t width = shape[-1];
    size_t height = shape[-2];

    if (mLayout == Layout::TILE) {
        width = round_up(width, mTileSize.width());
        height = round_up(height, mTileSize.height());
    }

    for (size_t i = 0; i < shape.rank() - 2; i++) {
        height *= shape[i];
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

    uint32_t elements_in_page = shape[-1];
    if(mTileSize.height() > 1) { // not row major
        elements_in_page = mTileSize.height() * mTileSize.width();
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

Size TensorLayout::get_tile_alignment_padding(const ttnn::SimpleShape& shape) const {
    size_t height = 0;
    size_t width = 0;
    if (mLayout == Layout::TILE) {
        height = mTileSize.height() - shape[-2] % mTileSize.height();
        width = mTileSize.width() - shape[-1] % mTileSize.width();
    }
    return Size(height, width);
}

ttnn::SimpleShape TensorLayout::get_padded_shape(const ttnn::SimpleShape& shape) const
{
    if (mLayout == Layout::TILE) {
        auto padding = get_tile_alignment_padding(shape);
        auto values = shape.as_vector();
        values[shape.rank() - 1] += padding.width();
        values[shape.rank() - 1] += padding.height();
        ttnn::SimpleShape padded_shape(std::move(values));
        return padded_shape;
    }

    return shape;
}

} // namespace tt::tt_metal
