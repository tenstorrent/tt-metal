// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_layout.hpp"

namespace tt::tt_metal {

Size::Size(size_t height, size_t width) : mHeight(height), mWidth(width) {}

Size::Size(const std::pair<size_t, size_t>& size) : mHeight(size.first), mWidth(size.second) {}

Size::operator std::pair<size_t, size_t>() const {
    return {mHeight, mWidth};
}

size_t Size::height() const { return mHeight; }
size_t Size::width() const { return mWidth; }

bool Size::operator==(const Size& rhs) const {
    return mHeight == rhs.mHeight && mWidth == rhs.mWidth;
}

// does not have to be a member, but it is easier to find if it is
Size Size::aligned_to_tile(const Size& tile) {
    auto round_up = [](size_t value, size_t multiple) {
        // can be faster if multiple is power of 2
        // return (value + multiple - 1) & ~(multiple - 1);
        return ((value + multiple - 1) / multiple) * multiple;
    };

    size_t height = round_up(mHeight, tile.height());
    size_t width = round_up(mWidth, tile.width());
    return Size(height, width);
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}


TensorLayout::TensorLayout(DataType dataType, Layout layout, const Size& tileSize, const MemoryConfig& memoryConfig)
    : mLayout(layout),
      mDataType(dataType),
      mTileSize(tileSize),
      mMemoryConfig(memoryConfig) {
}

Size TensorLayout::get_physical_size(const ttnn::SimpleShape& shape) const {
    size_t height = 1;
    size_t width = shape[-1];
    for (size_t i = 0; i < shape.rank() - 1; i++) {
        height *= shape[i];
    }
    Size size{height, width};
    if (mLayout == Layout::TILE) {
        size = size.aligned_to_tile(mTileSize);
    }

    return size;
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

} // namespace tt::tt_metal
