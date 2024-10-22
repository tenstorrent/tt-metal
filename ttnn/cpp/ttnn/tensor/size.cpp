// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "size.hpp"

namespace tt::tt_metal {

Size::Size(size_t height, size_t width)
    : mHeight(height), mWidth(width) {}

Size::Size(const std::pair<size_t, size_t>& size)
    : Size(size.first, size.second) {}

Size::Size(const std::array<size_t, 2>& size)
    : Size(size[0], size[1]) {}

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

size_t Size::height() const {
    return mHeight;
}

size_t Size::width() const {
    return mWidth;
}

bool Size::operator==(const Size& rhs) const {
    return mHeight == rhs.mHeight && mWidth == rhs.mWidth;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

} // namespace tt::tt_metal
