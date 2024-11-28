// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "size.hpp"
#include <ostream>

namespace tt::tt_metal {

Size::Size(size_t height, size_t width) : height_(height), width_(width) {}

Size::Size(const std::pair<size_t, size_t>& size) : Size(size.first, size.second) {}

Size::Size(const std::array<size_t, 2>& size) : Size(size[0], size[1]) {}

Size::Size(const std::array<uint32_t, 2>& size) : Size(size[0], size[1]) {}

Size Size::operator*(size_t scalar) const { return Size(height_ * scalar, width_ * scalar); }

Size::operator std::pair<size_t, size_t>() const { return {height_, width_}; }

Size::operator std::array<size_t, 2>() const { return {height_, width_}; }

Size::operator std::array<uint32_t, 2>() const {
    return {static_cast<uint32_t>(height_), static_cast<uint32_t>(width_)};
}

size_t Size::height() const { return height_; }

size_t Size::width() const { return width_; }

bool Size::operator==(const Size& rhs) const { return height_ == rhs.height_ && width_ == rhs.width_; }

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size) {
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

}  // namespace tt::tt_metal
