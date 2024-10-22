// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "size.hpp"

namespace tt::tt_metal {

Size::Size(size_t height, size_t width)
    : m_height(height), m_width(width) {}

Size::Size(const std::pair<size_t, size_t>& size)
    : Size(size.first, size.second) {}

Size::Size(const std::array<size_t, 2>& size)
    : Size(size[0], size[1]) {}

Size Size::operator/(const Size& rhs) const {
    return Size(m_height / rhs.m_height, m_width / rhs.m_width);
}

Size Size::operator*(size_t scalar) const {
    return Size(m_height * scalar, m_width * scalar);
}

Size Size::operator%(const Size& rhs) const {
    return Size(m_height % rhs.m_height,  m_width % rhs.m_width);
}

Size::operator std::pair<size_t, size_t>() const {
    return {m_height, m_width};
}

Size::operator std::array<size_t, 2>() const {
    return {m_height, m_width};
}

Size::operator std::array<uint32_t, 2>() const {
    return {static_cast<uint32_t>(m_height), static_cast<uint32_t>(m_width)};
}

size_t Size::height() const {
    return m_height;
}

size_t Size::width() const {
    return m_width;
}

bool Size::operator==(const Size& rhs) const {
    return m_height == rhs.m_height && m_width == rhs.m_width;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size)
{
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

} // namespace tt::tt_metal
