// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

namespace tt::tt_metal {

class Size final {
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

    [[nodiscard]] size_t height() const;
    [[nodiscard]] size_t width() const;

    static constexpr auto attribute_names = std::forward_as_tuple("height", "width");
    auto attribute_values() const { return std::forward_as_tuple(m_height, m_width); }

private:
    size_t m_height = 0;
    size_t m_width = 0;
};

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Size &size);

} // namespace tt::tt_metal
