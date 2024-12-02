// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <tuple>
#include <ostream>

namespace tt::tt_metal {

class Size final {
public:
    Size(std::size_t height, std::size_t width);
    Size(const std::pair<std::size_t, std::size_t>& size);
    Size(const std::array<std::size_t, 2>& size);
    Size(const std::array<std::uint32_t, 2>& size);

    operator std::pair<std::size_t, std::size_t>() const;
    operator std::array<std::size_t, 2>() const;
    operator std::array<std::uint32_t, 2>() const;

    Size operator*(std::size_t scalar) const;

    bool operator==(const Size& rhs) const;

    [[nodiscard]] std::size_t height() const;
    [[nodiscard]] std::size_t width() const;

    static constexpr auto attribute_names = std::forward_as_tuple("height", "width");
    auto attribute_values() const { return std::forward_as_tuple(height_, width_); }

private:
    std::size_t height_ = 0;
    std::size_t width_ = 0;
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Size& size);

}  // namespace tt::tt_metal
