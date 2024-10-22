// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vector_base.hpp"

namespace ttnn {

class SimpleShape final : protected tt::tt_metal::vector_base {
public:
    using tt::tt_metal::vector_base::vector_base;
    using tt::tt_metal::vector_base::operator[];
    using tt::tt_metal::vector_base::cbegin;
    using tt::tt_metal::vector_base::cend;
    using tt::tt_metal::vector_base::as_vector;

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        const bool sameSize = mValue.size() == N;
        return sameSize && std::equal(mValue.begin(), mValue.end(), other.begin());
    }

    bool operator==(const SimpleShape &other) const;
    bool operator==(const std::vector<uint32_t> &other) const;

    [[nodiscard]] size_t rank() const;
    [[nodiscard]] uint64_t volume() const;


    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->mValue); }

    friend std::ostream &operator<<(std::ostream &os, const SimpleShape &shape);
};

std::ostream &operator<<(std::ostream &os, const ttnn::SimpleShape &shape);

} // namespace ttnn
