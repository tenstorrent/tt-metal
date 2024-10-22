// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vector_base.hpp"

namespace tt::tt_metal {

class Alignment final : protected tt::tt_metal::VectorBase {
public:
    using tt::tt_metal::VectorBase::VectorBase;
    using tt::tt_metal::VectorBase::operator[];
    using tt::tt_metal::VectorBase::cbegin;
    using tt::tt_metal::VectorBase::cend;
    using tt::tt_metal::VectorBase::as_vector;

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        const bool sameSize = m_value.size() == N;
        return sameSize && std::equal(m_value.begin(), m_value.end(), other.begin());
    }

    bool operator==(const Alignment &other) const;
    bool operator==(const std::vector<uint32_t> &other) const;

    [[nodiscard]] size_t size() const;


    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->m_value); }

    friend std::ostream &operator<<(std::ostream &os, const Alignment &shape);
};

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Alignment &shape);

} // namespace tt::tt_metal
