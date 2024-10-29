// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/shape/shape_base.hpp"
#include "ttnn/tensor/shape/small_vector.hpp"

namespace ttnn {

class Alignment final : protected ShapeBase {
public:
    using ShapeBase::ShapeBase;
    using ShapeBase::operator[];
    using ShapeBase::cbegin;
    using ShapeBase::cend;
    using ShapeBase::view;
    using ShapeBase::size;

    template<std::size_t N>
    bool operator==(const std::array<uint32_t, N> &other) const {
        const bool sameSize = m_value.size() == N;
        return sameSize && std::equal(m_value.begin(), m_value.end(), other.begin());
    }

    bool operator==(const Alignment &other) const;
    bool operator==(const SmallVector<uint32_t> &other) const;

    // Needed for reflect / fmt
    static constexpr auto attribute_names = std::forward_as_tuple("value");
    auto attribute_values() const { return std::forward_as_tuple(this->m_value); }

    friend std::ostream &operator<<(std::ostream &os, const Alignment &shape);
};

std::ostream &operator<<(std::ostream &os, const ttnn::Alignment &shape);

} // namespace ttnn
