// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

#include <numeric>
#include <ostream>
#include "ttnn/tensor/shape/small_vector.hpp"
#include "tt_metal/common/assert.hpp"

namespace tt::tt_metal {

bool SimpleShape::operator==(const SimpleShape& other) const = default;

bool SimpleShape::operator==(const SmallVector<uint32_t>& other) const { return this->value_ == other; }

size_t SimpleShape::rank() const { return this->size(); }

uint64_t SimpleShape::volume() const {
    return std::accumulate(cbegin(), cend(), uint64_t{1}, std::multiplies<uint64_t>());
}

const uint32_t SimpleShape::get_normalized_index(std::int64_t index) const {
    std::int64_t rank = static_cast<std::int64_t>(this->rank());
    std::uint64_t normalized_index = index >= 0 ? index : rank + index;
    TT_FATAL(
        normalized_index >= 0 and normalized_index < rank,
        "Index is out of bounds for the rank, should be between 0 and {} however is {}",
        rank - 1,
        normalized_index);
    return normalized_index;
}

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::SimpleShape& shape) {
    os << "SimpleShape([";
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << "])";
    return os;
}

}  // namespace tt::tt_metal
