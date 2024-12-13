// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

#include <numeric>
#include <ostream>
#include "ttnn/tensor/shape/small_vector.hpp"

namespace tt::tt_metal {

bool SimpleShape::operator==(const SimpleShape& other) const = default;

bool SimpleShape::operator==(const SmallVector<uint32_t>& other) const { return this->value_ == other; }

size_t SimpleShape::rank() const { return this->size(); }

uint64_t SimpleShape::volume() const {
    return std::accumulate(cbegin(), cend(), uint64_t{1}, std::multiplies<uint64_t>());
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
