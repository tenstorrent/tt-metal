// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shape.hpp"

namespace ttnn {

bool SimpleShape::operator==(const SimpleShape &other) const {
    return this->mValue == other.mValue;
}

bool SimpleShape::operator==(const std::vector<uint32_t> &other) const {
    return this->mValue == other;
}

size_t SimpleShape::rank() const {
    return this->mValue.size();
}

uint64_t SimpleShape::volume() const {
    return std::accumulate(cbegin(), cend(),
                           uint64_t{1}, std::multiplies<uint64_t>());
}

std::ostream &operator<<(std::ostream &os, const ttnn::SimpleShape &shape) {
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

} // namespace ttnn
