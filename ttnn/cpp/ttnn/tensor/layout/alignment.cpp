// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "alignment.hpp"

#include <ostream>

namespace ttnn {

bool Alignment::operator==(const Alignment &other) const = default;

bool Alignment::operator==(const SmallVector<uint32_t> &other) const {
    return this->m_value == other;
}

std::ostream &operator<<(std::ostream &os, const Alignment &alignment) {
    os << "Alignment([";
    for (size_t i = 0; i < alignment.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << alignment[i];
    }
    os << "])";
    return os;
}

} // namespace ttnn
