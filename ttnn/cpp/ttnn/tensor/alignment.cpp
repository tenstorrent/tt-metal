// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "alignment.hpp"

namespace tt::tt_metal {

bool Alignment::operator==(const Alignment &other) const {
    return this->mValue == other.mValue;
}

bool Alignment::operator==(const std::vector<uint32_t> &other) const {
    return this->mValue == other;
}

size_t Alignment::size() const {
    return this->mValue.size();
}

std::ostream &operator<<(std::ostream &os, const tt::tt_metal::Alignment &alignment) {
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

} // namespace tt::tt_metal
