// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "alignment.hpp"

#include <boost/container/vector.hpp>
#include <stddef.h>
#include <ostream>

#include <tt-metalium/shape_base.hpp>
#include <tt-metalium/small_vector.hpp>

namespace tt::tt_metal {

bool Alignment::operator==(const Alignment& other) const = default;

bool Alignment::operator==(const tt::stl::SmallVector<uint32_t>& other) const { return this->value_ == other; }

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Alignment& alignment) {
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

}  // namespace tt::tt_metal
