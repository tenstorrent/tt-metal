// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/assert.hpp>

#include "ttnn/tensor/layout/layout.hpp"

namespace tt::tt_metal {
std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Layout& layout) {
    switch (layout) {
        case Layout::ROW_MAJOR: return os << "Layout::ROW_MAJOR";
        case Layout::TILE: return os << "Layout::TILE";
        case Layout::INVALID: return os << "Layout::INVALID";
    }
    TT_THROW("Unreachable");
}

}  // namespace tt::tt_metal
