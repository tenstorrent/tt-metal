// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>

namespace tt::tt_metal {

enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

inline std::ostream& operator<<(std::ostream& os, Layout layout) {
    switch (layout) {
        case Layout::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case Layout::TILE: os << "TILE"; break;
        case Layout::INVALID: os << "INVALID"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

} // namespace tt::tt_metal
