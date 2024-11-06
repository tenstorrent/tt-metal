// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "enum_types.hpp"

#include <magic_enum.hpp>

namespace tt::tt_metal {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Layout& value) {
    os << magic_enum::enum_name(value);
    return os;
}

} // namespace tt::tt_metal
