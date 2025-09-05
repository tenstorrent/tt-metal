// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>

namespace tt::tt_metal {

enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Layout& layout);

}  // namespace tt::tt_metal
