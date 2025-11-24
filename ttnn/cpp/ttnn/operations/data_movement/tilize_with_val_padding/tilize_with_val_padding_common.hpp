// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <cstdint>

namespace ttnn {
using PadValue = std::variant<uint32_t, float>;
}
