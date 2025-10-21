// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>
#include <cstdint>

using PadValue = std::variant<uint32_t, float>;
