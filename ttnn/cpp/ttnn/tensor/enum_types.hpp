// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <cstdint>

namespace tt {

namespace tt_metal {

enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

} // namespace tt_metal

} // namespace tt
