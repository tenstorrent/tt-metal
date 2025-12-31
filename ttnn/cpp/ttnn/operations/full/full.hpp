// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/full_device_operation.hpp"

namespace ttnn {
// Note: exposed as moreh_full for Python compatibility, but prim::full for C++
using prim::full;
// Alias for backward compatibility
inline auto moreh_full = prim::full;
}  // namespace ttnn
