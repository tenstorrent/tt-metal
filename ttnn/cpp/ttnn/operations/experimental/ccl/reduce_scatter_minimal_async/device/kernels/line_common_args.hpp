// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Fresh caller-owned addresses, uniform across a line reader/writer kernel's workers.
namespace line_common_arg {
enum : uint32_t { Input, Intermediate, Output, Ready, Barrier, Count };
}  // namespace line_common_arg
