// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Per-invocation addresses shared by every worker of a ring reader/writer kernel.
// Direction selects one of the two caller-owned out-ready semaphores.
namespace ring_common_arg {
enum : uint32_t { Input, Intermediate, Output, OutReady0, OutReady1, BatchReady, Barrier, Penult, Count };
}  // namespace ring_common_arg
