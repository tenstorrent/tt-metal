// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Limits shared by the unary program factory and its DRAM read kernels.

#pragma once

#include <cstdint>

namespace dram_read {

// The work-queue scheduler keeps one word per worker on its stack.
constexpr uint32_t kMaxWorkers = 144;

}  // namespace dram_read
