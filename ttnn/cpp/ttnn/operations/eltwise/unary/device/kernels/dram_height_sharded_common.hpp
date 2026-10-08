// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Resources of the DRAM height-sharded work queue, shared by the program factory and the kernels.

#pragma once

#include <cstdint>

namespace dram_hs {

constexpr uint32_t kCbComputeCount = 4;  // reader -> compute: tile count of the next chunk
constexpr uint32_t kCbWriterChunk = 5;   // reader -> writer: (first position, page count)
constexpr uint32_t kCbRequestTable = 7;  // one request word per worker, read on the scheduler core
constexpr uint32_t kReplySemaphore = 0;
constexpr uint32_t kGoSemaphore = 1;
constexpr uint32_t kMaxWorkers = 144;  // the scheduler keeps one word per worker on the writer's stack

}  // namespace dram_hs
