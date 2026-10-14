// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Layout shared by the unary program factory and its DRAM read kernels.

#pragma once

#include <cstdint>

namespace dram_read {

// Runtime args of the TILE reader and writer. Both kernels get the same list; arg 0 is the buffer address.
constexpr uint32_t kArgNumPages = 1;     // pages of this core; with the work queue, pages of the whole tensor
constexpr uint32_t kArgFirstPage = 2;    // first page of this core; with the work queue, the worker index
constexpr uint32_t kArgPageOrder = 3;    // five args, see RotatedPages
constexpr uint32_t kArgChunkPages = 8;   // work queue only from here on
constexpr uint32_t kArgSchedulerXy = 9;  // NoC coordinates of the scheduler core, x | (y << 16)
constexpr uint32_t kArgNumWorkers = 10;
constexpr uint32_t kArgIsScheduler = 11;
constexpr uint32_t kNumArgs = 12;

// Work-queue circular buffers and semaphores.
constexpr uint32_t kCbComputeCount = 4;  // reader to compute: tile count of the next chunk, 0 when done
constexpr uint32_t kCbWriterChunk = 5;   // reader to writer: first position and page count of the next chunk
constexpr uint32_t kCbRequestTable = 7;  // on the scheduler core: one request word per worker
constexpr uint32_t kReplySemaphore = 0;  // the scheduler's latest reply to this worker
constexpr uint32_t kGoSemaphore = 1;     // set once the scheduler accepts requests
constexpr uint32_t kMaxWorkers = 144;    // the scheduler keeps one sequence number per worker on its stack

}  // namespace dram_read
