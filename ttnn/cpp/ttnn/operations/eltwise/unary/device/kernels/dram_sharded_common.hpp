// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sharded unary layout shared by the program factory and the kernels.

#pragma once

#include <cstdint>

namespace dram_shard {

// TILE reader/writer runtime args; arg 0 is the buffer.
constexpr uint32_t kArgSplit = 1;      // page count and start, or worker args and total pages
constexpr uint32_t kArgPageOrder = 3;  // shard_stride, num_shards, last_shard_pages, shard_width, row_pages
constexpr uint32_t kArgQueue = 8;      // chunk pages, then scheduler coordinates (reader) / worker count (writer)
constexpr uint32_t kNumArgs = 10;

// Work-queue CBs and semaphores.
constexpr uint32_t kCbComputeCount = 4;  // reader -> compute: tile count of the next chunk
constexpr uint32_t kCbWriterChunk = 5;   // reader -> writer: (first position, page count)
constexpr uint32_t kCbRequestTable = 7;  // one request word per worker, on the scheduler core
constexpr uint32_t kReplySemaphore = 0;
constexpr uint32_t kGoSemaphore = 1;
constexpr uint32_t kMaxWorkers = 144;  // the scheduler keeps one word per worker on its stack

}  // namespace dram_shard
