// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Runtime-argument layout of the SDPA recipe reader and writer, shared by the host (device/recipe/) and the kernels.
// Every buffer address is a common runtime argument at a fixed slot (0 when the tensor is absent), so a program-cache
// hit patches one word per buffer. Per-core arguments carry only the core's work split.
namespace sdpa_recipe_args {

// Reader common arguments.
constexpr uint32_t kReaderQ = 0, kReaderK = 1, kReaderV = 2;
constexpr uint32_t kReaderMask = 3;
constexpr uint32_t kReaderSink = 4;
constexpr uint32_t kReaderJointQ = 5;  // joint Q, K, V: 5, 6, 7
constexpr uint32_t kReaderQOffset = 8;    // key ranges: device Q offset (chunk_start_idx / windowed Q offset tensor)
constexpr uint32_t kReaderSegments = 9;   // key ranges: cu_window_seqlens
constexpr uint32_t kReaderPageTable = 10;  // key ranges: page table
constexpr uint32_t kReaderCommonArgs = 11;

// Reader per-core arguments. Dense / joint: the core's Q chunks [first_job, first_job + jobs), its place in the
// K/V chain of its head and the next core's job count. Key ranges: the core's snake deal (z_first, z_count), K/V
// sharing partners, the scalar Q offset and, ring-distributed, the two Q slabs' first chunks.
constexpr uint32_t kReaderFirstJob = 0, kReaderJobs = 1, kReaderRank = 2, kReaderChainLength = 3;
constexpr uint32_t kReaderPrevX = 4, kReaderPrevY = 5, kReaderNextX = 6, kReaderNextY = 7, kReaderNextJobs = 8;
constexpr uint32_t kReaderQOffsetScalar = 9;  // key ranges
constexpr uint32_t kReaderSlab = 10;          // key ranges with Q slabs: 10, 11

// Writer common arguments.
constexpr uint32_t kWriterOut = 0, kWriterJointOut = 1, kWriterQOffset = 2, kWriterSegments = 3;
constexpr uint32_t kWriterCommonArgs = 4;

// Writer per-core arguments.
constexpr uint32_t kWriterFirstJob = 0, kWriterJobs = 1;
constexpr uint32_t kWriterQOffsetScalar = 2;  // key ranges
constexpr uint32_t kWriterSlab = 3;           // key ranges with Q slabs: 3, 4

}  // namespace sdpa_recipe_args
