// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// With GN_RUNTIME_SIZE_ARGS defined, the welford GroupNorm kernels read the values that depend on the
// input's spatial size from runtime args instead of named compile-time args, so a new input size reuses
// the compiled program. The tile rows are split unevenly over the cores of a reduction (mcast) group:
// the first NUM_TALL_CORES cores of every group hold BASE_ROWS + 1 tile rows, the others BASE_ROWS.
// Each core streams its input one tile at a time, in one out block. The program factory writes the
// same common args to every kernel.
namespace gn_size_args {
constexpr std::uint32_t BASE_ROWS = 0;        // tile rows of a core past the first NUM_TALL_CORES
constexpr std::uint32_t NUM_TALL_CORES = 1;   // cores of each group, from its first, with one more
constexpr std::uint32_t NUM_MCAST_CORES = 2;  // cores per mcast (reduction) group
// When the spatial extent is not whole tiles, its last tile row holds padding rows. The readers zero
// them, so the statistics see zeros, and the sender corrects the statistics for those zeros.
constexpr std::uint32_t PAD_ROW_FIRST_PAGE = 3;  // first page of the tile row with padding rows
constexpr std::uint32_t PAD_ROW_VALID_ROWS = 4;  // valid rows in that tile row, 32 when there is none
constexpr std::uint32_t PAD_STATS_SCALE = 5;     // fp32 bits of padded / logical spatial extent
constexpr std::uint32_t NUM_ARGS = 6;
}  // namespace gn_size_args

// Per-core runtime args, after each kernel's own: the receiver reader's at 7, the writer's at 10 and
// the compute kernel's at 0. The sender is the first core of its group, so it derives its tile rows
// from the common args.
namespace gn_core_size_args {
constexpr std::uint32_t RECEIVER_BASE = 7;
constexpr std::uint32_t WRITER_BASE = 10;
constexpr std::uint32_t COMPUTE_BASE = 0;
constexpr std::uint32_t BLOCK_H = 0;              // this core's tile rows
constexpr std::uint32_t TWO_PASS_RECIPROCAL = 1;  // compute only: fp32 bits of the two-pass mean reciprocal
constexpr std::uint32_t NUM_COMPUTE_ARGS = 2;
}  // namespace gn_core_size_args
