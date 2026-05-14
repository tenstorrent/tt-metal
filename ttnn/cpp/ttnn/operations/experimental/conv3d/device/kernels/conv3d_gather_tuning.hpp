// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// Shared between conv3d_program_factory (host) and reader_vol2col (kernel) so the
// trid-ring cutoffs stay in one place.
//
// Two cooperating gates select whether gather_rows_to_shard runs the trid ring:
//
//   1. Host per-shape classifier (program_factory): if the shape is compute-bound,
//      `gather_trids` is set to 0 and all ring code is constexpr-elided in the
//      kernel.  See conv3d_program_factory.cpp.
//
//   2. Kernel per-call fast-path (reader_vol2col): even when the host enables the
//      ring, an individual gather call with too few reads to fill at least two ring
//      cycles takes the issue-all + single-trailing-barrier path.  Threshold is
//      `2 * selected_trid_depth` reads.  The host cutoff below is intentionally
//      looser: it selects whether the shape should compile ring support at all;
//      each concrete gather call can still fall back locally when it is too small.
namespace conv3d_gather_tuning {

// Trid ring depths.  Two values: kGatherTridDepthHigh for shapes with large
// per-call gather bursts (full-rate pipelining), kGatherTridDepthLow for shapes
// with smaller bursts (shallower ring still hides per-read latency without the
// drain overhead of the deeper ring).  Selection happens at host time based on
// the per-shape inner-gather burst size.
constexpr uint32_t kGatherTridDepthHigh = 8;
constexpr uint32_t kGatherTridDepthLow = 4;

// Host gate: minimum bytes per matmul tile op (T_shard * H_shard * W_shard *
// C_in_block_bytes / matmul_tiles).  Below this the kernel is compute-bound and
// the ring overhead exceeds the reader gain.
constexpr uint64_t kGatherIntensityCutoffBytes = 128;

// Host selector: minimum reads per representative inner gather (T_shard * W_shard)
// before compiling the shallow ring.  This is one ring depth, not the kernel's
// two-depth per-call guard, because the host is choosing codegen for the shape
// while the kernel makes the final call-by-call decision.
constexpr uint32_t kGatherInnerBurstCutoff = kGatherTridDepthLow;

}  // namespace conv3d_gather_tuning
