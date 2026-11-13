// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm — CB indices shared by every kernel. Must match the CB_* constants in
// all_gather_rms_norm_program_descriptor.py.
#pragma once

#include <cstdint>

// Per-stage device zones (perf breakdown probes only): compiled in only when the host sets AGRMS_PROFILE_ZONES
// (env AGRMS_PROFILE_ZONES=1). Left on, they fill the profiler DRAM buffers under the eval perf harness (per-block
// markers on every member core, 500 calls) and add marker cost to the headline device time.
#ifdef AGRMS_PROFILE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define AGRMS_ZONE(name) DeviceZoneScopedN(name)
#else
#define AGRMS_ZONE(name)
#endif
// port_stream_rate experiment: per-packet / per-batch zones on the port kernels (AGRMS_PSR_PKT_ZONES, with zones on)
#if defined(AGRMS_PROFILE_ZONES) && defined(AGRMS_PSR_PKT_ZONES)
#define AGRMS_PZONE(name) DeviceZoneScopedN(name)
#else
#define AGRMS_PZONE(name)
#endif

namespace agrms_cb {
constexpr uint32_t cb_x_landing = 0;       // resident landing tiles (all blocks of the core); FIRST, on workers + ports
constexpr uint32_t cb_gamma = 1;           // gamma slice, compact faces-0/1 row-0 pages (row-broadcast by compute)
constexpr uint32_t cb_ones = 2;            // 1.0 reduce scaler of the FPU row collapse
constexpr uint32_t cb_sumsq_tiles = 3;     // per tile-row DEST-accumulated sum of x^2
constexpr uint32_t cb_partial_sumsq = 4;   // k > 1: this member's per-row partial (column 0)
constexpr uint32_t cb_gathered_sumsq = 5;  // k > 1: the k members' partials per tile-row (tile r*k + m)
constexpr uint32_t cb_inv_owned = 6;       // owner combine (k > K_SYM_MAX): inv_rms of the rows this member owns
constexpr uint32_t cb_inv_rms = 7;         // per tile-row rsqrt(mean + eps) (column 0)
constexpr uint32_t cb_x_aligned = 8;       // regime N: dense output-aligned x tiles composed by the reader
constexpr uint32_t cb_out_tiles = 9;       // normalized output tiles
constexpr uint32_t cb_port_chunks = 10;    // port cores only: chunk staging between port reader and sender
constexpr uint32_t cb_x_in = 11;           // native HEIGHT shard: the resident input shard (zero-copy, own slice)
constexpr uint32_t cb_x_stage = 12;        // regime N bfloat8_b: bf16 staging of one landed row (compute -> reader)
constexpr uint32_t cb_x_gamma = 13;        // prescale (Refinement 4): x * gamma of one block, T_acc (compute only)
}  // namespace agrms_cb
