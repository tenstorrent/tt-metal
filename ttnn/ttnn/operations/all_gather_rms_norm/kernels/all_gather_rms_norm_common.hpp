// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm — CB indices shared by every kernel. Must match the CB_* constants in
// all_gather_rms_norm_program_descriptor.py.
#pragma once

#include <cstdint>

namespace agrms_cb {
constexpr uint32_t cb_x_landing = 0;       // resident landing tiles (all blocks of the core); FIRST, on workers + ports
constexpr uint32_t cb_gamma = 1;           // gamma slice, compact faces-0/1 row-0 pages (row-broadcast by compute)
constexpr uint32_t cb_ones = 2;            // 1.0 reduce scaler of the FPU row collapse
constexpr uint32_t cb_sumsq_tiles = 3;     // per tile-row DEST-accumulated sum of x^2
constexpr uint32_t cb_partial_sumsq = 4;   // k > 1: this member's per-row partial (column 0)
constexpr uint32_t cb_gathered_sumsq = 5;  // k > 1: the k members' partials per tile-row (tile r*k + m)
constexpr uint32_t cb_inv_owned = 6;       // reserved (owner combine)
constexpr uint32_t cb_inv_rms = 7;         // per tile-row rsqrt(mean + eps) (column 0)
constexpr uint32_t cb_x_aligned = 8;       // regime N: dense output-aligned x tiles composed by the reader
constexpr uint32_t cb_out_tiles = 9;       // normalized output tiles
constexpr uint32_t cb_port_chunks = 10;    // port cores only: chunk staging between port reader and sender
}  // namespace agrms_cb
