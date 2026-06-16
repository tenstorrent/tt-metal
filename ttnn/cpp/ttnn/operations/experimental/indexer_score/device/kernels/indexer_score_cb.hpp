// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CB indices for indexer_score, single-sourced for the host factory and the kernels:
// both reference these same indices, so defining them once prevents corrupt/hang from
// index drift. Host+device safe (tt::CBIndex is in the shared hostdevcommon header).

#pragma once

#include <cstdint>

#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::experimental::indexer_score {

// Inputs (reader -> compute) and the persistent mask.
constexpr uint32_t cb_q = tt::CBIndex::c_0;     // q head-group block: heads_per_group * QC * Dt tiles
constexpr uint32_t cb_k = tt::CBIndex::c_1;     // k chunk, double buffered
constexpr uint32_t cb_w = tt::CBIndex::c_2;     // resident gate (w) group: Hi * QC tiles
constexpr uint32_t cb_mask = tt::CBIndex::c_3;  // [diag strict-upper -inf, full -inf], built once

// Compute intermediates.
constexpr uint32_t cb_qk = tt::CBIndex::c_24;         // relu(q.kT) for a whole head group
constexpr uint32_t cb_acc_strip = tt::CBIndex::c_27;  // unit accumulator: QC x KC strip (untilize input)

// Compute -> writer outputs.
constexpr uint32_t cb_out_strip = tt::CBIndex::c_18;  // untilized row-major strip output
constexpr uint32_t cb_scratch = tt::CBIndex::c_17;    // writer-only -inf scratch tile

// Two mask tiles in cb_mask: index 0 = diagonal strict-upper -inf, index 1 = full -inf.
constexpr uint32_t num_mask_tiles = 2;

}  // namespace ttnn::operations::experimental::indexer_score
