// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Ring joint SDPA on the precision recipes: the circular buffers the kernels share (host:
// ring_joint_sdpa_program_factory.cpp). CBs 0-16 (and 29-31 with fused chunks) are the recipe compute layout
// (recipe_compute_program): Q, K, V, the reduce scaler, the column identity, the reciprocal scratch, QK, the state banks
// (O 8/9, maxima 10/11, sums 12/13), exp(max diff) and the output. CBs 17/18 carry the state checkpoint request / ack
// between compute and writer for cores with several Q chunks (recipe_checkpoint.hpp, recipe_state_transfer.hpp); the
// ring dataflow CBs follow from 19.
namespace recipe_ring_cb {
constexpr uint32_t q = 0, k = 1, v = 2, identity_scale = 3, col_identity = 4, recip_scratch = 5, qk = 6;
constexpr uint32_t out_a = 8, out_b = 9, max_a = 10, max_b = 11, sum_a = 12, sum_b = 13, exp_max_diff = 14, out = 16;
constexpr uint32_t checkpoint_request = 17, checkpoint_ack = 18;
// The attention sink's tile (reader -> compute) and the reader's derived lengths (reader -> compute).
constexpr uint32_t sink = 19, derived = 20;
}  // namespace recipe_ring_cb
