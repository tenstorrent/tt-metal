// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CB-index argument layout for indexer_score. The factory forwards the concrete CB indices to the kernels
// as compile-time args; this enum is the single-sourced slot order both sides agree on (host push-order ==
// device read-order), so no buffer index can drift between host and device.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// One slot per circular buffer, in the order the factory appends the CB indices. Kernels read each back
// at (num common dim args) + the matching slot.
enum CbArg : uint32_t {
    cb_q_arg,             // q head-group block: heads_per_group * QC * Dt tiles
    cb_k_arg,             // k chunk, double buffered
    cb_w_arg,             // resident gate (w) group: Hi * QC tiles
    cb_mask_arg,          // [diag strict-upper -inf, full -inf], built once
    cb_qk_arg,            // act(q.kT) for a whole head group
    cb_acc_strip_arg,     // unit accumulator: QC x KC strip (untilize input)
    cb_out_strip_arg,     // untilized row-major strip output (block_size==0); tilized col-0 block-max (pool)
    cb_scaler_arg,        // block-max-pool only: one 1.0 tile (reduce-MAX scaler); index 0 / unused otherwise
    cb_pool_scratch_arg,  // block-max-pool only: writer's one-tile row-assembly scratch; unused otherwise
    num_cb_args
};

// Two mask tiles in cb_mask: index 0 = diagonal strict-upper -inf, index 1 = full -inf.
constexpr uint32_t num_mask_tiles = 2;

// Per-direction multicast role, written by the factory into the reader's runtime args. Single-sourced
// here so host and device can't drift on the encoding.
enum McastRole : uint32_t {
    mcast_role_none = 0,      // no multicast: plain per-core DRAM read
    mcast_role_sender = 1,    // reads DRAM, then multicasts the block to the rect
    mcast_role_receiver = 2,  // waits for the sender's multicast into its slot
};

}  // namespace ttnn::operations::experimental::indexer_score
