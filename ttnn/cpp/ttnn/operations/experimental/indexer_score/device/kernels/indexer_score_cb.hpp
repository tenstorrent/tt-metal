// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CB-index argument layout for indexer_score. The factory allocates the concrete (continuous)
// circular-buffer indices and forwards them to the reader / compute / writer kernels as compile-time
// args; this enum is the single-sourced slot order both sides agree on (host push-order ==
// device read-order), so the index of any buffer can never drift between host and device.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// One slot per circular buffer, in the order the factory appends the CB indices to the common
// compile-time args. Kernels read each back at (num common dim args) + the matching slot.
enum CbArg : uint32_t {
    cb_q_arg,          // q head-group block: heads_per_group * QC * Dt tiles
    cb_k_arg,          // k chunk, double buffered
    cb_w_arg,          // resident gate (w) group: Hi * QC tiles
    cb_mask_arg,       // [diag strict-upper -inf, full -inf], built once
    cb_qk_arg,         // relu(q.kT) for a whole head group
    cb_acc_strip_arg,  // unit accumulator: QC x KC strip (untilize input)
    cb_out_strip_arg,  // untilized row-major strip output
    num_cb_args
};

// Two mask tiles in cb_mask: index 0 = diagonal strict-upper -inf, index 1 = full -inf.
constexpr uint32_t num_mask_tiles = 2;

// Per-direction multicast role, written by the factory into the reader's runtime args and switched on by
// the reader. Single-sourced here so the host writer and device reader can't drift on the encoding.
enum McastRole : uint32_t {
    mcast_role_none = 0,      // no multicast: plain per-core DRAM read
    mcast_role_sender = 1,    // reads DRAM, then multicasts the block to the rect
    mcast_role_receiver = 2,  // waits for the sender's multicast into its slot
};

}  // namespace ttnn::operations::experimental::indexer_score
