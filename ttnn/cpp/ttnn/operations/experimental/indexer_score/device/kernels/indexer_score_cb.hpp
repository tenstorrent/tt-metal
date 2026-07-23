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

// Runtime-arg slot layout for the FUSED ring path, single-sourced host<->device so the factory's push order
// and the kernels' read order cannot drift. The kernels index a handful of these slots by name (rather than a
// bare literal), and the factory's builder pushes to the SAME offsets; a mismatch is caught by the static_assert
// in the factory that pins these to their concrete values. Slots 0..(kv_len_tiles) mirror the classic factory's
// reader layout; the fused path then appends its own tail.
namespace fused_rt {
constexpr uint32_t sched_width =
    6;  // schedule array {row_group0, group_rows, num_groups, band0, col_num_bands, max_bands}
constexpr uint32_t causal_scalars = 4;      // {kv_len_tiles, chunk_start_tiles, straddle_q_tile, straddle_jump_tiles}
constexpr uint32_t mcast_args_per_dir = 8;  // role, rect(xs,ys,xe,ye), sender(sx,sy), ndst
constexpr uint32_t reader_num_mcast_dirs = 2;     // K column, then Q/W row
constexpr uint32_t reader_fused_block_width = 6;  // {ring_size, ring_index, fwd, bwd, sem0, sem1}

// Compute RT: schedule(6), the 4 causal scalars, then the band-visit permutation (one entry per band).
constexpr uint32_t compute_band_perm_base = sched_width + causal_scalars;  // 10
// Writer RT: out addr(1), schedule(6), the 4 causal scalars, then the permutation.
constexpr uint32_t writer_band_perm_base = 1 + sched_width + causal_scalars;  // 11
// Reader RT: q/k/w addrs(3), schedule(6), 2 mcast dirs (8 args each), k_batch_offset(1), kv_len_tiles(1) -> fused
// block.
constexpr uint32_t reader_k_batch_offset = 3 + sched_width + reader_num_mcast_dirs * mcast_args_per_dir;  // 25
constexpr uint32_t reader_kv_len_tiles = reader_k_batch_offset + 1;                                       // 26
constexpr uint32_t reader_fused_rt_base = reader_kv_len_tiles + 1;                                        // 27
constexpr uint32_t reader_k_local_addr = reader_fused_rt_base + reader_fused_block_width;                 // 33
constexpr uint32_t reader_band_perm_base = reader_k_local_addr + 1;                                       // 34
}  // namespace fused_rt

}  // namespace ttnn::operations::experimental::indexer_score
