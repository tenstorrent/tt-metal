// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Ring and exp ring joint SDPA on the precision recipes: the circular buffers the kernels share (host:
// ring_joint_sdpa_program_factory.cpp, exp_ring_joint_sdpa_program_factory.cpp). CBs 0-16 (and 29-31 with fused chunks)
// are the recipe compute layout (recipe_compute_program): Q, K, V, the reduce scaler, the column identity, the
// reciprocal scratch, QK, the state banks (O 8/9, maxima 10/11, sums 12/13), exp(max diff) and the output. CBs 17/18
// are recipe_checkpoint.hpp's state request / ack (the sparse split-K handoff; the rings keep each Q chunk's state
// resident over its pass); the ring dataflow CBs follow from 19.
namespace recipe_ring_cb {
constexpr uint32_t q = 0, k = 1, v = 2, identity_scale = 3, col_identity = 4, recip_scratch = 5, qk = 6;
constexpr uint32_t out_a = 8, out_b = 9, max_a = 10, max_b = 11, sum_a = 12, sum_b = 13, exp_max_diff = 14, out = 16;
constexpr uint32_t checkpoint_request = 17, checkpoint_ack = 18;
// The sparse ring's PV scratch (recipe::pv_scratch_cb; CB 7 is only the FP32 recipes' score alias, and the sparse ring
// runs the BF16-state recipes).
constexpr uint32_t pv_scratch = 7;
// The attention sink's tile (reader -> compute) and the reader's derived lengths (reader -> compute).
constexpr uint32_t sink = 19, derived = 20;
// Exp ring: second handles on the K and V CBs for the fabric MUX writer, which pops them independently of compute.
constexpr uint32_t k_writer = 22, v_writer = 23;
// Sliding ring K split (recipe_split_merge.hpp): a band's state handoff page (compute -> writer: where the state sits),
// and on the reducer the received state (O rows, maxima, l) and the merge's c_b scratch.
constexpr uint32_t ksplit_signal = 24, ksplit_recv_out = 25, ksplit_recv_max = 26, ksplit_recv_sum = 27,
                   ksplit_corr = 28;
// The handoff page's words: the CB holding the band's maxima (CB 9 holds O and CB 13 l, at their bases) and whether
// the band ran any K chunk.
constexpr uint32_t ksplit_word_max_cb = 0, ksplit_word_has_state = 1;
}  // namespace recipe_ring_cb
