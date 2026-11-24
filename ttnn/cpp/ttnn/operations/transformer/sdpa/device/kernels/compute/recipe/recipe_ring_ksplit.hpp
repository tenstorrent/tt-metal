// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// K split of the sliding ring wrappers (RingJointSDPA's sliding pass, Gemma4SlidingRingSDPA; ring_joint_ksplit.hpp):
// cores with one (head, Q chunk) unit each run a band of its sliding work plan, non-final. A sender band hands its
// state to its writer (ring_ksplit_handoff.hpp signals the reducer, whose writer pulls it into the receive CBs); the
// reducer band (the last: it holds the diagonal chunk) merges the senders' states in band order
// (recipe_split_merge.hpp) and normalizes. The BF16-state recipes: O and l stay in CB 9 / CB 13 at their bases. Include
// after recipe/recipe_ring.hpp.
#include "recipe_split_merge.hpp"
#include "../../ring_joint_ksplit.hpp"

using RecipeRingKSplitReceived = recipe_merge::Received<
    recipe_ring_cb::ksplit_recv_out,
    recipe_ring_cb::ksplit_recv_max,
    recipe_ring_cb::ksplit_recv_sum,
    recipe_ring_cb::ksplit_corr>;

// A sender band's state (has_state: the band ran K chunks) to the writer. (The wrappers assert the BF16-state recipes:
// a static_assert here would fire in the discarded branches of their FP32 builds.)
ALWI void recipe_ring_ksplit_publish(const RecipeAccumulatorState& state, bool has_state) {
    recipe_merge::publish_state<
        recipe_ring_cb::ksplit_signal,
        recipe_ring_cb::ksplit_word_max_cb,
        recipe_ring_cb::ksplit_word_has_state>(state, has_state);
}

// The reducer band: the senders' states merged into its own in band order (a sender whose band was empty is popped
// unmerged), then the recipe's normalization into the output CB. Size-optimized: once per call.
template <uint32_t Sq_chunk_t, uint32_t vDHt, uint32_t scale_fp32, uint32_t ksplit_count>
__attribute__((noinline, optimize("Os"))) void recipe_ring_ksplit_merge_and_normalize(
    RecipeAccumulatorState& state, uint32_t plan_k_chunks) {
    namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;
    using namespace recipe_ring_cb;
    // The last sender band that ran K chunks: earlier merges keep the merged maxima for the next one.
    uint32_t last_live = ksplit_count;
    for (uint32_t band = 0; band + 1 < ksplit_count; ++band) {
        if (!ring_joint::sliding_ksplit_range(plan_k_chunks, band, ksplit_count).empty()) {
            last_live = band;
        }
    }
    for (uint32_t band = 0; band + 1 < ksplit_count; ++band) {
        if (ring_joint::sliding_ksplit_range(plan_k_chunks, band, ksplit_count).empty()) {
            recipe_merge::discard<Sq_chunk_t, vDHt>(RecipeRingKSplitReceived{});
        } else {
            recipe_merge::merge_bf16<Sq_chunk_t, vDHt, scale_fp32, exp_max_diff, RecipeRingKSplitReceived>(
                state, band != last_live);
        }
    }
    recipe_merge::normalize_bf16<Sq_chunk_t, vDHt>(state, [](uint32_t n_sum, uint32_t n_out) {
        pack_reconfig_data_format(recip_scratch);
        // The step's own copy of the normalization (the cold one with fused chunks).
        if constexpr (recipe::fused_active) {
            recipe_cold_step::normalize_row_streaming<
                vDHt,
                compute_kernel_lib::DEST_AUTO_LIMIT,
                col_identity,
                recip_scratch,
                out,
                scale_fp32>(n_sum, n_out, Sq_chunk_t);
        } else {
            recipe_hot_step::normalize_row_streaming<
                vDHt,
                compute_kernel_lib::DEST_AUTO_LIMIT,
                col_identity,
                recip_scratch,
                out,
                scale_fp32>(n_sum, n_out, Sq_chunk_t);
        }
    });
}
