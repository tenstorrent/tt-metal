// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exp ring joint SDPA reader for the named precision recipes B/C/D/E (compute:
// exp_ring_joint_sdpa_recipe.cpp).

#include "ttnn/operations/transformer/sdpa/device/kernels/exp_ring_recipe_cbs.hpp"

struct ExpRingJointReaderPolicy {
    // The recipe CB layout owns 0-16 (c_14 is its exp_max_diff); the MUX-writer aliases move above it.
    static constexpr uint32_t kKWriterAliasCb = ttnn::operations::transformer::sdpa::exp_ring::kRecipeKWriterAliasCb;
    static constexpr uint32_t kVWriterAliasCb = ttnn::operations::transformer::sdpa::exp_ring::kRecipeVWriterAliasCb;
    // Odd Q tile counts with the recipe's 2-row QK subblock.
    static constexpr bool kPartialQSubblocks = true;
    // One resident recurrent state and Q chunk per pass across the whole ring.
    static constexpr bool kPassOuterRing = true;
    // C/D keep one K slot (applied to every recipe so all share one schedule).
    static constexpr bool kCreditKAfterQ = true;
};

#include "exp_ring_joint_reader_impl.hpp"
