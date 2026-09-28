// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exp ring joint SDPA writer for the named precision recipes B/C/D/E (compute:
// exp_ring_joint_sdpa_recipe.cpp).

#include "ttnn/operations/transformer/sdpa/device/kernels/exp_ring_recipe_cbs.hpp"

struct ExpRingJointWriterPolicy {
    static constexpr uint32_t kKWriterAliasCb = ttnn::operations::transformer::sdpa::exp_ring::kRecipeKWriterAliasCb;
    static constexpr uint32_t kVWriterAliasCb = ttnn::operations::transformer::sdpa::exp_ring::kRecipeVWriterAliasCb;
    static constexpr bool kPassOuterRing = true;
    // Recipe layout: 3 = reduce scaler, 4 = column identity. The recipe folds the scale into its
    // exponential (no scale tile) and masks key tails in the pack thread (no mask tiles; c_3 is its
    // reduce scaler).
    static constexpr uint32_t kColIdentityCb = 4;
    static constexpr uint32_t kReduceScalerCb = 3;
    static constexpr bool kGeneratesScaleTile = false;
    static constexpr bool kGeneratesMaskTiles = false;
    // C/D keep a single K/V slot.
    static constexpr bool kDrainPhaseAlignmentAfterOutput = true;
};

#include "exp_ring_joint_writer_impl.hpp"
