// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA reader for the named precision recipes B/C/D/E (compute: ring_joint_sdpa_recipe.cpp).

// Size-optimized: the recipe compute kernels (fused FAST in particular) leave little of the
// kernel config buffer, and this kernel mostly issues NOC/fabric transfers and waits on them.
#pragma GCC optimize("Os")

#include <cstdint>

struct RingJointReaderPolicy {
    // Recipes accept odd Q tile counts with their 2-row QK subblock (the last group is a single row).
    static constexpr bool kPartialQSubblocks = true;
};

#include "ring_joint_reader_impl.hpp"
