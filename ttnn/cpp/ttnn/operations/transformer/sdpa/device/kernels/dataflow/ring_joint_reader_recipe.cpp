// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA reader for the named precision recipes B/C/D/E (compute: ring_joint_sdpa_recipe.cpp).

#pragma GCC optimize("Os")  // leaves kernel config buffer for the recipe compute

#include <cstdint>

struct RingJointReaderPolicy {
    // Recipes accept odd Q tile counts with their 2-row QK subblock (the last group is a single row).
    static constexpr bool kPartialQSubblocks = true;
};

#include "ring_joint_reader_impl.hpp"
