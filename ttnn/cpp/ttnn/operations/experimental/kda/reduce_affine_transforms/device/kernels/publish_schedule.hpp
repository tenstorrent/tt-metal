// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// The head reduction's publish schedule, which its compute and dataflow kernels must follow identically. Both have
// already returned for groups at or beyond `active`. The last active group composes the whole head and publishes it
// once: unchanged when it is the only active group, otherwise in its last composition stage, the one at which every
// later distance exceeds its index.
namespace kda_reduce {

inline bool publishes_unchanged(uint32_t active) { return active == 1; }
inline bool publishes_at_stage(uint32_t group, uint32_t active, uint32_t distance) {
    return group + 1 == active && 2 * distance >= active;
}

}  // namespace kda_reduce
