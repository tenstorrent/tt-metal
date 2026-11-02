// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// How `value` is laid out, shared by the host (validation, program factories)
// and the dataflow kernels, which receive it as a compile-time arg. Dependency-free
// so both sides can include it.
namespace fused_msda_value_layout {

enum class ValueLayout : uint32_t {
    Canonical = 0,  // (B, S, H, D)              one D-wide page per (b, s, h)
    Packed = 1,     // (B, S, H*D)               one page per (b, s), head h at h*D*2
    HeadMajor = 2,  // (B, H, ceil(S/32), 32*D)  32 keys of one head per page, so x and
                    //                           x+1 of a head sit side by side
};

// One head-major page is one 32-key tile row of a head: the producer untilizes
// a head's tiles of one tile row into exactly one page.
constexpr uint32_t HEAD_MAJOR_KEYS_PER_PAGE = 32;

constexpr uint32_t head_major_pages_per_head(uint32_t num_keys) {
    return (num_keys + HEAD_MAJOR_KEYS_PER_PAGE - 1) / HEAD_MAJOR_KEYS_PER_PAGE;
}

}  // namespace fused_msda_value_layout
