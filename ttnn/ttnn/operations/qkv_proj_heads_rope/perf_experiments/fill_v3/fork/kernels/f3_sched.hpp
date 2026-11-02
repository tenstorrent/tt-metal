// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// fill_v3: K-block schedule. F3_KSIZES (comma list) = per-round K sizes, each <= k_block (the CB slot size;
// slots stay k_block deep, a round fills the leading k_size tiles per row, compact). Default: uniform k_block.
#pragma once

#include <cstdint>

namespace f3 {

#ifdef F3_KSIZES
inline constexpr uint32_t sizes[] = {F3_KSIZES};
inline constexpr uint32_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
#endif

template <uint32_t k_block>
constexpr uint32_t ksize(uint32_t kb) {
#ifdef F3_KSIZES
    return sizes[kb];
#else
    return k_block;
#endif
}

template <uint32_t k_block>
constexpr uint32_t kfirst(uint32_t kb) {
    uint32_t sum = 0;
    for (uint32_t b = 0; b < kb; ++b) {
        sum += ksize<k_block>(b);
    }
    return sum;
}

template <uint32_t k_block, uint32_t num_k_blocks, uint32_t k_total>
constexpr bool valid() {
#ifdef F3_KSIZES
    if (num_sizes != num_k_blocks) {
        return false;
    }
    for (uint32_t b = 0; b < num_k_blocks; ++b) {
        if (sizes[b] == 0 || sizes[b] > k_block) {
            return false;
        }
    }
#endif
    return kfirst<k_block>(num_k_blocks) == k_total;
}

}  // namespace f3
