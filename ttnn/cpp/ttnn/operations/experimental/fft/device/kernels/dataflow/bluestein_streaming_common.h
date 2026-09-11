// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace bluestein_streaming {

inline constexpr uint32_t chunk_elements = 1024;
inline constexpr uint32_t element_bytes = sizeof(float);

constexpr uint32_t chunk_count(uint32_t elements) {
    return elements / chunk_elements + (elements % chunk_elements != 0u);
}

// Saturating subtraction avoids an unsigned underflow in wholly padded chunks.
constexpr uint32_t valid_elements(uint32_t elements, uint32_t chunk) {
    const uint32_t start = chunk * chunk_elements;
    if (start >= elements) {
        return 0;
    }
    const uint32_t remaining = elements - start;
    return remaining < chunk_elements ? remaining : chunk_elements;
}

}  // namespace bluestein_streaming
