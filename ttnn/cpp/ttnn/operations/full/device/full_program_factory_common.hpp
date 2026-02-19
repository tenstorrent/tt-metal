// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::operations::full {
union datatype {
    uint32_t u32;
    float f32;
};

// Tie-to-even rounding for float32 -> bfloat16 conversion.
// After the full modification and if there are no issues in the overall tests, it will be added to `bfloat16.hpp` and
// applied globally.
inline uint32_t get_bfloat16_rounded(const float val) {
    uint32_t float_bits = *reinterpret_cast<const uint32_t*>(&val);

    // upper 16 bits
    uint16_t bfloat16_bits = float_bits >> 16;

    // check Guard, Round, Sticky bits from lower 16 bits
    uint32_t lower_bits = float_bits & 0xFFFF;
    uint32_t guard_bit = (lower_bits >> 15) & 1;
    uint32_t round_bit = (lower_bits >> 14) & 1;
    uint32_t sticky_bit = (lower_bits & 0x3FFF) != 0;

    if (guard_bit && (round_bit || sticky_bit || (bfloat16_bits & 1))) {
        bfloat16_bits += 1;
    }

    return static_cast<uint32_t>(bfloat16_bits) << 16;
}

}  // namespace ttnn::operations::full
