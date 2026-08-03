// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <tt_stl/assert.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::uniform::detail {

struct OutputRange {
    float lower_bound;
    float upper_bound;
};

inline float bfloat16_ceil(float value) {
    constexpr uint32_t lower_bits_mask = 0x0000FFFFU;
    constexpr uint32_t bfloat16_step = 0x00010000U;

    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t truncated_bits = bits & ~lower_bits_mask;
    if ((bits & lower_bits_mask) == 0) {
        return value;
    }

    // Dropping the low bits rounds negative values towards +infinity. Positive
    // values need the next BF16 encoding to obtain the mathematical ceiling.
    const uint32_t ceiling_bits = (bits >> 31) ? truncated_bits : truncated_bits + bfloat16_step;
    return std::bit_cast<float>(ceiling_bits);
}

inline float bfloat16_predecessor(float value) {
    constexpr uint32_t lower_bits_mask = 0x0000FFFFU;
    constexpr uint32_t magnitude_mask = 0x7FFFFFFFU;
    constexpr uint32_t bfloat16_step = 0x00010000U;
    constexpr uint32_t negative_min_subnormal = 0x80010000U;

    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t truncated_bits = bits & ~lower_bits_mask;
    if ((bits & lower_bits_mask) != 0) {
        // Truncation is already below a positive input. For a negative input it
        // rounds upwards, so advance one encoding towards -infinity.
        const uint32_t predecessor_bits = (bits >> 31) ? truncated_bits + bfloat16_step : truncated_bits;
        return std::bit_cast<float>(predecessor_bits);
    }

    if ((bits & magnitude_mask) == 0) {
        return std::bit_cast<float>(negative_min_subnormal);
    }

    const uint32_t predecessor_bits = (bits >> 31) ? bits + bfloat16_step : bits - bfloat16_step;
    return std::bit_cast<float>(predecessor_bits);
}

inline OutputRange make_output_range(float from, float to, DataType output_dtype) {
    OutputRange range;
    if (output_dtype == DataType::BFLOAT16) {
        // The writer converts FP32 to BF16 by dropping the low 16 bits. Restrict
        // the generated interval to exact BF16 endpoints so that conversion
        // cannot cross either requested bound.
        range.lower_bound = bfloat16_ceil(from);
        range.upper_bound = bfloat16_predecessor(to);
    } else {
        range.lower_bound = from;
        range.upper_bound = std::nextafter(to, -std::numeric_limits<float>::infinity());
    }

    TT_FATAL(
        range.lower_bound <= range.upper_bound,
        "Requested range [{}, {}) contains no value representable in the output dtype",
        from,
        to);
    return range;
}

}  // namespace ttnn::operations::uniform::detail
