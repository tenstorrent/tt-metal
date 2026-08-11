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

namespace ttnn::operations::uniform {

struct InclusiveOutputRange {
    float lower_bound;
    float upper_bound;
};

constexpr std::uint32_t bfloat16_discarded_bits_mask = 0x0000FFFFU;
constexpr std::uint32_t bfloat16_encoding_step = 0x00010000U;
constexpr std::uint32_t float_sign_mask = 0x80000000U;
// FP32 and BF16 have the same exponent field, hence the same minimum normal
// value when BF16 is widened to FP32. SFPU arithmetic flushes subnormals.
constexpr float minimum_normal = std::numeric_limits<float>::min();

inline float smallest_supported_float32_at_least(float value) {
    if (value > 0.0F && value < minimum_normal) {
        return minimum_normal;
    }
    if (value <= 0.0F && value > -minimum_normal) {
        return 0.0F;
    }
    return value;
}

inline float largest_supported_float32_below(float value) {
    if (value > 0.0F && value <= minimum_normal) {
        return 0.0F;
    }
    if (value <= 0.0F && value > -minimum_normal) {
        return -minimum_normal;
    }
    return std::nextafter(value, -std::numeric_limits<float>::infinity());
}

inline float smallest_supported_bfloat16_at_least(float value) {
    if (value > 0.0F && value < minimum_normal) {
        return minimum_normal;
    }
    if (value <= 0.0F && value > -minimum_normal) {
        return 0.0F;
    }

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t truncated_bits = bits & ~bfloat16_discarded_bits_mask;
    if ((bits & bfloat16_discarded_bits_mask) == 0) {
        return value;
    }

    // Dropping the low bits rounds negative values towards +infinity. Positive
    // values need the next BF16 encoding to obtain the mathematical ceiling.
    const std::uint32_t ceiling_bits =
        (bits & float_sign_mask) ? truncated_bits : truncated_bits + bfloat16_encoding_step;
    return std::bit_cast<float>(ceiling_bits);
}

inline float largest_supported_bfloat16_below(float value) {
    if (value > 0.0F && value <= minimum_normal) {
        return 0.0F;
    }
    if (value <= 0.0F && value > -minimum_normal) {
        return -minimum_normal;
    }

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t truncated_bits = bits & ~bfloat16_discarded_bits_mask;
    if ((bits & bfloat16_discarded_bits_mask) != 0) {
        // Truncation is already below a positive input. For a negative input it
        // rounds upwards, so advance one encoding towards -infinity.
        const std::uint32_t predecessor_bits =
            (bits & float_sign_mask) ? truncated_bits + bfloat16_encoding_step : truncated_bits;
        return std::bit_cast<float>(predecessor_bits);
    }

    const std::uint32_t predecessor_bits =
        (bits & float_sign_mask) ? bits + bfloat16_encoding_step : bits - bfloat16_encoding_step;
    return std::bit_cast<float>(predecessor_bits);
}

inline InclusiveOutputRange validate_inclusive_output_range(InclusiveOutputRange range, float from, float to) {
    TT_FATAL(
        range.lower_bound <= range.upper_bound,
        "Requested range [{}, {}) contains no value representable in the output dtype without subnormals",
        from,
        to);

    const float scale = range.upper_bound - range.lower_bound;
    TT_FATAL(std::isfinite(scale), "Requested range [{}, {}) is too wide for a finite FP32 scale", from, to);
    TT_FATAL(
        scale == 0.0F || std::isnormal(scale),
        "Requested range [{}, {}) requires a subnormal scale unsupported by the SFPU",
        from,
        to);
    return range;
}

inline InclusiveOutputRange make_inclusive_output_range(float from, float to, DataType output_dtype) {
    TT_FATAL(
        std::isfinite(from) && std::isfinite(to), "Uniform range endpoints must be finite, got [{}, {})", from, to);
    TT_FATAL(from < to, "Uniform range lower bound must be less than upper bound, got [{}, {})", from, to);

    switch (output_dtype) {
        case DataType::BFLOAT16:
            // The packer converts the FP32 destination to BF16 in hardware. Restrict
            // the generated interval to exact BF16 endpoints so that rounding cannot
            // cross either requested bound.
            return validate_inclusive_output_range(
                {smallest_supported_bfloat16_at_least(from), largest_supported_bfloat16_below(to)}, from, to);
        case DataType::FLOAT32:
            return validate_inclusive_output_range(
                {smallest_supported_float32_at_least(from), largest_supported_float32_below(to)}, from, to);
        default: TT_THROW("Uniform: unsupported output dtype {}", output_dtype);
    }
}

}  // namespace ttnn::operations::uniform
