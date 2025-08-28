// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Modular functions for combining Welford statistics (mean and variance) from subgroups.
// Assumes population variance (divide by subgroup size). For RISC-V compatibility.

#ifndef WELFORD_COMBINE_H
#define WELFORD_COMBINE_H

#include <cstdint>
#include <cstring>

// Helper functions for bfloat16 conversion
namespace detail {
// Convert bfloat16 (stored as uint16_t) to float for arithmetic
inline float bfloat16_to_float(uint16_t bf16) {
    // Pad with 16 zero bits to make it a valid float
    uint32_t float_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(float));
    return result;
}

// Convert float to bfloat16 (stored as uint16_t)
inline uint16_t float_to_bfloat16(float f) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &f, sizeof(float));
    // Extract upper 16 bits (sign + exponent + upper 7 bits of mantissa)
    return static_cast<uint16_t>(float_bits >> 16);
}
}  // namespace detail

// Struct to hold Welford stats for a single subgroup (mean, variance, count).
template <typename T>
struct WelfordStats {
    T mean;
    T variance;  // Population variance (M2 / count)
    uint32_t count;
};

// Combine two sets of Welford stats into one.
// This is the core building block—use iteratively for multiple groups.
WelfordStats<float> combine_two(const WelfordStats<float>& a, const WelfordStats<float>& b) {
    WelfordStats<float> result;
    result.count = a.count + b.count;

    float delta = b.mean - a.mean;
    result.mean = a.mean + delta * (static_cast<float>(b.count) / result.count);

    float m2_a = a.variance * a.count;
    float m2_b = b.variance * b.count;
    result.variance =
        (m2_a + m2_b + delta * delta * (static_cast<float>(a.count) * b.count / result.count)) / result.count;

    return result;
}

// Combine ARRAY_SIZE subgroup stats into overall mean and variance.
// Input: arrays of means and vars (each of size ARRAY_SIZE), uniform subgroup size COUNT_PER_VALUE.
// Output: overall_mean and overall_var are populated.
template <uint32_t ARRAY_SIZE, uint32_t COUNT_PER_VALUE, uint32_t STRIDE>
WelfordStats<float> combine_welford(const float* means, const float* vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");

    // Initialize with first subgroup
    WelfordStats<float> result;
    result.mean = means[0];
    result.variance = vars[0];
    result.count = COUNT_PER_VALUE;

    // Iteratively combine the rest - compiler will unroll this loop
    for (uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        WelfordStats<float> next;
        next.mean = means[i * STRIDE];
        next.variance = vars[i * STRIDE];
        next.count = COUNT_PER_VALUE;
        result = combine_two(result, next);
    }

    return result;
}

template <uint32_t ARRAY_SIZE, uint32_t COUNT_PER_VALUE, uint32_t STRIDE>
WelfordStats<uint16_t> combine_welford(const uint16_t* means, const uint16_t* vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");

    // Initialize with first subgroup
    WelfordStats<float> overall;
    overall.mean = detail::bfloat16_to_float(means[0]);
    overall.variance = detail::bfloat16_to_float(vars[0]);
    overall.count = COUNT_PER_VALUE;

    // Iteratively combine the rest
    for (uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        WelfordStats<float> next;
        next.mean = detail::bfloat16_to_float(means[i * STRIDE]);
        next.variance = detail::bfloat16_to_float(vars[i * STRIDE]);
        next.count = COUNT_PER_VALUE;
        overall = combine_two(overall, next);
    }

    // Convert back to bfloat16
    WelfordStats<uint16_t> result;
    result.mean = detail::float_to_bfloat16(overall.mean);
    result.variance = detail::float_to_bfloat16(overall.variance);
    result.count = overall.count;

    return result;
}

// Overload for volatile uint16_t pointers
template <uint32_t ARRAY_SIZE, uint32_t COUNT_PER_VALUE, uint32_t STRIDE>
WelfordStats<uint16_t> combine_welford(volatile const uint16_t* means, volatile const uint16_t* vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");

    // Initialize with first subgroup
    WelfordStats<float> overall;
    overall.mean = detail::bfloat16_to_float(means[0]);
    overall.variance = detail::bfloat16_to_float(vars[0]);
    overall.count = COUNT_PER_VALUE;

    // Iteratively combine the rest
    for (uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        WelfordStats<float> next;
        next.mean = detail::bfloat16_to_float(means[i * STRIDE]);
        next.variance = detail::bfloat16_to_float(vars[i * STRIDE]);
        next.count = COUNT_PER_VALUE;
        overall = combine_two(overall, next);
    }

    // Convert back to bfloat16
    WelfordStats<uint16_t> result;
    result.mean = detail::float_to_bfloat16(overall.mean);
    result.variance = detail::float_to_bfloat16(overall.variance);
    result.count = overall.count;

    return result;
}

#endif  // WELFORD_COMBINE_H
