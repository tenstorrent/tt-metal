// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file welford_combine.h
 * @brief Modular functions for combining multiple Welford statistics (mean and variance)
 *        from subgroups. Assumes population variance (divide by subgroup size).
 *        Typically invoked from reader kernels (BRISC or NCRISC).
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

/// @brief Helper functions for float <-> bfloat16 conversion.
namespace detail {
/**
 * @brief Convert bfloat16 (uint16_t) to float for arithmetic.
 * @param bf16 bfloat16 value as uint16_t.
 * @return float representation.
 */
inline float bfloat16_to_float(uint16_t bf16) {
    uint32_t float_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(float));
    return result;
}

/**
 * @brief Convert float to bfloat16 (stored as uint16_t).
 * @param f Float value.
 * @return bfloat16 representation as uint16_t.
 */
inline uint16_t float_to_bfloat16(float f) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &f, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}
}  // namespace detail

/**
 * @brief Struct to hold Welford stats for a single subgroup (mean, variance, count).
 * @tparam T Data type for mean and variance (float or uint16_t).
 */
template <typename T>
struct WelfordStats {
    T mean;          // Mean of the subgroup.
    T variance;      // Variance of the subgroup.
    uint32_t count;  // Number of elements in the subgroup.
};

/**
 * @brief Combine two sets of Welford stats into one.
 *        This is the core building block—use iteratively for multiple groups.
 * @param a First WelfordStats.
 * @param b Second WelfordStats.
 * @return Combined WelfordStats.
 */
WelfordStats<float> combine(const WelfordStats<float>& a, const WelfordStats<float>& b) {
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

/**
 * @brief Combine ARRAY_SIZE subgroup stats into overall mean and variance (float version).
 * @tparam ARRAY_SIZE Number of subgroups.
 * @tparam COUNT_PER_VALUE Number of elements per subgroup.
 * @tparam STRIDE Stride between elements in input arrays.
 * @param means Array of subgroup means.
 * @param vars Array of subgroup variances.
 * @return Combined WelfordStats<float> for all subgroups.
 */
template <uint32_t ARRAY_SIZE, uint32_t COUNT_PER_VALUE, uint32_t STRIDE>
WelfordStats<float> combine_welford_stats(const float* means, const float* vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");

    WelfordStats<float> result;
    result.mean = means[0];
    result.variance = vars[0];
    result.count = COUNT_PER_VALUE;

    for (uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        WelfordStats<float> next;
        next.mean = means[i * STRIDE];
        next.variance = vars[i * STRIDE];
        next.count = COUNT_PER_VALUE;
        result = combine(result, next);
    }

    return result;
}

/**
 * @brief Combine ARRAY_SIZE subgroup stats into overall mean and variance (bfloat16 version).
 *        Unified template that handles both volatile and non-volatile pointers.
 * @tparam ARRAY_SIZE Number of subgroups.
 * @tparam COUNT_PER_VALUE Number of elements per subgroup.
 * @tparam STRIDE Stride between elements in input arrays.
 * @tparam T Pointer type (uint16_t* or volatile uint16_t*).
 * @param means Array of subgroup means (bfloat16).
 * @param vars Array of subgroup variances (bfloat16).
 * @return Combined WelfordStats<uint16_t> for all subgroups.
 */
template <uint32_t ARRAY_SIZE, uint32_t COUNT_PER_VALUE, uint32_t STRIDE, typename T>
WelfordStats<uint16_t> combine_welford_stats(T means, T vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");
    static_assert(
        std::is_same_v<std::remove_volatile_t<std::remove_pointer_t<T>>, uint16_t>,
        "T must be uint16_t* or volatile uint16_t*");

    WelfordStats<float> overall;
    overall.mean = detail::bfloat16_to_float(means[0]);
    overall.variance = detail::bfloat16_to_float(vars[0]);
    overall.count = COUNT_PER_VALUE;

    for (uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        WelfordStats<float> next;
        next.mean = detail::bfloat16_to_float(means[i * STRIDE]);
        next.variance = detail::bfloat16_to_float(vars[i * STRIDE]);
        next.count = COUNT_PER_VALUE;
        overall = combine(overall, next);
    }

    WelfordStats<uint16_t> result;
    result.mean = detail::float_to_bfloat16(overall.mean);
    result.variance = detail::float_to_bfloat16(overall.variance);
    result.count = overall.count;

    return result;
}
