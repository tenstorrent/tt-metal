// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include <type_traits>

#include "api/numeric/bfloat16.h"

/**
 * @brief Struct to hold Welford stats for a single subgroup (mean, variance, count).
 * @tparam T Data type for mean and variance (float or uint16_t).
 */
template <typename T>
struct WelfordStats {
    T mean;               // Mean of the subgroup.
    T variance;           // Variance of the subgroup.
    std::uint32_t count;  // Number of elements in the subgroup.
};

/**
 * @brief Combine two sets of Welford stats into one.
 *        This is the core building block—use iteratively for multiple groups.
 * @param a First WelfordStats.
 * @param b Second WelfordStats.
 * @return Combined WelfordStats.
 */
inline WelfordStats<float> combine(const WelfordStats<float>& a, const WelfordStats<float>& b) {
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
template <std::uint32_t ARRAY_SIZE, std::uint32_t COUNT_PER_VALUE, std::uint32_t STRIDE>
inline WelfordStats<float> combine_welford_stats(const float* means, const float* vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");

    WelfordStats<float> result;
    result.mean = means[0];
    result.variance = vars[0];
    result.count = COUNT_PER_VALUE;

    for (std::uint32_t i = 1; i < ARRAY_SIZE; ++i) {
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
template <std::uint32_t ARRAY_SIZE, std::uint32_t COUNT_PER_VALUE, std::uint32_t STRIDE, typename T>
inline WelfordStats<std::uint16_t> combine_welford_stats(T means, T vars) {
    static_assert(ARRAY_SIZE > 0, "ARRAY_SIZE must be greater than 0");
    static_assert(
        std::is_same_v<std::remove_volatile_t<std::remove_pointer_t<T>>, std::uint16_t>,
        "T must be uint16_t* or volatile uint16_t*");

    const float base_mean = bf16_to_fp32(means[0]);
    float mean_delta_sum = 0.0f;
    float mean_delta_sq_sum = 0.0f;
    float var_sum = bf16_to_fp32(vars[0]);

    for (std::uint32_t i = 1; i < ARRAY_SIZE; ++i) {
        const float next_mean = bf16_to_fp32(means[i * STRIDE]);
        const float delta = next_mean - base_mean;
        mean_delta_sum += delta;
        mean_delta_sq_sum += delta * delta;
        var_sum += bf16_to_fp32(vars[i * STRIDE]);
    }

    // Equal-sized populations have total variance equal to the average subgroup
    // variance plus the population variance of their means. Centering the means
    // around the first value avoids cancellation from their absolute magnitude:
    // M2(means) = sum(delta^2) - sum(delta)^2 / ARRAY_SIZE.
    constexpr float inv_size = 1.0f / static_cast<float>(ARRAY_SIZE);
    const float mean_delta = mean_delta_sum * inv_size;
    const float means_m2 = mean_delta_sq_sum - mean_delta_sum * mean_delta;

    WelfordStats<std::uint16_t> result;
    result.mean = fp32_to_bf16_truncate(base_mean + mean_delta);
    result.variance = fp32_to_bf16_truncate((var_sum + means_m2) * inv_size);
    result.count = ARRAY_SIZE * COUNT_PER_VALUE;

    return result;
}
