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

// Construct already-combined statistics without converting packed BF16 values.
template <typename T>
constexpr WelfordStats<T> make_welford_stats(T mean, T variance, std::uint32_t count) {
    return {.mean = mean, .variance = variance, .count = count};
}

/**
 * @brief Statistics over a population that excludes padding zeros, from statistics that include them.
 * @param padded Mean and population variance (bfloat16) and count over the padded population.
 * @param scale Padded count / logical count, at least 1.
 *
 * With P = scale * N values of which P - N are zeros, sum(x) = P * mean and
 * sum(x^2) = P * (variance + mean^2), so the logical mean is scale * mean and the logical variance is
 * scale * variance - scale * (scale - 1) * mean^2. The second form avoids the cancellation of
 * sum(x^2) / N - mean^2 when scale is close to 1. The bfloat16 inputs still leave a variance error
 * that grows with (scale - 1) * mean^2.
 */
inline WelfordStats<std::uint16_t> unpad_welford_stats(const WelfordStats<std::uint16_t>& padded, float scale) {
    const float m = bf16_to_fp32(padded.mean);
    const float v = bf16_to_fp32(padded.variance);
    float logical_variance = scale * v - scale * (scale - 1.0f) * m * m;
    logical_variance = logical_variance < 0.0f ? 0.0f : logical_variance;
    return {
        .mean = fp32_to_bf16_truncate(scale * m),
        .variance = fp32_to_bf16_truncate(logical_variance),
        .count = static_cast<std::uint32_t>(static_cast<float>(padded.count) / scale + 0.5f)};
}

/**
 * @brief Zeroes the rows at and below valid_rows of a 32x32 tile in L1, in the 16x16 face layout.
 * @tparam elem_bytes 2 for bfloat16, 4 for float32.
 */
template <std::uint32_t elem_bytes>
inline void zero_tile_rows_from(std::uint32_t tile_addr, std::uint32_t valid_rows) {
    static_assert(elem_bytes == 2 || elem_bytes == 4, "zero_tile_rows_from supports bfloat16 and float32 tiles");
    constexpr std::uint32_t face_dim = 16;
    constexpr std::uint32_t face_row_words = face_dim * elem_bytes / 4;
    constexpr std::uint32_t face_words = face_dim * face_row_words;
    auto* words = reinterpret_cast<volatile std::uint32_t*>(tile_addr);
    for (std::uint32_t row = valid_rows; row < 2 * face_dim; ++row) {
        // Rows 0-15 are in faces 0 and 1, rows 16-31 in faces 2 and 3.
        const std::uint32_t first_face = row < face_dim ? 0 : 2;
        const std::uint32_t face_row = row % face_dim;
        for (std::uint32_t face = first_face; face < first_face + 2; ++face) {
            auto* p = words + face * face_words + face_row * face_row_words;
            for (std::uint32_t w = 0; w < face_row_words; ++w) {
                p[w] = 0;
            }
        }
    }
}

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

    const float delta = b.mean - a.mean;
    result.mean = a.mean + delta * (static_cast<float>(b.count) / static_cast<float>(result.count));

    const float m2_a = a.variance * static_cast<float>(a.count);
    const float m2_b = b.variance * static_cast<float>(b.count);
    result.variance =
        (m2_a + m2_b +
         delta * delta *
             (static_cast<float>(a.count) * static_cast<float>(b.count) / static_cast<float>(result.count))) /
        static_cast<float>(result.count);

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
    const float means_m2 = mean_delta_sq_sum - (mean_delta_sum * mean_delta);

    WelfordStats<std::uint16_t> result;
    result.mean = fp32_to_bf16_truncate(base_mean + mean_delta);
    result.variance = fp32_to_bf16_truncate((var_sum + means_m2) * inv_size);
    result.count = ARRAY_SIZE * COUNT_PER_VALUE;

    return result;
}

/**
 * @brief The equal-count combine of bfloat16 subgroup stats, with the number of subgroups and their
 *        size given at run time, returning fp32 stats. Same arithmetic as the template above, without
 *        its final rounding to bfloat16. array_size must be at least 1.
 */
template <std::uint32_t STRIDE, typename T>
inline WelfordStats<float> combine_welford_stats_fp32(
    T means, T vars, std::uint32_t array_size, std::uint32_t count_per_value) {
    static_assert(
        std::is_same_v<std::remove_volatile_t<std::remove_pointer_t<T>>, std::uint16_t>,
        "T must be uint16_t* or volatile uint16_t*");

    const float base_mean = bf16_to_fp32(means[0]);
    float mean_delta_sum = 0.0f;
    float mean_delta_sq_sum = 0.0f;
    float var_sum = bf16_to_fp32(vars[0]);

    for (std::uint32_t i = 1; i < array_size; ++i) {
        const float next_mean = bf16_to_fp32(means[i * STRIDE]);
        const float delta = next_mean - base_mean;
        mean_delta_sum += delta;
        mean_delta_sq_sum += delta * delta;
        var_sum += bf16_to_fp32(vars[i * STRIDE]);
    }

    const float inv_size = 1.0f / static_cast<float>(array_size);
    const float mean_delta = mean_delta_sum * inv_size;
    const float means_m2 = mean_delta_sq_sum - (mean_delta_sum * mean_delta);

    WelfordStats<float> result;
    result.mean = base_mean + mean_delta;
    result.variance = (var_sum + means_m2) * inv_size;
    result.count = array_size * count_per_value;

    return result;
}

/**
 * @brief The bfloat16 combine with the number of subgroups and their size given at run time.
 *        Same arithmetic as the template above. array_size must be at least 1.
 */
template <std::uint32_t STRIDE, typename T>
inline WelfordStats<std::uint16_t> combine_welford_stats(
    T means, T vars, std::uint32_t array_size, std::uint32_t count_per_value) {
    const WelfordStats<float> result = combine_welford_stats_fp32<STRIDE>(means, vars, array_size, count_per_value);
    return {
        .mean = fp32_to_bf16_truncate(result.mean),
        .variance = fp32_to_bf16_truncate(result.variance),
        .count = result.count};
}

/**
 * @brief Combine array_size bfloat16 subgroup stats whose first num_first subgroups hold first_count
 *        elements each and the others rest_count each.
 *
 * Each run of equal counts is combined as above, and the two runs are merged with Chan's parallel
 * algorithm, weighted by their counts. With num_first == 0 this is the equal-count combine, bit for
 * bit. num_first must be below array_size.
 */
template <std::uint32_t STRIDE, typename T>
inline WelfordStats<std::uint16_t> combine_welford_stats_two_counts(
    T means,
    T vars,
    std::uint32_t array_size,
    std::uint32_t num_first,
    std::uint32_t first_count,
    std::uint32_t rest_count) {
    if (num_first == 0) {
        return combine_welford_stats<STRIDE>(means, vars, array_size, rest_count);
    }
    const WelfordStats<float> first = combine_welford_stats_fp32<STRIDE>(means, vars, num_first, first_count);
    const WelfordStats<float> rest = combine_welford_stats_fp32<STRIDE>(
        means + num_first * STRIDE, vars + num_first * STRIDE, array_size - num_first, rest_count);
    const WelfordStats<float> result = combine(first, rest);
    return {
        .mean = fp32_to_bf16_truncate(result.mean),
        .variance = fp32_to_bf16_truncate(result.variance),
        .count = result.count};
}
