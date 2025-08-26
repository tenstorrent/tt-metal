// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Modular functions for combining Welford statistics (mean and variance) from subgroups.
// Assumes population variance (divide by subgroup size). For RISC-V compatibility.

#ifndef WELFORD_COMBINE_H
#define WELFORD_COMBINE_H

#include <cstdint>  // For uint32_t

// Struct to hold Welford stats for a single subgroup (mean, variance, count).
struct WelfordStats {
    float mean;
    float variance;  // Population variance (M2 / count)
    uint32_t count;
};

// Combine two sets of Welford stats into one.
// This is the core building block—use iteratively for multiple groups.
WelfordStats combine_two(const WelfordStats& a, const WelfordStats& b) {
    WelfordStats result;
    result.count = a.count + b.count;
    if (result.count == 0) {
        result.mean = 0.0f;
        result.variance = 0.0f;
        return result;
    }

    float delta = b.mean - a.mean;
    result.mean = a.mean + delta * (static_cast<float>(b.count) / result.count);

    float m2_a = a.variance * a.count;
    float m2_b = b.variance * b.count;
    result.variance =
        (m2_a + m2_b + delta * delta * (static_cast<float>(a.count) * b.count / result.count)) / result.count;

    return result;
}

// Combine K subgroup stats into overall mean and variance.
// Input: arrays of means and vars (each of size K), uniform subgroup size m.
// Output: overall_mean and overall_var are populated.
WelfordStats combine_welford(uint32_t K, const float* means, const float* vars, uint32_t m) {
    // Initialize with first subgroup
    WelfordStats overall = {means[0], vars[0], m};

    // Iteratively combine the rest
    for (uint32_t i = 1; i < K; ++i) {
        WelfordStats next = {means[i], vars[i], m};
        overall = combine_two(overall, next);
    }

    return overall;
}

#endif  // WELFORD_COMBINE_H
