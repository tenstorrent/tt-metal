#pragma once

#include <cstdint>
#include <vector>

namespace tt {
namespace welford {

// Equal-count Welford merge structure
struct EqualCountWelfordMerge {
    float mean;
    float m2;
    float count;
};

// Generic HW reduction with equal-count Welford merge
template <typename T>
void equal_count_welford_merge(
    const std::vector<T>& data,
    EqualCountWelfordMerge& result,
    const std::vector<float>& reciprocal_counts) {

    // Precompute reciprocal scaling factors
    float total_count = 0.0f;
    for (float rc : reciprocal_counts) {
        total_count += 1.0f / rc;
    }

    // Initialize result
    result.mean = 0.0f;
    result.m2 = 0.0f;
    result.count = total_count;

    // Equal-count Welford merge
    for (size_t i = 0; i < data.size(); ++i) {
        float delta = data[i] - result.mean;
        result.mean += delta * reciprocal_counts[i];
        result.m2 += delta * (data[i] - result.mean) * reciprocal_counts[i];
    }
}

// GroupNorm with equal-count Welford merge
template <typename T>
void groupnorm_equal_count_welford(
    const std::vector<T>& data,
    std::vector<T>& output,
    const std::vector<float>& reciprocal_counts,
    float epsilon = 1e-5f) {

    // Equal-count Welford merge for each group
    std::vector<EqualCountWelfordMerge> group_stats(data.size() / reciprocal_counts.size());

    for (size_t i = 0; i < reciprocal_counts.size(); ++i) {
        size_t group_size = static_cast<size_t>(1.0f / reciprocal_counts[i]);
        for (size_t j = 0; j < group_size; ++j) {
            size_t idx = i * group_size + j;
            float delta = data[idx] - group_stats[i].mean;
            group_stats[i].mean += delta * reciprocal_counts[i];
            group_stats[i].m2 += delta * (data[idx] - group_stats[i].mean) * reciprocal_counts[i];
        }
    }

    // Normalize each group
    for (size_t i = 0; i < reciprocal_counts.size(); ++i) {
        size_t group_size = static_cast<size_t>(1.0f / reciprocal_counts[i]);
        float stddev = std::sqrt(group_stats[i].m2 / group_stats[i].count);
        float scale = 1.0f / (stddev + epsilon);

        for (size_t j = 0; j < group_size; ++j) {
            size_t idx = i * group_size + j;
            output[idx] = (data[idx] - group_stats[i].mean) * scale;
        }
    }
}

} // namespace welford
} // namespace tt