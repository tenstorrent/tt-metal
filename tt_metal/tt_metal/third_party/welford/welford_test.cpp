#include "tt_metal/third_party/welford/welford.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace tt {
namespace welford {

TEST(WelfordTest, EqualCountMerge) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> reciprocal_counts = {0.5f, 0.5f, 0.5f};
    EqualCountWelfordMerge result;

    equal_count_welford_merge(data, result, reciprocal_counts);

    // Expected values calculated manually
    float expected_mean = 3.5f;
    float expected_m2 = 17.5f;
    float expected_count = 6.0f;

    EXPECT_NEAR(result.mean, expected_mean, 1e-5f);
    EXPECT_NEAR(result.m2, expected_m2, 1e-5f);
    EXPECT_NEAR(result.count, expected_count, 1e-5f);
}

TEST(WelfordTest, GroupNormEqualCount) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> output(data.size());
    std::vector<float> reciprocal_counts = {0.5f, 0.5f, 0.5f};
    float epsilon = 1e-5f;

    groupnorm_equal_count_welford(data, output, reciprocal_counts, epsilon);

    // Expected values calculated manually
    std::vector<float> expected_output = {
        -1.22474f, -0.408248f, 0.408248f, 0.408248f, 1.22474f, 2.04124f
    };

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
    }
}

} // namespace welford
} // namespace tt