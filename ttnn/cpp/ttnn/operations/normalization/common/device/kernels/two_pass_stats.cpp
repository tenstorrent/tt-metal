#include "two_pass_stats.hpp"
#include <cmath>

namespace tt::tt_metal::stats {

void compute_shifted_mean_fp32(const float* input, float* mean, uint32_t size) {
    if (size == 0) {
        *mean = 0.0f;
        return;
    }
    float shift = input[0];
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += (input[i] - shift);
    }
    *mean = shift + (sum / static_cast<float>(size));
}

void compute_variance_fp32(const float* input, float mean, float* var, uint32_t size, bool population) {
    if (size == 0 || (!population && size == 1)) {
        *var = 0.0f;
        return;
    }
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }
    float divisor = population ? static_cast<float>(size) : static_cast<float>(size - 1);
    *var = sum_sq / divisor;
}

bool should_use_two_pass_stats(const ttnn::Tensor& input, uint32_t reduction_dim_size) {
    if (input.dtype() == ttnn::DataType::FLOAT32 && reduction_dim_size >= 1024) {
        return true;
    }
    return false;
}

} // namespace tt::tt_metal::stats