#pragma once

#include <cstdint>
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::stats {

struct TwoPassStatsConfig {
    bool use_l1_replay;
    uint32_t tile_size;
    uint32_t num_tiles;
    bool is_population;
};

void compute_shifted_mean_fp32(const float* input, float* mean, uint32_t size);
void compute_variance_fp32(const float* input, float mean, float* var, uint32_t size, bool population);
bool should_use_two_pass_stats(const ttnn::Tensor& input, uint32_t reduction_dim_size);

} // namespace tt::tt_metal::stats