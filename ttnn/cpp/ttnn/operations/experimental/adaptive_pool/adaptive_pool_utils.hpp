// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <cstdint>

namespace ttnn {
namespace operations::experimental::adaptive_pool {

struct AdaptiveRegion {
    uint32_t output_h;
    uint32_t output_w;
    uint32_t start_h;
    uint32_t end_h;
    uint32_t start_w;
    uint32_t end_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
};
struct AdaptivePoolingParams {
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding;  // [pad_top, pad_bottom, pad_left, pad_right]
};

AdaptivePoolingParams calculate_adaptive_pool_params(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w);

void validate_adaptive_pool_feasibility(uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w);

// Helper functions for stride validation
std::vector<uint32_t> calculate_stride_pattern(uint32_t input_size, uint32_t output_size);
bool are_strides_uniform(const std::vector<uint32_t>& strides);

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn
