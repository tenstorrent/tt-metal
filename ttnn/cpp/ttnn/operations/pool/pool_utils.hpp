// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <optional>
#include <vector>

namespace ttnn::operations::pool {

enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

struct ScalarInfo {
    // Scalar Info is used to store the information abpou the scalar used in avg pool op
    // start and end refer to indices of the output stick core is calculating.
    // These are directly mapped to the for loop that can be found in reader and compute kernel of the pool op
    // for (uint32_t i = 0; i < nsticks_per_core; ++i), start is first stick which should be reduced and multiplied by
    // scalar value, end is the first stick which should not be reduced and multiplied by scalar value. So the interval
    // [start, end) is the range of sticks that should be reduced and multiplied by scalar value.
    uint32_t start;
    uint32_t value;
    uint32_t end;
};

struct AvgPoolConfig {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t stride_h;
    uint32_t stride_w;
    bool ceil_mode;
    uint32_t ceil_h;
    uint32_t ceil_w;
    uint32_t pad_h;
    uint32_t pad_w;
};

std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    AvgPoolConfig config, uint32_t output_stick_x, uint32_t output_stick_y, uint32_t num_of_elements_per_core);

uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type, uint32_t kernel_h, uint32_t kernel_w, std::optional<int32_t> divisor_override);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
