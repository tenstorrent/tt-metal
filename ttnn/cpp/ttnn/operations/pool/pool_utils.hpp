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
    uint32_t start;
    uint32_t value;
    uint32_t end;
};

std::vector<ScalarInfo> get_bf16_avg_pool_config_scalars(
    Pool2DType pool_type,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    bool ceil_mode,
    uint32_t ceil_h,
    uint32_t ceil_w,
    uint32_t out_x,
    uint32_t out_y,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t out_nhw_per_core,
    std::optional<int32_t> divisor_override);

uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type, uint32_t kernel_h, uint32_t kernel_w, std::optional<int32_t> divisor_override);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
