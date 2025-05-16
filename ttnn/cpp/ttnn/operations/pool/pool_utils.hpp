// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
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

std::vector<uint32_t> get_bf16_pool_scalar(
    Pool2DType pool_type,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    bool ceil_mode,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t ceil_w,
    uint32_t out_stick_x_start,
    uint32_t out_stick_y_start,
    std::vector<uint32_t>& sinchronization_indexes,
    std::optional<uint32_t> out_nhw_per_core,
    std::optional<int32_t> divisor_override);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
