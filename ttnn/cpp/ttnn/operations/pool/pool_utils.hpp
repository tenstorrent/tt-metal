// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
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

uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type,
    uint32_t kernel_h,
    uint32_t kernel_w,
    std::optional<int32_t> divisor_override = std::nullopt,
    std::optional<uint32_t> in_h = std::nullopt,
    std::optional<uint32_t> in_w = std::nullopt,
    std::optional<uint32_t> out_h = std::nullopt,
    std::optional<uint32_t> out_w = std::nullopt,
    std::optional<uint32_t> stride_h = std::nullopt,
    std::optional<uint32_t> stride_w = std::nullopt,
    std::optional<bool> ceil_mode = std::nullopt,
    std::optional<uint32_t> ceil_w = std::nullopt,
    std::optional<uint32_t> out_x = std::nullopt,
    std::optional<uint32_t> out_y = std::nullopt,
    std::optional<uint32_t> out_nhw_per_core = std::nullopt,
    std::vector<uint32_t>* sinchronization_indexes = nullptr,
    std::vector<uint32_t>* scalars = nullptr);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
