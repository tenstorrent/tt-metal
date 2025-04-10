// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string>
#include <map>

namespace ttnn::operations::pool {

enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

uint32_t get_bf16_pool_scalar(Pool2DType pool_type, uint32_t kernel_size_hw);
uint32_t get_bf16_pool_init_value(Pool2DType pool_type);
std::map<std::string, std::string> get_defines(Pool2DType pool_type);

}  // namespace ttnn::operations::pool
