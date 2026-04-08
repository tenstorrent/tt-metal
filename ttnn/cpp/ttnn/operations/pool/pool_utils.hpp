// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <string>
#include <map>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::pool {

enum class Pool2DType {
    MAX_POOL2D = 0,
    AVG_POOL2D = 1,
};

std::map<std::string, std::string> get_defines(Pool2DType pool_type);

uint32_t get_aligned_stick_size(const ttnn::Shape& shape, const Tensor& tensor);

}  // namespace ttnn::operations::pool
