// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"

namespace ttnn::operations::pool {

tt::tt_metal::ReduceOpMath get_reduce_op(Pool2DType pool_type) {
    tt::tt_metal::ReduceOpMath reduce_op_math;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: reduce_op_math = tt::tt_metal::ReduceOpMath::MAX; break;
        case Pool2DType::AVG_POOL2D: reduce_op_math = tt::tt_metal::ReduceOpMath::SUM; break;
    }
    return reduce_op_math;
}

// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_scalar(Pool2DType pool_type, uint32_t kernel_size_hw) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = 1.; break;
        case Pool2DType::AVG_POOL2D: value = 1. / (float)kernel_size_hw; break;
    }
    return bfloat16(value).to_packed();
}

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
        case Pool2DType::AVG_POOL2D: value = 0.; break;
    }
    return bfloat16(value).to_packed();
}
}  // namespace ttnn::operations::pool
