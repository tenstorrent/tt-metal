
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

enum class ReduceOpMath { SUM, MAX, MIN };

enum class ReduceOpDim { H, W, HW };

enum class ReduceOpParallelizationStrategy { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW };

}  // namespace tt::tt_metal

namespace ttnn::prim {

tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensors, tt::tt_metal::ReduceOpDim reduce_dim);

}  // namespace ttnn::prim

namespace ttnn::operations::reduction {

inline uint32_t get_neutral_policy(tt::tt_metal::ReduceOpMath math_op) {
    switch (math_op) {
        case tt::tt_metal::ReduceOpMath::MAX: return 1;
        case tt::tt_metal::ReduceOpMath::MIN: return 2;
        default: return 0;
    }
}

inline bool supports_native_reduce_padding(const tt::tt_metal::DataType dtype, const bool negate) {
    return !negate && (dtype == tt::tt_metal::DataType::FLOAT32 || dtype == tt::tt_metal::DataType::BFLOAT16);
}

inline bool supports_native_reduce_padding(const tt::DataFormat data_format, const bool negate) {
    return !negate && (data_format == tt::DataFormat::Float32 || data_format == tt::DataFormat::Float16_b);
}

}  // namespace ttnn::operations::reduction
