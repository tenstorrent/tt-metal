// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <cstdint>

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

/**
 * @brief Returns the neutral policy value for a given reduce operation.
 *
 * Different reduction operations require different neutral (identity) values
 * for padding to ensure correct results:
 * - 0 (Zero): For sum/mean operations (adding 0 doesn't change the sum)
 * - 1 (NegInf): For max operations (max with -inf gives the original value)
 * - 2 (PosInf): For min operations (min with +inf gives the original value)
 *
 * @param math_op The reduction operation type
 * @return uint32_t The neutral policy value (0=Zero, 1=NegInf, 2=PosInf)
 */
inline uint32_t get_neutral_policy(tt::tt_metal::ReduceOpMath math_op) {
    switch (math_op) {
        case tt::tt_metal::ReduceOpMath::MAX: return 1;  // NegInf
        case tt::tt_metal::ReduceOpMath::MIN: return 2;  // PosInf
        default: return 0;                               // Zero for SUM
    }
}

}  // namespace ttnn::operations::reduction
