// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "row_mean_sub_square_reduce.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::row_mean_sub_square_reduce {

using namespace tt;
using namespace tt::tt_metal;

ttnn::Tensor ExecuteRowMeanSubSquareReduce::invoke(
    const ttnn::Tensor& input, std::optional<DataType> output_dtype, const std::optional<MemoryConfig>& memory_config) {
    // Call the primitive device operation
    // Unwrap optional memory_config, defaulting to input tensor's memory config
    return ttnn::prim::row_mean_sub_square_reduce(input, output_dtype, memory_config.value_or(input.memory_config()));
}

}  // namespace ttnn::operations::row_mean_sub_square_reduce
