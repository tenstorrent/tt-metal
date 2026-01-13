// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/row_mean_sub_square_reduce_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace row_mean_sub_square_reduce {

struct ExecuteRowMeanSubSquareReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        std::optional<DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace row_mean_sub_square_reduce
}  // namespace operations

// Register the operation
constexpr auto row_mean_sub_square_reduce = ttnn::register_operation<
    "ttnn::row_mean_sub_square_reduce",
    ttnn::operations::row_mean_sub_square_reduce::ExecuteRowMeanSubSquareReduce>();

}  // namespace ttnn
