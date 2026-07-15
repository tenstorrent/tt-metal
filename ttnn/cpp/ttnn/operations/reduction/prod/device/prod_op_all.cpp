// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_op_all.hpp"
#include "prod_all_device_operation.hpp"

#include <ttnn/operations/functions.hpp>

namespace tt::operations::primary {

tt::tt_metal::Tensor prod_all(const tt::tt_metal::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    tt::tt_metal::Tensor result = ttnn::prim::prod_all(input, output_mem_config);
    switch (result.dtype()) {
        case tt::tt_metal::DataType::FLOAT32:
            return ttnn::prod_result_computation_WH_B0<float>(
                result, result.dtype(), result.layout(), result.device(), output_mem_config);
        case tt::tt_metal::DataType::BFLOAT16:
            return ttnn::prod_result_computation_WH_B0<bfloat16>(
                result, result.dtype(), result.layout(), result.device(), output_mem_config);
        default:
            TT_THROW(
                "Error - unexpected prod_all result data type, expected BFLOAT16 or FLOAT32 but got {}.",
                result.dtype());
    }
}

}  // namespace tt::operations::primary
